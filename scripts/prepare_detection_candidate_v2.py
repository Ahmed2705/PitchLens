"""Apply the completed Candidate v1 review to a new development-only dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

from audit_foundation_datasets import digest, image_features
from prepare_detection_candidate import NEAR_DISTANCE, write_json
from review_candidate_annotations import (
    CLASSES, EXPORT_COMPLETE, MANIFEST_SHA256, Review, parse_yolo,
)

PROVENANCE_SHA256 = "af549ebcf90416d28f0f094a5aa35e3b7e6439f7bd4245b5c88323f2dd3190f5"
EXPORT_MANIFEST_SHA256 = "89ed7ee1a6aeaf90e06bc4f52d1328025b6caa2a6334466b00ee591a339a9fba"
CANDIDATE_MANIFEST_SHA256 = "eda48473212a3f4911b56ae9a061e1f938d8ccccc4961345c42bf62f6f666d1e"
EXPECTED_SPLITS = {"train": 371, "val": 57}
EXPECTED_REVIEW = {"train": 113, "val": 20}
EXPECTED_BOXES = {"train": dict(zip(CLASSES, (6651, 229, 722, 341))),
                  "val": dict(zip(CLASSES, (977, 37, 120, 54)))}
DATA_YAML = "train: train/images\nval: val/images\nnames: [player, goalkeeper, referee, ball]\n"


def label_path(row: dict) -> str:
    return f"{row['split']}/labels/{row['sha256']}.txt"


def verified_boxes(data: bytes) -> list[list]:
    """Check normalized values before pixel conversion and rounding."""
    boxes = parse_yolo(data)
    for line in data.splitlines():
        _, x, y, width, height = line.split()
        if (not all(0 <= float(value) <= 1 for value in (x, y))
                or not all(0 < float(value) <= 1 for value in (width, height))):
            raise ValueError("YOLO coordinates must be normalized with positive dimensions")
    if not boxes or len({tuple(box) for box in boxes}) != len(boxes):
        raise ValueError("Empty labels or duplicate boxes")
    return boxes


def verified_json(path: Path, expected: str):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Approved manifest hash mismatch: {path}")
    return json.loads(data)


def approved_candidate_manifest(rows: list[dict]) -> bytes:
    """Bind the full corrected label payload to the historical Candidate v2 manifest."""
    data = (json.dumps(rows, indent=2) + "\n").encode("utf-8")
    if hashlib.sha256(data).hexdigest() != CANDIDATE_MANIFEST_SHA256:
        raise ValueError("Candidate v2 manifest differs from the approved image and label hashes")
    return data


def verify_export(review: Review, export: Path) -> tuple[dict, dict[str, bytes]]:
    marker = export / EXPORT_COMPLETE
    if not marker.is_file() or marker.read_bytes():
        raise ValueError("Missing or invalid export completion marker")
    manifest = verified_json(export / "review.json", EXPORT_MANIFEST_SHA256)
    if (manifest["manifest_sha256"] != MANIFEST_SHA256
            or manifest["classes"] != list(CLASSES)
            or manifest["progress_revision"] != 423):
        raise ValueError("Export does not match the approved correction snapshot")

    labels, counts = {}, Counter()
    expected_files = {"review.json", EXPORT_COMPLETE}
    for record in manifest["frames"]:
        key = record["frame_id"]
        if key not in review.frames or key in labels or record["status"] != "reviewed":
            raise ValueError(f"Unexpected, repeated or unreviewed export entry: {key}")
        path = review.frames[key]["label_path"]
        if record["label_path"] != path:
            raise ValueError(f"Wrong label path for reviewed frame: {key}")
        data = (export / path).read_bytes()
        boxes = verified_boxes(data)
        changed = data != review.labels[key]
        if record["changed"] is not changed or (changed and boxes == review.frames[key]["boxes"]):
            raise ValueError(f"Incorrect change flag for reviewed frame: {key}")
        labels[key] = data
        expected_files.add(path)
        counts.update({path.split("/", 1)[0]: 1, "boxes": len(boxes), "changed": int(changed)})

    actual_files = {path.relative_to(export).as_posix() for path in export.rglob("*") if path.is_file()}
    if (labels.keys() != review.frames.keys() or actual_files != expected_files
            or counts != {**EXPECTED_REVIEW, "boxes": 2106, "changed": 15}):
        raise ValueError(f"Unexpected export contents or totals: {dict(counts)}")
    return manifest, labels


def verify_boundaries(rows: list[dict], provenance: list[dict]) -> None:
    def related(left: dict, right: dict) -> bool:
        return (left["sha256"] == right["sha256"]
                or left["pixel_sha256"] == right["pixel_sha256"]
                or (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count() <= NEAR_DISTANCE)

    def crosses(left: dict, right: dict) -> bool:
        return any(left[key] == right[key] for key in ("group", "frame_id"))

    by_source = {row["source"]: row for row in provenance}
    protected = [row for row in provenance if row["proposed_partition"] in {"test", "ifap", "frozen"}]
    for row in rows:
        prior = by_source[row["source"]]
        if (prior["proposed_partition"] != row["split"]
                or any(prior[key] != row[key] for key in (
                    "group", "frame_id", "sha256", "pixel_sha256", "phash", "raw_label_sha256"
                ))):
            raise ValueError(f"Development entry differs from recorded source: {row['source']}")
        if row["related_original"]:
            original = by_source[row["related_original"]]
            if original["proposed_partition"] != row["split"] or original["group"] != row["group"]:
                raise ValueError(f"Crop crosses a source boundary: {row['source']}")
        for other in protected:
            if crosses(row, other) or related(row, other):
                raise ValueError(f"Development overlaps protected provenance: {row['source']}")
    for left, right in combinations(rows, 2):
        if related(left, right) or (left["split"] != right["split"] and crosses(left, right)):
            raise ValueError(f"Duplicate or split overlap: {left['source']} / {right['source']}")


def build(source: Path, export: Path, output: Path) -> dict:
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in (source, export)):
        raise ValueError("Use a separate destination for the new dataset")
    if output.exists():
        raise FileExistsError(output)
    provenance = verified_json(source / "provenance.json", PROVENANCE_SHA256)
    # The review loader reuses the editor's original frame selection and image
    # identities. Point it at the export so it does not load or modify the draft.
    review = Review(source, export)
    export_manifest, corrections = verify_export(review, export)
    rows = verified_json(source / "manifest.json", MANIFEST_SHA256)
    if Counter(row["split"] for row in rows) != EXPECTED_SPLITS:
        raise ValueError("Unexpected Candidate v1 development split")
    verify_boundaries(rows, provenance)
    payload = {}
    reviewed = []
    stats = {split: Counter() for split in EXPECTED_SPLITS}
    for row in rows:
        image_name = f"{row['split']}/images/{row['sha256']}.jpg"
        label_name = label_path(row)
        if image_name != row["output"] or image_name in payload:
            raise ValueError(f"Unexpected image path: {row['output']}")
        image = (source / image_name).read_bytes()
        prior = (source / label_name).read_bytes()
        decoded, phash, _ = image_features(image)
        height, width = decoded.shape[:2]
        pixel_hash = hashlib.sha256(f"{width}x{height}:".encode() + decoded.tobytes()).hexdigest()
        if (hashlib.sha256(image).hexdigest() != row["sha256"]
                or pixel_hash != row["pixel_sha256"] or f"{phash:016x}" != row["phash"]
                or hashlib.sha256(prior).hexdigest() != row["label_sha256"]):
            raise ValueError(f"Original image or label hash mismatch: {image_name}")
        label = corrections.get(row["frame_id"], prior)
        if row["frame_id"] in corrections:
            if (label_name != review.frames[row["frame_id"]]["label_path"]
                    or not row["source"].startswith("wrkplace-football-detection/")):
                raise ValueError(f"Correction maps to the wrong source: {row['source']}")
            reviewed.append({**row, "changed": label != prior,
                             "corrected_label_sha256": hashlib.sha256(label).hexdigest()})
        try:
            boxes = verified_boxes(label)
        except ValueError as exc:
            raise ValueError(f"Invalid label file: {label_name}") from exc
        stats[row["split"]].update(CLASSES[box[0]] for box in boxes)
        payload.update({image_name: image, label_name: label})
        row["label_sha256"] = hashlib.sha256(label).hexdigest()
    if len(reviewed) != len(corrections):
        raise ValueError("Reviewed frames must each map to exactly one development entry")
    if stats != EXPECTED_BOXES:
        raise ValueError(f"Unexpected Candidate v2 class totals: {stats}")
    manifest_bytes = approved_candidate_manifest(rows)

    output.mkdir(parents=True, exist_ok=False)
    for name, data in payload.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        if destination.read_bytes() != data:
            raise OSError(f"Dataset write verification failed: {name}")
    (output / "manifest.json").write_bytes(manifest_bytes)
    write_json(output / "corrections.json", reviewed)
    (output / "data.yaml").write_text(DATA_YAML, encoding="utf-8")
    report = {
        "dataset": "detection-candidate-v2",
        "parent_manifest_sha256": MANIFEST_SHA256,
        "parent_provenance_sha256": PROVENANCE_SHA256,
        "export": export.name,
        "export_manifest_sha256": EXPORT_MANIFEST_SHA256,
        "export_revision": export_manifest["progress_revision"],
        "manifest_sha256": digest(output / "manifest.json"),
        "corrections_manifest_sha256": digest(output / "corrections.json"),
        "data_yaml_sha256": digest(output / "data.yaml"),
        "partitions": {split: {"images": count, "reviewed": EXPECTED_REVIEW[split],
                               "changed": sum(row["changed"] for row in reviewed if row["split"] == split),
                               "boxes": dict(stats[split])}
                       for split, count in EXPECTED_SPLITS.items()},
        "integrity": {"missing_labels": 0, "duplicate_images_or_near_pairs": 0, "cross_partition_links": 0,
                      "protected_check": "Candidate v1 provenance metadata only",
                      "verified_image_label_pairs": len(rows), "reviewed_boxes": 2106},
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "export", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.source.resolve(), args.export.resolve(), args.output.resolve())
    print(report)


if __name__ == "__main__":
    main()
