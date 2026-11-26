"""Build the reviewed detection candidate from the local Foundation Reliability audit."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import yaml

from audit_foundation_datasets import ALIASES, NAMES, digest, source_keys
from pitchlens.detection_data import read_labels
from verify_detection_crop import CropVerifier

V10, OBJECTS, WRKPLACE, IFAP, EXCLUDED = NAMES[:5]
CLASSES = ["player", "goalkeeper", "referee", "ball"]
PARTITIONS = ("train", "val", "test", "ifap")
NEAR_DISTANCE = 6


@dataclass
class Image:
    source: str
    path: Path
    evidence: dict
    labels: tuple
    label_sha256: str
    group: str
    proposed: str
    status: str = ""
    related: str = ""

    @property
    def key(self):
        return f"{self.source}/{self.evidence['path']}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def checked_frozen(frozen, weights, config):
    manifest = frozen / "manifest.json"
    if digest(manifest) != config["frozen_manifest_sha256"] or digest(weights) != config["frozen_weights_sha256"]:
        raise ValueError("Frozen dataset manifest or checkpoint does not match the approved baseline")
    rows = read_json(manifest)
    history = read_json(frozen / "report.json")["groups"]
    for row in rows:
        prefix = row["frame_id"].split("_")[0]
        if history[prefix] != {"group": row["group"], "split": row["split"]}:
            raise ValueError(f"Frozen split disagreement: {row['source']}")
        if row["split"] == "test":
            image = frozen / row["output"]
            label = image.parent.parent / "labels" / (image.stem + ".txt")
            if digest(image) != row["sha256"] or digest(label) != row["label_sha256"]:
                raise ValueError(f"Frozen test bytes changed: {image}")
    return history, rows


def read_inventory(raw, audit, legacy, config):
    records, attribution = [], {}
    for name in NAMES[:5]:
        source_root = raw / name
        report = read_json(audit / f"{name}.json")
        metadata = yaml.safe_load((source_root / "data.yaml").read_text(encoding="utf-8"))
        if metadata != report["config"] or metadata["roboflow"]["license"] != "CC BY 4.0":
            raise ValueError(f"Source metadata differs from the audit: {name}")
        attribution[name] = metadata["roboflow"]
        classes = [ALIASES.get(label, label) for label in metadata["names"]]
        expected_paths = {source_root / row["path"] for row in report["rows"]}
        expected_labels = {source_root / row["label"] for row in report["rows"]}
        if (expected_paths != set(source_root.glob("*/images/*.jpg"))
                or expected_labels != set(source_root.glob("*/labels/*.txt"))
                or len(expected_paths) != len(report["rows"])):
            raise ValueError(f"Extracted members differ from the audit: {name}")
        for row in report["rows"]:
            path = source_root / row["path"]
            label_path = source_root / row["label"]
            if digest(path) != row["sha256"] or source_keys(path.stem) != (row["stem"], row["clip"], row["group"]):
                raise ValueError(f"Image differs from the audit: {path}")
            width, height = map(int, row["resolution"].split("x"))
            labels, _, _, _ = read_labels(label_path, classes, CLASSES, ["corner", "goal"], width, height)
            if not labels:
                raise ValueError(f"No retained labels: {path}")
            raw_counts = Counter(classes[int(line.split()[0])] for line in label_path.read_text(encoding="utf-8").splitlines()
                                 if line.strip())
            if dict(raw_counts) != row["classes"]:
                raise ValueError(f"Class counts differ from the audit: {label_path}")
            provenance = row["group"]
            if name in (V10, OBJECTS):
                group = "legacy/" + legacy[provenance]["group"]
                fold = legacy[provenance]["split"]
                proposed = "frozen" if fold == "test" else fold
            elif name == WRKPLACE:
                # Some numbered scouting clips show different venues. Only map the
                # visually confirmed aliases in the exclusion ledger.
                group = config["wrkplace_reviewed_aliases"].get(provenance, f"wrkplace/{provenance}")
                proposed = config["wrkplace_groups"].get(provenance, "quarantine")
            else:
                group = f"{name}/{provenance or row['stem']}"
                proposed = "ifap" if name == IFAP else "excluded"
            records.append(Image(name, path, row, labels, digest(label_path), group, proposed))
    return records, attribution


def exact_conflicts(records):
    by_pixels = defaultdict(list)
    for record in records:
        by_pixels[record.evidence["pixel_sha256"]].append(record)
    for copies in by_pixels.values():
        if any(copy.labels != copies[0].labels for copy in copies):
            for copy in copies:
                copy.status = "conflicting_exact_annotations"
                copy.related = next(other.key for other in copies if other.labels != copy.labels)


def related_picture(left, right):
    a, b = left.evidence, right.evidence
    return (a["sha256"] == b["sha256"] or a["pixel_sha256"] == b["pixel_sha256"]
            or (a["phash"] ^ b["phash"]).bit_count() <= NEAR_DISTANCE)


def verify_group_boundaries(records, frozen_rows):
    """Fail closed if audit evidence links allowed partitions or the frozen group."""
    known = [record for record in records if record.proposed in (*PARTITIONS, "frozen")]
    protected = [record for record in known if record.proposed == "frozen"]
    hash_set = {record.evidence["sha256"] for record in protected}
    frame_set = {record.evidence["stem"] for record in protected}
    for row in frozen_rows:
        if row["split"] == "test" and (row["sha256"] not in hash_set or row["frame_id"] not in frame_set):
            raise ValueError(f"Frozen test frame missing from its protected source group: {row['source']}")
    for index, left in enumerate(known):
        for right in known[index + 1:]:
            if left.proposed != right.proposed and (
                left.group == right.group or left.evidence["stem"] == right.evidence["stem"]
                or related_picture(left, right)
            ):
                raise ValueError(f"Cross-partition provenance requires review: {left.key} / {right.key}")
    # New broadcasts must be separate even from the disabled detection exports.
    blocked = [record for record in records if record.proposed not in PARTITIONS or record.source in (V10, OBJECTS)]
    for left in known:
        if left.source != WRKPLACE:
            continue
        for right in blocked:
            if left.evidence["stem"] == right.evidence["stem"] or related_picture(left, right):
                raise ValueError(f"Wrkplace source overlap requires review: {left.key} / {right.key}")


def frame_number(record):
    return int(record.evidence["stem"].rsplit("_frame", 1)[1])


def screen_wrkplace(records, config):
    clips = defaultdict(list)
    for record in records:
        if record.source == WRKPLACE and record.proposed in PARTITIONS and not record.status:
            if record.evidence["classes"].get("player", 0) < config["wrkplace_min_players"]:
                record.status = "below_wide_view_player_floor"
            else:
                clips[record.evidence["clip"]].append(record)
    for clip, rows in clips.items():
        rows.sort(key=frame_number)
        limit = config["wrkplace_train_cap_per_clip"]
        if rows[0].proposed == "train" and len(rows) > limit:
            positions = {i * (len(rows) - 1) // (limit - 1) for i in range(limit)}
            for i, record in enumerate(rows):
                if i not in positions:
                    record.status = "training_clip_cap"
        for record in rows:
            if frame_number(record) in config["wrkplace_camera_exclusions"].get(clip, ()):
                record.status = "camera_screen"


def select(records, config):
    exact_conflicts(records)
    screen_wrkplace(records, config)
    originals = {record.evidence["stem"]: record for record in records if record.source == V10}
    if len(originals) != sum(record.source == V10 for record in records):
        raise ValueError("Multiple Players v10 originals in one frame family")
    kept = defaultdict(list)
    accepted_crops = set()
    verifier = CropVerifier()
    ordered = sorted(records, key=lambda record: (NAMES.index(record.source), record.group, record.evidence["stem"], record.key))
    for record in ordered:
        if record.status:
            continue
        if record.proposed not in PARTITIONS:
            record.status = {"frozen": "frozen_group", "quarantine": "unresolved_scouting_source",
                             "excluded": "excluded_export"}[record.proposed]
            continue
        if record.source == OBJECTS and record.proposed != "train":
            record.status = "holdout_provider_transform"
            continue
        for other in kept[record.proposed]:
            if related_picture(other, record):
                record.status = "exact_or_perceptual_duplicate"
                record.related = other.key
                break
        if record.status:
            continue
        if record.source == OBJECTS:
            stem = record.evidence["stem"]
            original = originals[stem]
            record.related = original.key
            if original.status != "retained":
                record.status = "original_not_retained"
                continue
            if stem in accepted_crops:
                record.status = "additional_crop_same_frame"
                continue
            outcome = verifier.check(original.path, record.path, original.labels, record.labels)
            if outcome != "compatible":
                record.status = outcome
                continue
            accepted_crops.add(stem)
        record.status = "retained"
        kept[record.proposed].append(record)
    return kept


def materialize(output, kept, records, attribution, inputs):
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    stats = {}
    for partition in PARTITIONS:
        rows = kept[partition]
        counts = Counter(CLASSES[box[0]] for row in rows for box in row.labels)
        if set(counts) != set(CLASSES):
            raise ValueError(f"Missing class coverage in {partition}")
        split_root = output / partition
        (split_root / "images").mkdir(parents=True)
        (split_root / "labels").mkdir()
        manifest = []
        for record in rows:
            sha = record.evidence["sha256"]
            image = split_root / "images" / (sha + ".jpg")
            label = split_root / "labels" / (sha + ".txt")
            shutil.copyfile(record.path, image)
            data = "".join(f"{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for c, x, y, w, h in record.labels).encode("ascii")
            label.write_bytes(data)
            if digest(image) != sha or digest(label) != hashlib.sha256(data).hexdigest():
                raise OSError(f"Candidate copy verification failed: {image}")
            manifest.append({
                "source": record.key, "split": partition, "group": record.group,
                "frame_id": record.evidence["stem"], "sha256": sha,
                "pixel_sha256": record.evidence["pixel_sha256"],
                "phash": f"{record.evidence['phash']:016x}", "raw_label_sha256": record.label_sha256,
                "output": image.relative_to(output).as_posix(), "label_sha256": digest(label),
                "related_original": record.related or None,
            })
        stats[partition] = {
            "images": len(rows), "sources": dict(Counter(row.source for row in rows)),
            "groups": dict(Counter(row.group for row in rows)), "boxes": dict(counts),
            "images_with_class": dict(Counter(CLASSES[c] for row in rows for c in {box[0] for box in row.labels})),
        }
        manifests[partition] = manifest

    write_json(output / "manifest.json", manifests["train"] + manifests["val"])
    write_json(output / "reserved-test.json", manifests["test"])
    write_json(output / "ifap-manifest.json", manifests["ifap"])
    write_json(output / "provenance.json", [
        {"source": row.key, "source_group": row.evidence["group"], "group": row.group, "clip": row.evidence["clip"],
         "frame_id": row.evidence["stem"], "sha256": row.evidence["sha256"],
         "pixel_sha256": row.evidence["pixel_sha256"], "phash": f"{row.evidence['phash']:016x}",
         "raw_label_sha256": row.label_sha256, "proposed_partition": row.proposed,
         "decision": row.status, "related_source": row.related or None}
        for row in sorted(records, key=lambda record: record.key)
    ])
    names = "names: [player, goalkeeper, referee, ball]\n"
    (output / "data.yaml").write_text("train: train/images\nval: val/images\n" + names, encoding="utf-8")
    (output / "ifap.yaml").write_text("train: []\nval: ifap/images\n" + names, encoding="utf-8")
    report = {
        "inputs": inputs, "attribution": attribution, "partitions": stats,
        "decisions": {source: dict(Counter(row.status for row in records if row.source == source)) for source in NAMES[:5]},
        "integrity": {
            "phash_distance": NEAR_DISTANCE,
            "cross_partition_group_frame_file_pixel_or_near_links": 0,
            "retained_file_pixel_or_near_duplicate_pairs": 0,
            "frozen_test_manifest_and_payload_verified": True,
            "reserved_test_manifest_sha256": digest(output / "reserved-test.json"),
            "development_manifest_sha256": digest(output / "manifest.json"),
            "ifap_manifest_sha256": digest(output / "ifap-manifest.json"),
            "test_available_in_development_yaml": False,
        },
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("raw", "audit", "frozen", "weights", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "configs/detection_candidate_v1.json")
    args = parser.parse_args()
    raw, audit, frozen, output = (path.resolve() for path in (args.raw, args.audit, args.frozen, args.output))
    for protected in (raw, audit, frozen):
        if protected == output or protected in output.parents or output in protected.parents:
            raise ValueError(f"Output must be separate from {protected}")
    if output.exists():
        raise FileExistsError(output)

    config = read_json(args.config)
    legacy, frozen_rows = checked_frozen(frozen, args.weights, config)
    records, attribution = read_inventory(raw, audit, legacy, config)
    verify_group_boundaries(records, frozen_rows)
    cv2.setNumThreads(2)
    cv2.setRNGSeed(42)
    kept = select(records, config)
    inputs = {
        "config_sha256": digest(args.config), "frozen_report_sha256": digest(frozen / "report.json"),
        "frozen_manifest_sha256": digest(frozen / "manifest.json"), "weights_sha256": digest(args.weights),
        "audit_inventory_sha256": {name: digest(audit / f"{name}.json") for name in NAMES[:5]},
        "opencv": cv2.__version__,
    }
    report = materialize(output, kept, records, attribution, inputs)
    checked_frozen(frozen, args.weights, config)
    print(json.dumps({key: value["images"] for key, value in report["partitions"].items()}))


if __name__ == "__main__":
    main()
