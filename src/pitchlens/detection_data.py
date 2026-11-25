"""Prepare the approved detection exports without modifying the originals.

The six-character source prefixes are imperfect provenance identifiers. A small
set of visually similar sources is grouped conservatively, and cross-prefix
perceptual-hash candidates are grouped too. This reduces, but cannot eliminate,
the risk of different prefixes referring to the same match.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_NAME = re.compile(r"^([0-9a-f]{6})_(\d+)_(\d+)_png(?:_jpg)?\.rf\.[0-9a-f]+\.jpg$")


@dataclass(frozen=True)
class Record:
    path: Path
    relative_path: str
    source: str
    source_prefix: str
    frame_id: str
    original: bool
    sha256: str
    phash: int
    width: int
    height: int
    labels: tuple[tuple[int, float, float, float, float], ...]


class Groups:
    """Disjoint source groups; names are stable regardless of merge order."""

    def __init__(self, names: set[str]):
        self.parent = {name: name for name in names}

    def find(self, name: str) -> str:
        while self.parent[name] != name:
            name = self.parent[name]
        return name

    def merge(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def image_phash(gray: np.ndarray) -> int:
    """Return a 64-bit DCT perceptual hash; it is only a similarity heuristic."""
    pixels = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(pixels)[:8, :8].ravel()
    median = float(np.median(low_frequency[1:]))
    return sum(int(value > median) << i for i, value in enumerate(low_frequency))


def read_labels(
    label_path: Path, classes: list[str], names: list[str], excluded: list[str], width: int, height: int
) -> tuple[tuple[tuple[int, float, float, float, float], ...], Counter, int, int]:
    """Validate source annotations, remove excluded classes and map boxes.

    Boxes extending no more than one pixel beyond the edge are clipped. Larger
    errors are rejected rather than silently changing their meaning.
    """
    labels = []
    removed: Counter[str] = Counter()
    clipped = duplicates = 0
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        try:
            source_id = int(fields[0])
            if not 0 <= source_id < len(classes):
                raise ValueError("unknown class ID")
            values = [float(value) for value in fields[1:]]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("non-finite coordinate")
            name = classes[source_id]
            if name in excluded:
                if len(values) < 4 or len(values) % 2 or any(not 0 <= value <= 1 for value in values):
                    raise ValueError("invalid excluded box or polygon")
                removed[name] += 1
                continue
            if name not in names or len(values) != 4:
                raise ValueError("expected a four-coordinate detection box")
            x, y, w, h = values
            if w <= 0 or h <= 0:
                raise ValueError("non-positive box size")
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            if x1 < -1 / width - 1e-9 or x2 > 1 + 1 / width + 1e-9 or y1 < -1 / height - 1e-9 or y2 > 1 + 1 / height + 1e-9:
                raise ValueError("box extends more than one pixel beyond image")
            bounded = (max(0., x1), max(0., y1), min(1., x2), min(1., y2))
            if bounded != (x1, y1, x2, y2):
                clipped += 1
            x1, y1, x2, y2 = bounded
            if x2 <= x1 or y2 <= y1:
                raise ValueError("box lies outside image")
            box = (names.index(name), round((x1 + x2) / 2, 8), round((y1 + y2) / 2, 8), round(x2 - x1, 8), round(y2 - y1, 8))
            if box in labels:
                duplicates += 1
            else:
                labels.append(box)
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{label_path}:{line_number}: {exc}") from exc
    return tuple(sorted(labels)), removed, clipped, duplicates


def scan_sources(raw_root: Path, config: dict) -> tuple[list[Record], dict]:
    """Decode every image, validate every label, and collect hashes and counts."""
    records = []
    removed: Counter[str] = Counter()
    clipped = duplicate_labels = 0
    counts: Counter[str] = Counter()
    for source, spec in config["datasets"].items():
        source_root = raw_root / source
        if not source_root.is_dir():
            raise FileNotFoundError(f"Dataset not found: {source_root}")
        metadata = (source_root / "data.yaml").read_text(encoding="utf-8")
        names_match = re.search(r"^names:\s*(\[[^\n]*\])\s*$", metadata, re.MULTILINE)
        if names_match is None or ast.literal_eval(names_match.group(1)) != spec["classes"]:
            raise ValueError(f"Unexpected source class mapping in {source_root / 'data.yaml'}")
        for split in ("train", "valid", "test"):
            images, labels = source_root / split / "images", source_root / split / "labels"
            if not images.is_dir() or not labels.is_dir():
                raise FileNotFoundError(f"Missing image or label directory in {source_root / split}")
            image_paths = sorted(images.glob("*.jpg"))
            if not image_paths:
                raise ValueError(f"No images in {images}")
            orphans = {p.stem for p in labels.glob("*.txt")} - {p.stem for p in image_paths}
            if orphans:
                raise ValueError(f"Orphan labels in {labels}: {sorted(orphans)[:5]}")
            for image_path in image_paths:
                match = _NAME.fullmatch(image_path.name)
                if not match:
                    raise ValueError(f"Unrecognized provenance filename: {image_path.name}")
                label_path = labels / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(f"Missing label: {label_path}")
                content = image_path.read_bytes()
                gray = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError(f"Unreadable image: {image_path}")
                height, width = gray.shape
                mapped, discarded, nclipped, nduplicates = read_labels(
                    label_path, spec["classes"], config["class_names"], config["exclude_classes"], width, height
                )
                removed.update(discarded)
                clipped += nclipped
                duplicate_labels += nduplicates
                counts[source] += 1
                records.append(Record(image_path, image_path.relative_to(raw_root).as_posix(), source, match[1],
                                      "_".join(match.group(1, 2, 3)), spec["original"], hashlib.sha256(content).hexdigest(),
                                      image_phash(gray), width, height, mapped))
    return records, {"source_images": dict(counts), "removed_annotations": dict(removed),
                     "clipped_boxes": clipped, "duplicate_boxes_removed": duplicate_labels}


def group_sources(records: list[Record], config: dict) -> tuple[Groups, dict]:
    groups = Groups({r.source_prefix for r in records})
    for cluster in config["conservative_source_groups"]:
        if not all(name in groups.parent for name in cluster):
            raise ValueError(f"Unknown source in conservative group: {cluster}")
        for name in cluster[1:]:
            groups.merge(cluster[0], name)
    by_sha: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_sha[record.sha256].append(record)
    conflicts = []
    for copies in by_sha.values():
        for copy in copies[1:]:
            groups.merge(copies[0].source_prefix, copy.source_prefix)
            if copy.labels != copies[0].labels:
                conflicts.append([copies[0].relative_path, copy.relative_path])
    if conflicts:
        raise ValueError(f"Exact duplicate images have conflicting annotations: {conflicts[:5]}")
    threshold = config["phash_distance"]
    near_total = 0
    cross_prefix = []
    for index, left in enumerate(records):
        for right in records[index + 1:]:
            distance = (left.phash ^ right.phash).bit_count()
            if distance <= threshold and left.sha256 != right.sha256:
                near_total += 1
                if left.source_prefix != right.source_prefix:
                    cross_prefix.append({"left": left.relative_path, "right": right.relative_path, "distance": distance})
                    groups.merge(left.source_prefix, right.source_prefix)
    return groups, {"exact_duplicate_groups": sum(len(copies) > 1 for copies in by_sha.values()),
                    "exact_duplicate_extras": sum(len(copies) - 1 for copies in by_sha.values()),
                    "phash_threshold": threshold, "near_duplicate_pairs": near_total,
                    "cross_prefix_near_duplicate_pairs": cross_prefix}


def choose_split(records: list[Record], groups: Groups, seed: int,
                 test_excluded_prefixes: tuple[str, ...] = ()) -> dict[str, str]:
    """Select disjoint held-out source groups targeting 10% of original frames each.

    We use annotation totals only to improve class representation, before any
    training or evaluation. Augmented variants never enter validation or test.
    """
    originals = [r for r in records if r.original]
    source_groups = sorted({groups.find(r.source_prefix) for r in originals})
    if len(source_groups) < 5:
        raise ValueError("Too few independent source groups for a useful split")
    totals = Counter(box[0] for r in originals for box in r.labels)
    sizes = Counter(groups.find(r.source_prefix) for r in originals)
    per_group = {group: Counter(box[0] for r in originals if groups.find(r.source_prefix) == group for box in r.labels)
                 for group in source_groups}
    target = len(originals) * .1
    options = []
    for size in range(1, len(source_groups) - 1):
        for subset in itertools.combinations(source_groups, size):
            count = sum(sizes[group] for group in subset)
            if not target * .5 <= count <= target * 1.5:
                continue
            box_counts = sum((per_group[group] for group in subset), Counter())
            if any(box_counts[category] == 0 for category in range(4)):
                continue
            size_error = abs(count - target) / target
            class_error = sum(abs(box_counts[c] - totals[c] * .1) / max(totals[c] * .1, 1) for c in range(4)) / 4
            options.append((frozenset(subset), size_error + .35 * class_error))
    random.Random(seed).shuffle(options)
    excluded_groups = {groups.find(prefix) for prefix in test_excluded_prefixes}
    candidates = ((a_score + b_score, a, b) for a, a_score in options
                  for b, b_score in options if not a & b and not b & excluded_groups)
    # Prefer more diverse held-out sets when the small number of source groups
    # allows it; then minimize image-count and class-representation errors.
    best = min(candidates, key=lambda item: (-(int(len(item[1]) > 1) + int(len(item[2]) > 1)), item[0]), default=None)
    if best is None:
        raise ValueError("Could not construct disjoint, class-representative held-out source groups")
    _, validation, test = best
    return {group: "val" if group in validation else "test" if group in test else "train" for group in source_groups}


def prepare(raw_root: Path, output: Path, config: dict, seed: int = 42) -> dict:
    """Create a new processed dataset and a detailed reproducibility manifest."""
    raw_root, output = raw_root.resolve(), output.resolve()
    if output == raw_root or raw_root in output.parents or output in raw_root.parents:
        raise ValueError("Processed output must be separate from the raw dataset directory")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    records, report = scan_sources(raw_root, config)
    groups, similarity = group_sources(records, config)
    assignments = choose_split(records, groups, seed, tuple(config.get("test_excluded_source_prefixes", ())))
    excluded_test_frames = set(config.get("test_excluded_frame_ids", ()))
    found_test_frames = {r.frame_id for r in records if r.original and r.frame_id in excluded_test_frames
                         and assignments[groups.find(r.source_prefix)] == "test"}
    if found_test_frames != excluded_test_frames:
        raise ValueError("An explicitly excluded test frame was not found in the assigned test group")
    report.update(similarity)
    report["seed"] = seed
    report["groups"] = {prefix: {"group": groups.find(prefix), "split": assignments[groups.find(prefix)]}
                        for prefix in sorted(groups.parent)}
    report["splits"] = {s: {"images": 0, "original_images": 0, "boxes": {name: 0 for name in config["class_names"]},
                            "images_with_class": {name: 0 for name in config["class_names"]}}
                        for s in ("train", "val", "test")}
    report["excluded_heldout_augmentations"] = 0
    report["excluded_exact_duplicates"] = 0
    report["excluded_previously_inspected_test_frames"] = sorted(excluded_test_frames)
    output.mkdir(parents=True)
    seen = set()
    manifest = []
    for record in records:
        split = assignments[groups.find(record.source_prefix)]
        if split != "train" and not record.original:
            report["excluded_heldout_augmentations"] += 1
            continue
        if split == "test" and record.frame_id in excluded_test_frames:
            continue
        if record.sha256 in seen:
            report["excluded_exact_duplicates"] += 1
            continue
        seen.add(record.sha256)
        image_dir, label_dir = output / split / "images", output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        dest = f"{record.sha256}.jpg"
        shutil.copyfile(record.path, image_dir / dest)
        label_text = "".join(f"{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for c, x, y, w, h in record.labels)
        # Use explicit CRLF bytes to match previously prepared Windows labels
        # while making the saved-byte checksum portable and unambiguous.
        label_bytes = label_text.replace("\n", "\r\n").encode("utf-8")
        (label_dir / f"{record.sha256}.txt").write_bytes(label_bytes)
        stats = report["splits"][split]
        stats["images"] += 1
        stats["original_images"] += int(record.original)
        for category in {box[0] for box in record.labels}:
            stats["images_with_class"][config["class_names"][category]] += 1
        for box in record.labels:
            stats["boxes"][config["class_names"][box[0]]] += 1
        manifest.append({"source": record.relative_path, "frame_id": record.frame_id, "sha256": record.sha256,
                         "phash": f"{record.phash:016x}", "group": groups.find(record.source_prefix),
                         "split": split, "original": record.original, "output": f"{split}/images/{dest}",
                         "label_sha256": hashlib.sha256(label_bytes).hexdigest()})
    (output / "data.yaml").write_text("train: train/images\nval: val/images\ntest: test/images\nnames: [player, goalkeeper, referee, ball]\n", encoding="utf-8", newline="\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report
