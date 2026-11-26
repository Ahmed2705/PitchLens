"""Verify and copy a manifest-pinned detection partition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def linked(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def label_for(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def labels_match(raw: bytes, loaded: dict) -> bool:
    """Check that the loader preserved every class and box, allowing row reordering."""
    rows = np.array([line.split() for line in raw.decode().splitlines() if line.strip()],
                    dtype=np.float32).reshape(-1, 5)
    accepted = np.column_stack((loaded["cls"], loaded["bboxes"]))
    return Counter(map(tuple, rows)) == Counter(map(tuple, accepted))


@dataclass(frozen=True)
class Frame:
    image: Path
    image_sha256: str
    label_sha256: str
    group: str

    def verified_payload(self) -> tuple[bytes, bytes]:
        label = label_for(self.image)
        return (verified_bytes(self.image, self.image_sha256),
                verified_bytes(label, self.label_sha256))

    def snapshot(self, destination: Path) -> Frame:
        image, label = self.verified_payload()
        path = destination / "images" / self.image.name
        path.parent.mkdir(parents=True, exist_ok=True)
        target_label = label_for(path)
        target_label.parent.mkdir(exist_ok=True)
        path.write_bytes(image)
        target_label.write_bytes(label)
        return Frame(path, self.image_sha256, self.label_sha256, self.group)


def verified_bytes(path: Path, expected: str) -> bytes:
    if linked(path) or not path.is_file():
        raise ValueError(f"Missing or linked evaluation file: {path}")
    data = path.read_bytes()
    if sha256(data) != expected:
        raise ValueError(f"Evaluation file SHA-256 mismatch: {path}")
    return data


def tree_files(directory: Path) -> set[Path]:
    """Inventory both trees, including nested and unsupported files."""
    if linked(directory) or not directory.is_dir():
        raise ValueError(f"Missing or linked evaluation directory: {directory}")
    files = set()
    for path in directory.iterdir():
        if linked(path):
            raise ValueError(f"Unexpected linked evaluation path: {path}")
        if path.is_dir():
            files.update(tree_files(path))
        else:
            files.add(path)
    return files


def manifest_frames(root: Path, rows: list[dict], split: str) -> list[Frame]:
    partition = root / split
    if linked(partition):
        raise ValueError(f"Linked evaluation partition: {partition}")
    frames = {}
    for row in rows:
        if row["split"] != split:
            continue
        image_hash, label_hash = row["sha256"], row["label_sha256"]
        if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in (image_hash, label_hash)):
            raise ValueError("Invalid SHA-256 in the evaluation manifest")
        relative = f"{split}/images/{image_hash}.jpg"
        if row["output"] != relative:
            raise ValueError(f"Unexpected manifest image path: {row['output']}")
        path = root / relative
        if path in frames:
            raise ValueError(f"Duplicate evaluation manifest entry: {path}")
        group = row["group"]
        if not isinstance(group, str) or not group:
            raise ValueError(f"Missing source group for {path}")
        frames[path] = Frame(path, image_hash, label_hash, group)
    if not frames:
        raise ValueError(f"Frozen manifest has no {split} frames")
    expected = set(frames) | {label_for(path) for path in frames}
    actual = tree_files(partition / "images") | tree_files(partition / "labels")
    if actual != expected:
        raise ValueError(f"Evaluation files differ from manifest. Missing: {sorted(expected - actual)}. "
                         f"Unexpected: {sorted(actual - expected)}")
    return [frames[path] for path in sorted(frames)]
