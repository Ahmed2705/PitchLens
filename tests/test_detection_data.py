"""Tests for detection annotation validation and leakage-aware splitting."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pitchlens.detection_data import Groups, Record, choose_split, group_sources, prepare, read_labels


class LabelTests(unittest.TestCase):
    def test_maps_classes_removes_polygons_and_clips_small_edge_error(self):
        with tempfile.TemporaryDirectory() as temp:
            label = Path(temp) / "sample.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n1 0 0 1 0 1 1\n2 0.9995 0.5 0.002 0.1\n", encoding="utf-8")
            boxes, removed, clipped, duplicates = read_labels(
                label, ["ball", "goal", "player"], ["player", "goalkeeper", "referee", "ball"], ["goal"], 1000, 1000
            )
            self.assertEqual([box[0] for box in boxes], [0, 3])
            self.assertEqual(removed["goal"], 1)
            self.assertEqual(clipped, 1)
            self.assertEqual(duplicates, 0)
            self.assertLessEqual(boxes[0][1] + boxes[0][3] / 2, 1.00000001)

    def test_invalid_box_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            label = Path(temp) / "bad.txt"
            for content in ("0 0.5 0.5 -0.2 0.1", "0 nan 0.5 0.2 0.1", "0 1 0.5 0.1 0.1", "3 0.5 0.5 0.1 0.1"):
                with self.subTest(content=content):
                    label.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        read_labels(label, ["ball"], ["ball"], [], 1000, 1000)

    def test_duplicate_labels_are_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            label = Path(temp) / "duplicate.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            boxes, _, _, duplicates = read_labels(label, ["ball"], ["ball"], [], 100, 100)
            self.assertEqual(len(boxes), 1)
            self.assertEqual(duplicates, 1)


def record(prefix: str, frame: int, sha: str, phash: int, labels=None) -> Record:
    if labels is None:
        labels = tuple((category, .5, .5, .1, .1) for category in range(4))
    return Record(Path("unused"), f"{prefix}/{frame}", "source", prefix, f"{prefix}_{frame}", True,
                  sha, phash, 100, 100, labels)


class SplitTests(unittest.TestCase):
    def test_split_is_reproducible_and_keeps_groups_disjoint(self):
        records = [record(f"{group:06x}", frame, f"{group}-{frame}", group * 100 + frame)
                   for group in range(20) for frame in range(5)]
        groups = Groups({r.source_prefix for r in records})
        split = choose_split(records, groups, 42)
        self.assertEqual(split, choose_split(records, groups, 42))
        self.assertEqual({name: list(split.values()).count(name) for name in ("train", "val", "test")},
                         {"train": 16, "val": 2, "test": 2})
        excluded = tuple(list(sorted(split))[:4])
        reassigned = choose_split(records, groups, 42, excluded)
        self.assertTrue(all(reassigned[prefix] != "test" for prefix in excluded))

    def test_similarity_merges_cross_source_candidates(self):
        a = record("aaaaaa", 1, "one", 0)
        b = record("bbbbbb", 2, "two", 1)
        groups, report = group_sources([a, b], {"conservative_source_groups": [], "phash_distance": 1})
        self.assertEqual(groups.find("aaaaaa"), groups.find("bbbbbb"))
        self.assertEqual(len(report["cross_prefix_near_duplicate_pairs"]), 1)

    def test_exact_duplicate_with_conflicting_labels_fails(self):
        a = record("aaaaaa", 1, "same", 0)
        b = record("bbbbbb", 2, "same", 0, labels=((0, .5, .5, .1, .1),))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            group_sources([a, b], {"conservative_source_groups": [], "phash_distance": 1})

    def test_raw_and_existing_output_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(ValueError):
                prepare(raw, raw / "processed", {})
            with self.assertRaises(FileExistsError):
                prepare(raw, existing, {})

    def test_previously_inspected_test_frames_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            image = raw / "sample.jpg"
            image.write_bytes(b"sample")
            records = [Record(image, "sample.jpg", "source", "aaaaaa", "frame-1", True,
                              "abc", 0, 100, 100, ((0, .5, .5, .1, .1),)),
                       Record(image, "sample.jpg", "source", "bbbbbb", "frame-2", True,
                              "def", 0, 100, 100, ((0, .5, .5, .1, .1),))]
            groups = Groups({"aaaaaa", "bbbbbb"})
            config = {"class_names": ["player", "goalkeeper", "referee", "ball"],
                      "test_excluded_frame_ids": ["frame-1"]}
            with patch("pitchlens.detection_data.scan_sources", return_value=(records, {})), \
                 patch("pitchlens.detection_data.group_sources", return_value=(groups, {})), \
                 patch("pitchlens.detection_data.choose_split", return_value={"aaaaaa": "test", "bbbbbb": "train"}):
                report = prepare(raw, root / "output", config)
            self.assertEqual(report["splits"]["test"]["images"], 0)
            self.assertEqual(report["splits"]["train"]["images"], 1)
            self.assertEqual(report["excluded_previously_inspected_test_frames"], ["frame-1"])
            output = root / "output"
            entry = json.loads((output / "manifest.json").read_text(encoding="utf-8"))[0]
            label_bytes = (output / "train" / "labels" / "def.txt").read_bytes()
            self.assertEqual(hashlib.sha256(label_bytes).hexdigest(), entry["label_sha256"])
            self.assertIn(b"\r\n", label_bytes)
            with patch("pitchlens.detection_data.scan_sources", return_value=(records, {})), \
                 patch("pitchlens.detection_data.group_sources", return_value=(groups, {})), \
                 patch("pitchlens.detection_data.choose_split", return_value={"aaaaaa": "train", "bbbbbb": "test"}):
                other = prepare(raw, root / "other-output", config, seed=43)
            self.assertEqual(other["splits"]["train"]["images"], 1)
            self.assertEqual(other["excluded_previously_inspected_test_frames"], [])


if __name__ == "__main__":
    unittest.main()
