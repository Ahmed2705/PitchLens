"""Regression checks for detection candidate integrity and crop labels."""

import json
from hashlib import sha256
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_detection_candidate import Image, exact_conflicts, verify_group_boundaries
from prepare_detection_candidate_v2 import approved_candidate_manifest, verified_boxes, verify_boundaries, verify_export
from review_candidate_annotations import CLASSES, MANIFEST_SHA256
from verify_detection_crop import boxes_agree


def record(key, partition, group, *, phash, pixels=None, labels=()):
    path = Path(key + ".jpg")
    evidence = {"path": path.name, "stem": key, "sha256": key, "pixel_sha256": pixels or key, "phash": phash}
    return Image("fixture", path, evidence, labels, "", group, partition)


class CandidateBoundaryTests(unittest.TestCase):
    def test_frozen_source_alias_and_near_copy_cannot_train(self):
        frozen = record("frozen", "frozen", "legacy/a", phash=0)
        same_group = record("variant", "train", "legacy/a", phash=2**63 - 1)
        same_view = record("reencoded", "train", "wrkplace/b", phash=7)
        independent = record("broadcast", "test", "wrkplace/c", phash=2**63 - 1)
        frozen_inventory = [{"split": "test", "sha256": "frozen", "frame_id": "frozen", "source": frozen.key}]
        verify_group_boundaries([frozen, independent], frozen_inventory)
        for leak in (same_group, same_view):
            with self.subTest(leak=leak.key), self.assertRaisesRegex(ValueError, "Cross-partition"):
                verify_group_boundaries([frozen, leak], frozen_inventory)

    def test_pixel_identical_annotation_conflict_quarantines_all_copies(self):
        a = record("a", "train", "a", phash=0, pixels="same", labels=((0, .5, .5, .1, .1),))
        b = record("b", "train", "a", phash=0, pixels="same", labels=((1, .5, .5, .1, .1),))
        c = record("c", "train", "a", phash=0, pixels="same", labels=a.labels)
        exact_conflicts([a, b, c])
        self.assertEqual([item.status for item in (a, b, c)], ["conflicting_exact_annotations"] * 3)
        self.assertEqual(c.related, b.key)


class CropLabelsTests(unittest.TestCase):
    def test_crop_and_mirror_require_matching_labels_for_visible_objects(self):
        original = ((0, .7, .5, .10, .10), (3, .2, .5, .01, .01), (2, .99, .5, .01, .01))
        projected = ((0, .2, .5, .15, .10), (3, .95, .5, .015, .01))
        scale, offset = np.array([1.5, 1.]), np.array([-.25 * 640, 0.])
        self.assertTrue(boxes_agree(original, projected, scale, offset, mirrored=True))
        self.assertFalse(boxes_agree(original, projected[:1], scale, offset, mirrored=True))
        swapped = ((1, *projected[0][1:]), projected[1])
        self.assertFalse(boxes_agree(original, swapped, scale, offset, mirrored=True))
        self.assertFalse(boxes_agree(original, projected * 2, scale, offset, mirrored=True))


class CorrectedCandidateTests(unittest.TestCase):
    def test_export_requires_completion_marker_and_exact_frame_path(self):
        review = SimpleNamespace(frames={"frame": {"label_path": "train/labels/frame.txt"}})
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "completion marker"):
                verify_export(review, root)
            (root / "_COMPLETE").touch()
            manifest = {"manifest_sha256": MANIFEST_SHA256, "classes": CLASSES, "progress_revision": 423,
                        "frames": [{"frame_id": "frame", "status": "reviewed", "label_path": "val/labels/frame.txt"}]}
            raw = json.dumps(manifest).encode()
            (root / "review.json").write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "Approved manifest hash"):
                verify_export(review, root)
            with patch("prepare_detection_candidate_v2.EXPORT_MANIFEST_SHA256", sha256(raw).hexdigest()):
                with self.assertRaisesRegex(ValueError, "Wrong label path"):
                    verify_export(review, root)

    def test_coordinate_edit_with_unchanged_box_count_breaks_candidate_pin(self):
        original = b"3 0.5 0.5 0.2 0.2\n"
        edited = b"3 0.6 0.5 0.2 0.2\n"
        self.assertEqual(len(verified_boxes(original)), len(verified_boxes(edited)))
        rows = [{"frame_id": "reviewed-frame", "label_sha256": sha256(original).hexdigest()}]
        canonical = (json.dumps(rows, indent=2) + "\n").encode()
        with patch("prepare_detection_candidate_v2.CANDIDATE_MANIFEST_SHA256", sha256(canonical).hexdigest()):
            self.assertEqual(approved_candidate_manifest(rows), canonical)
            rows[0]["label_sha256"] = sha256(edited).hexdigest()
            with self.assertRaisesRegex(ValueError, "differs from the approved"):
                approved_candidate_manifest(rows)

    def test_normalized_boxes_allow_edge_roundoff_but_reject_invalid_labels(self):
        self.assertEqual(len(verified_boxes(b"3 0.95 0.5 0.10000001 0.1\n")), 1)
        good = b"0 0.5 0.5 0.1 0.1\n"
        for label in (b"", good * 2, b"4 0.5 0.5 0.1 0.1\n", b"0 0.5 0.5 0.0 0.1\n",
                      b"0 0.5 nan 0.1 0.1\n", b"0 0.97 0.5 0.1 0.1\n"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                verified_boxes(label)

    def test_candidate_boundaries_reject_frame_keys_and_protected_near_duplicates(self):
        train = {"source": "training.jpg", "split": "train", "group": "a", "frame_id": "frame",
                 "sha256": "a", "pixel_sha256": "pixels-a", "phash": "0000000000000000",
                 "raw_label_sha256": "label-a", "related_original": None}
        val = {**train, "source": "validation.jpg", "split": "val", "group": "b", "frame_id": "other",
               "sha256": "b", "pixel_sha256": "pixels-b", "phash": "ffffffffffffffff"}
        rows = [train, val]
        provenance = [{**row, "proposed_partition": row["split"]} for row in rows]
        verify_boundaries(rows, provenance)
        protected = {**train, "source": "protected.jpg", "proposed_partition": "ifap",
                     "group": "c", "frame_id": "protected", "sha256": "c", "pixel_sha256": "pixels-c",
                     "phash": "0000000000000007"}
        with self.assertRaisesRegex(ValueError, "protected provenance"):
            verify_boundaries(rows, [*provenance, protected])
        val["frame_id"] = provenance[1]["frame_id"] = train["frame_id"]
        with self.assertRaisesRegex(ValueError, "split overlap"):
            verify_boundaries(rows, provenance)


if __name__ == "__main__":
    unittest.main()
