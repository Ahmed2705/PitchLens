"""Regression checks for detection candidate integrity and crop labels."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_detection_candidate import Image, exact_conflicts, verify_group_boundaries
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


if __name__ == "__main__":
    unittest.main()
