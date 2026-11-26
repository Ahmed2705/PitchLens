"""Check fixed-threshold detection matching against known boxes."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pitchlens.detector import Detection
from scripts.evaluate_detection import confusion_at_default_threshold, fixed_threshold_scores, group_validation_images, targets_for


class FakeDetector:
    confidence = .25

    def detect(self, frame):
        return [Detection((40., 40., 60., 60.), 3, .9), Detection((0., 0., 10., 10.), 0, .8)]


class EvaluationTests(unittest.TestCase):
    def test_counts_correct_ball_false_positive_and_missed_referee(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()
            image = images / "sample.jpg"
            self.assertTrue(cv2.imwrite(str(image), np.zeros((100, 100, 3), dtype=np.uint8)))
            (labels / "sample.txt").write_text("3 0.5 0.5 0.2 0.2\n2 0.8 0.8 0.1 0.1\n", encoding="utf-8")
            matrix = confusion_at_default_threshold(FakeDetector(), [image])
            self.assertEqual(matrix[3][3], 1)  # correct ball
            self.assertEqual(matrix[0][4], 1)  # false player
            self.assertEqual(matrix[4][2], 1)  # missed referee
            scores = fixed_threshold_scores(matrix)
            self.assertEqual(scores["ball"]["precision"], 1.)
            self.assertEqual(scores["ball"]["recall"], 1.)
            self.assertEqual(scores["player"]["false_positive"], 1)
            self.assertEqual(scores["referee"]["false_negative"], 1)

    def test_empty_annotations_are_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            image = root / "images" / "sample.jpg"
            (root / "labels" / "sample.txt").touch()
            targets = targets_for(image, 100, 100)
            self.assertEqual(tuple(targets["bboxes"].shape), (0, 4))
            self.assertEqual(tuple(targets["cls"].shape), (0,))

    def test_cross_class_confusion_counts_as_false_positive_and_false_negative(self):
        matrix = [[0] * 5 for _ in range(5)]
        matrix[3][0] = 1  # a player was predicted as a ball
        scores = fixed_threshold_scores(matrix)
        self.assertEqual(scores["ball"]["false_positive"], 1)
        self.assertEqual(scores["player"]["false_negative"], 1)

    def test_grouping_covers_exactly_the_validation_image_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            data = root / "data.yaml"
            images = [root / "val/images/a.jpg", root / "val/images/b.jpg"]
            rows = [{"split": "val", "output": path.relative_to(root).as_posix(), "group": group}
                    for path, group in zip(images, ("group-a", "group-b"))]
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(group_validation_images(data, images), {"group-a": images[:1], "group-b": images[1:]})
            with self.assertRaisesRegex(ValueError, "does not match"):
                group_validation_images(data, images[:1])
            manifest.write_text(json.dumps([*rows, rows[0]]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                group_validation_images(data, images)


if __name__ == "__main__":
    unittest.main()
