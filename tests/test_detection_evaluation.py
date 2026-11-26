"""Check fixed-threshold detection matching against known boxes."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml

from pitchlens.detection_manifest import label_for, sha256
from pitchlens.detector import CLASS_NAMES, Detection
from scripts.evaluate_detection import confusion_at_default_threshold, evaluate, fixed_threshold_scores, targets_for
from ultralytics.utils import YAML


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


class ManifestEvaluationTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(folder).resolve()
        self.dataset = self.root / "dataset"
        images = self.dataset / "val/images"
        images.mkdir(parents=True)
        (self.dataset / "val/labels").mkdir()
        self.images, rows = [], []
        for index, group in enumerate(("group-a", "group-a", "group-b")):
            label = b"3 0.5 0.5 0.2 0.2\n"
            ok, encoded = cv2.imencode(".jpg", np.full((100, 100, 3), index, dtype=np.uint8))
            self.assertTrue(ok)
            payload = encoded.tobytes()
            image_hash = sha256(payload)
            image = images / f"{image_hash}.jpg"
            image.write_bytes(payload)
            label_for(image).write_bytes(label)
            self.images.append(image)
            rows.append({"split": "val", "output": image.relative_to(self.dataset).as_posix(), "group": group,
                         "sha256": image_hash, "label_sha256": sha256(label)})
        self.rows = rows
        raw = json.dumps(rows).encode()
        (self.dataset / "manifest.json").write_bytes(raw)
        data = self.dataset / "data.yaml"
        data.write_text(yaml.safe_dump({"train": "train/images", "val": "val/images",
                                        "names": list(CLASS_NAMES)}), encoding="utf-8")
        weights = self.root / "fixture.pt"
        weights.write_bytes(b"checkpoint fixture")
        self.args = SimpleNamespace(data=data, weights=weights, run=self.root / "evaluation", split="val",
                                    manifest_sha256=sha256(raw), image_size=64, confidence=.25,
                                    benchmark_frames=1, by_source_group=True)
        self.assert_no_model = self.enterContext(patch("scripts.evaluate_detection.YOLO"))

    def assert_rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            evaluate(self.args)
        self.assert_no_model.assert_not_called()
        self.assertFalse((self.args.run / "evaluation.json").exists())

    def test_modified_image_fails_before_model_load(self):
        self.images[0].write_bytes(b"changed image")
        self.assert_rejected("SHA-256 mismatch")

    def test_modified_annotation_fails_before_model_load(self):
        label_for(self.images[0]).write_bytes(b"3 0.6 0.5 0.2 0.2\n")
        self.assert_rejected("SHA-256 mismatch")

    def test_modified_manifest_fails_before_model_load(self):
        (self.dataset / "manifest.json").write_bytes(json.dumps(self.rows[::-1]).encode())
        self.assert_rejected("SHA-256 mismatch")

    def test_missing_label_fails_before_model_load(self):
        label_for(self.images[0]).unlink()
        self.assert_rejected("Evaluation files differ")

    def test_nested_unexpected_supported_image_fails_before_model_load(self):
        nested = self.images[0].parent / "nested"
        nested.mkdir()
        (nested / "extra.PNG").write_bytes(self.images[0].read_bytes())
        self.assert_rejected("Unexpected.*extra.PNG")

    def test_existing_evaluation_output_is_preserved(self):
        self.args.run.mkdir()
        result = self.args.run / "evaluation.json"
        result.write_bytes(b"old report")
        with self.assertRaises(FileExistsError):
            evaluate(self.args)
        self.assertEqual(result.read_bytes(), b"old report")
        self.assert_no_model.assert_not_called()

    def synthetic_evaluations(self, *, drop_last=False, change_source=False, repeat=False):
        """Use the real YOLO label loader and cache with synthetic detector results."""
        cache_paths, ap_paths, confusion_paths = [], [], []
        names = dict(enumerate(CLASS_NAMES))
        model = SimpleNamespace(task="detect", names=names)
        source_changed = False

        def validate(validator, data, **kwargs):
            nonlocal source_changed
            if change_source and not source_changed:
                label_for(self.images[0]).write_bytes(b"3 0.6 0.5 0.2 0.2\n")
                source_changed = True
            config = YAML.load(data)
            self.assertEqual(config["train"], config["val"])
            validator = validator(args={**kwargs, "mode": "val", "data": data})
            validator.device = torch.device("cpu")
            validator.stride = 32
            validator.data = {"names": names, "nc": len(names), "channels": 3}
            validator.dataloader = validator.get_dataloader(config["val"], batch_size=1)
            validator.init_metrics(model)
            dataset = validator.dataloader.dataset
            visited = list(map(Path, dataset.im_files))
            ap_paths.append([path.name for path in visited])
            cache = visited[0].parent.parent / "labels.cache"
            self.assertTrue(cache.is_file())
            cache_paths.append(cache)
            limit = len(visited) - int(drop_last)
            for index, batch in enumerate(validator.dataloader):
                if index == limit:
                    break
                targets = validator._prepare_batch(0, batch)
                prediction = {"cls": targets["cls"], "bboxes": targets["bboxes"], "conf": torch.tensor([.9])}
                validator.update_metrics([prediction], batch)
            if limit:
                validator.get_stats()
            return validator.metrics

        def confusion(detector, paths):
            confusion_paths.append([path.name for path in paths])
            return confusion_at_default_threshold(detector, paths)

        model.val = validate
        self.assert_no_model.return_value = model
        with (patch("scripts.evaluate_detection.Detector", return_value=FakeDetector()),
              patch("scripts.evaluate_detection.confusion_at_default_threshold", side_effect=confusion),
              patch.multiple(torch.cuda, empty_cache=lambda: None, reset_peak_memory_stats=lambda: None,
                             synchronize=lambda: None, max_memory_allocated=lambda: 0,
                             max_memory_reserved=lambda: 0, get_device_name=lambda _: "fixture")):
            results = [evaluate(self.args)]
            if repeat:
                self.args.run = self.root / "repeat"
                results.append(evaluate(self.args))
        return results, ap_paths, confusion_paths, cache_paths

    def test_ap_and_confusion_share_the_full_population_and_group_subsets(self):
        results, ap_paths, confusion_paths, _ = self.synthetic_evaluations(change_source=True)
        self.assertCountEqual(ap_paths[0], [path.name for path in self.images])
        self.assertCountEqual(ap_paths[0], [name for group in ap_paths[1:] for name in group])
        self.assertEqual([set(group) for group in ap_paths[1:]], [set(group) for group in confusion_paths])
        self.assertEqual(results[0]["confusion_matrix"]["matrix"][3][3], 3)
        self.assertEqual(results[0]["per_class"]["ball"]["instances"], 3)
        self.assertGreater(results[0]["overall"]["map50"], .9)
        self.assertEqual(sum(group["images"] for group in results[0]["source_groups"].values()), 3)

    def test_ap_rejects_a_validator_that_skips_a_frame(self):
        with self.assertRaisesRegex(ValueError, "AP metrics population differs"):
            self.synthetic_evaluations(drop_last=True)
        self.assertFalse((self.args.run / "evaluation.json").exists())

    def test_caches_are_isolated_and_repeated_evaluation_has_identical_results(self):
        old_cache = self.dataset / "val/labels.cache"
        old_cache.write_bytes(b"existing cache")
        results, ap_paths, confusion_paths, caches = self.synthetic_evaluations(repeat=True)
        self.assertEqual(ap_paths[:3], ap_paths[3:])
        self.assertEqual(confusion_paths[:2], confusion_paths[2:])
        self.assertEqual(len(set(caches)), 6)
        self.assertTrue(all(not cache.exists() for cache in caches))
        self.assertEqual(list(self.dataset.rglob("*.cache")), [old_cache])
        self.assertEqual(old_cache.read_bytes(), b"existing cache")
        for result in results:
            result.pop("benchmark")  # Runtime measurements vary.
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
