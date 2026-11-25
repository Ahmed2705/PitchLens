"""Contract tests for the frame detector with a controlled model backend."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from pitchlens.detector import Detection, Detector


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.weights = Path(self.temp.name) / "weights.pt"
        self.weights.touch()
        self.backend = MagicMock()
        self.backend.names = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
        self.backend.task = "detect"
        self.model_patch = patch("pitchlens.detector.YOLO", return_value=self.backend)
        self.model_factory = self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    @patch("pitchlens.detector.torch.cuda.is_available", return_value=False)
    def test_cpu_fallback_empty_result_and_model_loaded_once(self, _cuda):
        self.backend.predict.return_value = [SimpleNamespace(boxes=None)]
        detector = Detector(self.weights)
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        self.assertEqual(detector.device, "cpu")
        self.assertEqual(detector.detect(frame), [])
        self.assertEqual(detector.detect(frame), [])
        self.model_factory.assert_called_once()
        self.assertEqual(self.backend.predict.call_count, 2)

    @patch("pitchlens.detector.torch.cuda.is_available", return_value=True)
    def test_gpu_selection_and_original_frame_coordinates(self, _cuda):
        boxes = SimpleNamespace(xyxy=torch.tensor([[-5., 10., 120., 55.]]), cls=torch.tensor([3.]),
                                conf=torch.tensor([.75]), __len__=lambda self: 1)
        # SimpleNamespace does not implement special methods on instances.
        class Boxes:
            xyxy, cls, conf = boxes.xyxy, boxes.cls, boxes.conf
            def __len__(self):
                return 1
        self.backend.predict.return_value = [SimpleNamespace(boxes=Boxes())]
        detector = Detector(self.weights, confidence=.4, iou=.6, image_size=960)
        result = detector.detect(np.zeros((60, 100, 3), dtype=np.uint8))
        self.assertEqual(detector.device, "cuda:0")
        self.assertEqual(result, [Detection((0., 10., 100., 55.), 3, .75)])
        self.assertEqual(self.backend.predict.call_args.kwargs["imgsz"], 960)
        self.assertEqual(self.backend.predict.call_args.kwargs["conf"], .4)

    @patch("pitchlens.detector.torch.cuda.is_available", return_value=False)
    def test_invalid_inputs_and_unavailable_cuda(self, _cuda):
        with self.assertRaises(FileNotFoundError):
            Detector(self.weights.parent / "missing.pt")
        with self.assertRaises(RuntimeError):
            Detector(self.weights, device="cuda")
        for kwargs in ({"confidence": -1}, {"iou": float("nan")}, {"image_size": 641}, {"device": "other"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Detector(self.weights, **kwargs)
        detector = Detector(self.weights, device="cpu")
        for frame in (None, np.zeros((0, 2, 3), np.uint8), np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), np.float32)):
            with self.subTest(frame_type=type(frame), shape=getattr(frame, "shape", None)), self.assertRaises((TypeError, ValueError)):
                detector.detect(frame)

    def test_wrong_model_classes_are_rejected(self):
        self.backend.names = {0: "person"}
        with self.assertRaises(ValueError):
            Detector(self.weights)


if __name__ == "__main__":
    unittest.main()
