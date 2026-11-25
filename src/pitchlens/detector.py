"""Frame-level football object detection, independent of tracking."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Keep Ultralytics' process-local settings and plot cache in the project when
# running from the project root; explicit environment overrides take priority.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "outputs" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "outputs" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
from ultralytics import YOLO  # noqa: E402

CLASS_NAMES = ("player", "goalkeeper", "referee", "ball")


@dataclass(frozen=True, slots=True)
class Detection:
    """One detection in original-frame pixel coordinates: (left, top, right, bottom)."""

    xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float


class Detector:
    """Load a trained model once and detect objects in BGR uint8 frames."""

    def __init__(self, weights: str | Path, *, confidence: float = .25, iou: float = .7,
                 class_confidence: Mapping[int, float] | None = None,
                 image_size: int = 960, device: str = "auto") -> None:
        path = Path(weights).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Detector weights not found: {path}")
        for name, value in (("confidence", confidence), ("iou", iou)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number between 0 and 1")
        if class_confidence is not None and not isinstance(class_confidence, Mapping):
            raise ValueError("class_confidence must map class IDs to thresholds")
        for category, value in (class_confidence or {}).items():
            if isinstance(category, bool) or not isinstance(category, int) or not 0 <= category < len(CLASS_NAMES):
                raise ValueError("class_confidence contains an invalid class ID")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("class_confidence thresholds must be finite numbers between 0 and 1")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 32 or image_size % 32:
            raise ValueError("image_size must be a positive multiple of 32")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = "cuda:0" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.confidence = float(confidence)
        self.class_confidence = dict(class_confidence or {})
        self.iou = float(iou)
        self.image_size = image_size
        self._model = YOLO(str(path))
        names = self._model.names
        actual = tuple(names.get(i) for i in range(len(names))) if isinstance(names, dict) else tuple(names)
        if self._model.task != "detect" or actual != CLASS_NAMES:
            raise ValueError(f"Expected a four-class PitchLens detection model, got {actual}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in one nonempty H×W×3 OpenCV BGR frame.

        Coordinates are clipped to the original frame boundaries. An empty
        detection result is returned as an empty list.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.shape[0] or not frame.shape[1]:
            raise ValueError("frame must be a nonempty H×W×3 uint8 BGR array")
        height, width = frame.shape[:2]
        minimum_confidence = min([self.confidence, *self.class_confidence.values()])
        results = self._model.predict(source=np.ascontiguousarray(frame), conf=minimum_confidence, iou=self.iou,
                                      imgsz=self.image_size, device=self.device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        detections = []
        for xyxy, category, score in zip(coordinates, classes, scores, strict=True):
            if not all(math.isfinite(float(value)) for value in (*xyxy, category, score)):
                raise ValueError("Model returned non-finite detection values")
            class_id = int(category)
            if category != class_id or not 0 <= class_id < len(CLASS_NAMES) or not 0 <= score <= 1:
                raise ValueError("Model returned an invalid class ID or confidence")
            if score < self.class_confidence.get(class_id, self.confidence):
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            x1, x2 = max(0., min(x1, width)), max(0., min(x2, width))
            y1, y2 = max(0., min(y1, height)), max(0., min(y2, height))
            if x2 > x1 and y2 > y1:
                detections.append(Detection((x1, y1, x2, y2), class_id, float(score)))
        return detections
