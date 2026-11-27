"""Optional SoccerNet-v3D ball detection for the tracking runner."""

from dataclasses import dataclass
from pathlib import Path

from pitchlens.detector import BALL_CLASS_ID, Detection, Detector, validate_inference_settings


@dataclass(frozen=True)
class BallDetectorConfig:
    image_size: int = 960
    confidence: float = 0.15

    def __post_init__(self) -> None:
        validate_inference_settings(self.confidence, self.image_size)


class BallDetector(Detector):
    """Load a local one-class Ultralytics ball detection checkpoint."""

    def __init__(self, weights: str | Path, config: BallDetectorConfig, *, device: str) -> None:
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dedicated ball checkpoint not found: {path}")
        if path.suffix.lower() != ".pt":
            raise ValueError(f"Dedicated ball checkpoint must be a local .pt file: {path}")
        try:
            super().__init__(path, confidence=config.confidence, image_size=config.image_size, device=device)
        except Exception as exc:
            raise RuntimeError(f"Cannot load dedicated ball checkpoint {path}: {type(exc).__name__}: {exc}") from exc

    def _class_mapping(self) -> dict[int, int]:
        names = self._model.names
        if isinstance(names, (tuple, list)):
            names = dict(enumerate(names))
        if self._model.task != "detect" or names != {0: "ball"}:
            raise ValueError(f"Expected a one-class detection model with class 0 named 'ball', "
                             f"got task {self._model.task!r} and classes {names!r}")
        return {0: BALL_CLASS_ID}


def combine_detections(general: list[Detection], ball: list[Detection] | None) -> list[Detection]:
    """Keep the general detector's output when disabled, even if it contains a ball."""
    if ball is None:
        return general
    return [detection for detection in general if detection.class_id != BALL_CLASS_ID] + ball
