"""SoccerNet-v3D ball detection and shared video runner model selection."""

import argparse
from dataclasses import dataclass
import json
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
            raise FileNotFoundError(f"Dedicated ball checkpoint not found: {path}. "
                                    "Use --storage-root or --ball-weights to locate the local checkpoint.")
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_CONFIG = PROJECT_ROOT / "configs" / "detection.json"


@dataclass(frozen=True)
class VideoDetectionSettings:
    profile: str
    people_weights: Path
    people_image_size: int
    people_confidence: float
    ball_weights: Path | None
    ball_config: BallDetectorConfig | None


def add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--detector-profile', choices=('final', 'legacy', 'gsr'),
                        help='Defaults to final, or legacy when --weights is given without a profile')
    parser.add_argument('--weights', type=Path, help='Override the profile’s local people checkpoint')
    parser.add_argument('--storage-root', type=Path, help='Override the configured external checkpoint directory')
    parser.add_argument('--people-image-size', type=int, help='Override the people inference resolution')
    parser.add_argument('--people-confidence', type=float, help='Override the people detection output threshold')
    parser.add_argument('--ball-weights', type=Path, help='Override or enable the dedicated local ball checkpoint')
    parser.add_argument('--ball-config', type=Path, help='JSON overrides for dedicated ball inference')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')


def video_detection_settings(args: argparse.Namespace) -> VideoDetectionSettings:
    profile = args.detector_profile or ('legacy' if args.weights is not None else 'final')
    try:
        defaults = json.loads(DETECTION_CONFIG.read_text(encoding='utf-8-sig'))
        selected = defaults['profiles'][profile]
        storage = (args.storage_root or Path(defaults['storage_root'])).expanduser()
        people_root = storage if profile == 'gsr' else PROJECT_ROOT
        people_weights = args.weights or people_root / selected['people_weights']
        size = selected['image_size'] if args.people_image_size is None else args.people_image_size
        confidence = selected['confidence'] if args.people_confidence is None else args.people_confidence
        validate_inference_settings(confidence, size)
        ball_weights = args.ball_weights
        if profile != 'legacy' and ball_weights is None:
            ball_weights = storage / defaults['ball_weights']
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid people detector settings for profile {profile!r} ({DETECTION_CONFIG}): {exc}') from exc
    if ball_weights is None and args.ball_config is not None:
        raise ValueError('--ball-config requires a dedicated ball checkpoint. Supply --ball-weights or use a dual profile.')
    ball_config = None
    if ball_weights is not None:
        try:
            overrides = json.loads(args.ball_config.read_text(encoding='utf-8-sig')) if args.ball_config else {}
            if not isinstance(overrides, dict):
                raise ValueError('expected a JSON object')
            ball_config = BallDetectorConfig(**overrides)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f'Invalid dedicated ball configuration ({args.ball_config}): {exc}') from exc
    return VideoDetectionSettings(profile, people_weights.expanduser().resolve(), size, confidence,
                                  ball_weights.expanduser().resolve() if ball_weights is not None else None, ball_config)
