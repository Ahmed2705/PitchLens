"""Local inference for the 32-landmark Roboflow soccer pose checkpoint."""

import hashlib
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from ultralytics import YOLO
from ultralytics.nn.tasks import SAFE_LOAD

CHECKPOINT_SHA256 = '28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36'


class PitchLandmarkDetector:
    """Return ordered image-space landmarks and visibility scores for one pitch."""

    def __init__(self, weights: Path, device: str, image_size: int = 640):
        if not weights.is_file() or weights.suffix.lower() != '.pt':
            raise ValueError('Supply an existing, trusted pitch pose .pt checkpoint')
        if image_size < 32 or image_size % 32:
            raise ValueError('Pitch model input size must be a positive multiple of 32')
        if not SAFE_LOAD:
            raise RuntimeError('Set ULTRALYTICS_SAFE_LOAD=1 before starting Python')
        # YOLO accepts a path and reopens it. Load a private snapshot rather than
        # reopening the external path after verification.
        with TemporaryDirectory(prefix='pitchlens-pose-') as directory:
            snapshot = Path(directory) / 'pitch-v9.pt'
            with weights.open('rb') as source, snapshot.open('xb') as target:
                shutil.copyfileobj(source, target)
            with snapshot.open('rb') as verified:
                if hashlib.file_digest(verified, 'sha256').hexdigest() != CHECKPOINT_SHA256:
                    raise ValueError('Expected the verified Roboflow v9 checkpoint (SHA-256 mismatch)')
            self.model = YOLO(str(snapshot), task='pose')
        if self.model.task != 'pose' or tuple(getattr(self.model.model.model[-1], 'kpt_shape', ())) != (32, 3):
            raise ValueError('Expected a pose checkpoint with 32 ordered landmarks and confidence')
        self.device = device
        self.image_size = image_size

    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        result = self.model.predict(frame, device=self.device, imgsz=self.image_size,
                                    conf=0.25, max_det=2, verbose=False)[0]
        # Multiple pitch instances can indicate a replay panel or composite view.
        if result.keypoints is None or len(result.keypoints) != 1:
            return None
        return result.keypoints.data[0].cpu().numpy()
