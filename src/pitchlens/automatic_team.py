"""Clip-wide automatic team identities from a bounded set of SigLIP crops."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pitchlens.assisted_team import SiglipEncoder, normalized
from pitchlens.detector import Detection, Detector
from pitchlens.tracking import FrameTracks


@dataclass(frozen=True)
class AutomaticTeamConfig:
    sample_frames: int = 16
    crops_per_frame: int = 12
    min_samples: int = 16
    min_cluster_samples: int = 4
    min_detection_confidence: float = .40
    batch_size: int = 8
    retry_interval: int = 12
    min_observations: int = 2
    min_similarity: float = .65
    min_margin: float = .035


def player_crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None:
    """Take the jersey region of a plausible, fully in-frame outfield box."""
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    w, h = x2 - x1, y2 - y1
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and w >= 14 and h >= 36):
        return None
    if not .22 <= w / h <= .9:
        return None
    return frame[round(y1 + .14*h):round(y1 + .62*h), round(x1 + .15*w):round(x2 - .15*w)]


def fit_crops(frame: np.ndarray, detections: list[Detection], config: AutomaticTeamConfig) -> list[np.ndarray]:
    """Select high-confidence players with limited box overlap, up to the frame budget."""
    people = [d for d in detections if d.class_id in (0, 1, 2) and d.confidence >= config.min_detection_confidence]
    crops = []
    for player in sorted(people, key=lambda d: d.confidence, reverse=True):
        if player.class_id != 0:
            continue
        crop = player_crop(frame, player.xyxy)
        if crop is None:
            continue
        x1, y1, x2, y2 = player.xyxy
        area = (x2 - x1) * (y2 - y1)
        for other in people:
            if other is player:
                continue
            a, b, c, d = other.xyxy
            overlap = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b))
            if overlap > .35 * area:
                break
        else:
            crops.append(crop.copy())
            if len(crops) >= config.crops_per_frame:
                break
    return crops


@dataclass
class _TrackEvidence:
    last_sample: int
    candidate: int | None = None
    count: int = 0
    team: int | None = None


class AutomaticTeamClassifier:
    """Fit team identities once and cache accepted assignments within each shot."""

    def __init__(self, encoder, config: AutomaticTeamConfig | None = None):
        self.encoder = encoder
        self.config = config or AutomaticTeamConfig()
        self.prototypes = None
        self.sample_count = 0
        self.cluster_sizes = ()
        self.reset()

    def reset(self) -> None:
        self.segment = None
        self.history: dict[int, _TrackEvidence] = {}

    def fit(self, crops: list[np.ndarray]) -> None:
        if len(crops) < self.config.min_samples:
            raise ValueError(f'Automatic teams require at least {self.config.min_samples} reliable player crops, '
                             f'but found {len(crops)}. Use footage showing both teams or select --team-mode color.')
        from sklearn.cluster import KMeans
        from umap import UMAP

        features = normalized(self.encoder(crops))
        reducer = UMAP(n_components=3, n_neighbors=12, metric='cosine', random_state=42, n_jobs=1)
        projected = reducer.fit_transform(features)
        clusters = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(projected)
        sizes = np.bincount(clusters, minlength=2)
        if min(sizes) < self.config.min_cluster_samples:
            raise ValueError('Automatic teams found too few reliable crops in one cluster. '
                             'Use footage showing both teams or select --team-mode color.')
        prototypes = normalized([features[clusters == label].mean(axis=0) for label in range(2)])
        if prototypes[0] @ prototypes[1] >= .98:
            raise ValueError('Automatic teams could not distinguish two appearances in the sampled crops. '
                             'Use clearer footage or select --team-mode color.')
        self.prototypes = prototypes
        self.sample_count = len(crops)
        self.cluster_sizes = tuple(map(int, sizes))
        self.reset()

    def update(self, frame: np.ndarray, tracked: FrameTracks) -> list[int | None]:
        if self.prototypes is None:
            raise RuntimeError('Fit the automatic teams on clip samples before assigning tracks.')
        if self.segment != tracked.segment or tracked.camera_cut:
            self.reset()
            self.segment = tracked.segment
        c = self.config
        targets, crops = [], []
        for track in tracked.tracks:
            if track.class_id != 0 or track.track_id is None or not track.observed or track.confidence is None:
                continue
            evidence = self.history.get(track.track_id)
            if evidence and (evidence.team is not None or tracked.frame_index - evidence.last_sample < c.retry_interval):
                continue
            if track.confidence < c.min_detection_confidence:
                continue
            crop = player_crop(frame, track.xyxy)
            if crop is not None:
                targets.append(track.track_id)
                crops.append(crop)
        if crops:
            # Use the UMAP/K-means cluster means in normalized SigLIP space so
            # unfamiliar observations can be rejected before assigning a team.
            scores = normalized(self.encoder(crops)) @ self.prototypes.T
            for key, scores_ in zip(targets, scores, strict=True):
                evidence = self.history.setdefault(key, _TrackEvidence(last_sample=tracked.frame_index))
                evidence.last_sample = tracked.frame_index
                label = int(scores_.argmax())
                if scores_[label] < c.min_similarity or abs(scores_[0] - scores_[1]) < c.min_margin:
                    evidence.candidate, evidence.count = None, 0
                    continue
                label += 1
                evidence.count = evidence.count + 1 if evidence.candidate == label else 1
                evidence.candidate = label
                if evidence.count >= c.min_observations:
                    evidence.team = label
        return [self.history[track.track_id].team if track.class_id == 0 and track.track_id in self.history else None
                for track in tracked.tracks]


def prepare_automatic_teams(video: Path, detector: Detector, snapshot: Path,
                            max_frames: int | None = None) -> AutomaticTeamClassifier:
    """Seek over the analysis range on a separate capture before the main pass."""
    snapshot = snapshot.expanduser().resolve()
    for name in ('config.json', 'preprocessor_config.json', 'model.safetensors'):
        if not (snapshot / name).is_file():
            raise FileNotFoundError(f'Missing SigLIP snapshot file {snapshot / name}. '
                                    'Supply --siglip-snapshot with the local checkpoint directory.')
    config = AutomaticTeamConfig()
    try:
        encoder = SiglipEncoder(snapshot, detector.device, batch_size=config.batch_size, mean_pool=True)
    except Exception as exc:
        raise RuntimeError(f'Cannot initialize SigLIP at {snapshot} on {detector.device}: {exc}') from exc
    classifier = AutomaticTeamClassifier(encoder, config)
    capture = cv2.VideoCapture(str(video))
    crops = []
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or frame_count < 1:
            raise ValueError('Automatic teams require a seekable video with a known frame count.')
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        samples = min(config.sample_frames, frame_count)
        for index in range(samples):
            frame_index = (2 * index + 1) * frame_count // (2 * samples)
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError(f'Cannot seek to frame {frame_index} for automatic team sampling.')
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {frame_index} for automatic team sampling.')
            crops.extend(fit_crops(frame, detector.detect(frame), config))
    finally:
        capture.release()
    try:
        classifier.fit(crops)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(f'Cannot fit SigLIP/UMAP team clusters: {exc}') from exc
    return classifier
