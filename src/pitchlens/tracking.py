"""Frame-by-frame person and ball tracking in original image coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import math

import numpy as np
from pitchlens.camera_cut import is_hard_cut, thumbnail
from pitchlens.detector import Detection  # Configure Ultralytics before importing its tracking modules.
from ultralytics.engine.results import Boxes
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker

PEOPLE_TRACKERS = ('botsort', 'bytetrack')


@dataclass(frozen=True)
class TrackingConfig:
    person_low: float = 0.1
    person_high: float = 0.25
    person_new: float = 0.25
    person_match: float = 0.8
    person_buffer_seconds: float = 1.0
    ball_confidence: float = 0.25
    ball_gap_seconds: float = 0.25
    ball_gate_fraction: float = 0.08
    cut_threshold: float = 0.32
    trajectory_seconds: float = 1.0
    cut_appearance_threshold: float = 0.055
    people_tracker: str = 'botsort'

    def __post_init__(self):
        if self.people_tracker not in PEOPLE_TRACKERS:
            raise ValueError("people_tracker must be 'botsort' or 'bytetrack'")
        for name in ('person_low', 'person_high', 'person_new', 'person_match', 'ball_confidence',
                     'ball_gate_fraction', 'cut_threshold', 'cut_appearance_threshold'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be finite and in (0, 1]')
        if self.person_low >= self.person_high or self.person_new < self.person_high:
            raise ValueError('person thresholds must satisfy low < high <= new')
        for name in ('person_buffer_seconds', 'ball_gap_seconds', 'trajectory_seconds'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class Track:
    """A current track. Person IDs are class-scoped and disjoint across cuts within a video."""
    xyxy: tuple[float, float, float, float]
    class_id: int
    track_id: int | None
    confidence: float | None
    observed: bool


@dataclass(frozen=True)
class FrameTracks:
    frame_index: int
    segment: int
    camera_cut: bool
    tracks: tuple[Track, ...]
    ball_trajectory: tuple[tuple[float, float, bool], ...]
    ball_observation_ambiguous: bool = False


class Tracker:
    """Separate tracker instances per person class and a gated ball predictor.

    Create a fresh instance per video, or call reset() before processing another.
    Consecutive frames must be supplied in order at the configured source FPS.
    """
    def __init__(self, fps: float, config: TrackingConfig | None = None):
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be positive and finite')
        self.fps = float(fps)
        self.config = config or TrackingConfig()
        self.reset()

    def _reset_segment(self):
        c = self.config
        args = SimpleNamespace(track_high_thresh=c.person_high, track_low_thresh=c.person_low,
                               new_track_thresh=c.person_new, match_thresh=c.person_match,
                               track_buffer=max(1, round(c.person_buffer_seconds * self.fps)), fuse_score=False)
        if c.people_tracker == 'botsort':
            args.gmc_method = 'sparseOptFlow'
            args.with_reid = False
            args.proximity_thresh = 0.5
            args.appearance_thresh = 0.8
            self.people = [BOTSORT(args) for _ in range(3)]
        else:
            self.people = [BYTETracker(args) for _ in range(3)]
        self.ball = None
        self.velocity = np.zeros(2)
        self.gap = 0
        self.trajectory = []

    def reset(self):
        self.segment = 0
        self.frame_index = 0
        self.track_id_offset = 0
        self.largest_track_id = 0
        self.previous = None
        self.shape = None
        self._reset_segment()

    def update(self, frame: np.ndarray, detections: list[Detection]) -> FrameTracks:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.shape[0] or not frame.shape[1]:
            raise ValueError('frame must be a nonempty HxWx3 uint8 array')
        if self.shape is not None and frame.shape != self.shape:
            raise ValueError('frame dimensions changed; reset before a new video')
        h, w = frame.shape[:2]
        for d in detections:
            if not isinstance(d, Detection) or isinstance(d.class_id, bool) or d.class_id not in range(4) or not math.isfinite(d.confidence) or not 0 <= d.confidence <= 1 or len(d.xyxy) != 4 or not all(math.isfinite(x) for x in d.xyxy) or not (0 <= d.xyxy[0] < d.xyxy[2] <= w and 0 <= d.xyxy[1] < d.xyxy[3] <= h):
                raise ValueError('invalid detection')
        small = thumbnail(frame)
        cut = self.previous is not None and is_hard_cut(
            self.previous, small, appearance_threshold=self.config.cut_appearance_threshold,
            low_detail_threshold=self.config.cut_threshold)
        if cut:
            self.segment += 1
            self.track_id_offset = self.largest_track_id
            self._reset_segment()
        self.previous = small
        self.shape = frame.shape
        result = []
        # BoT-SORT needs unannotated source frames, including frames without boxes.
        motion_frame = frame if self.config.people_tracker == 'botsort' else None
        for cls, tracker in enumerate(self.people):
            rows = [[*d.xyxy, d.confidence, cls] for d in detections if d.class_id == cls]
            boxes = Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), (h, w))
            for row in tracker.update(boxes, img=motion_frame):
                x1, y1, x2, y2 = row[:4]
                xyxy = (max(0., min(float(x1), w)), max(0., min(float(y1), h)),
                        max(0., min(float(x2), w)), max(0., min(float(y2), h)))
                if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                    track_id = self.track_id_offset + int(row[4])
                    self.largest_track_id = max(self.largest_track_id, track_id)
                    result.append(Track(xyxy, cls, track_id, float(row[5]), True))
        ball = self._update_ball([d for d in detections if d.class_id == 3 and d.confidence >= self.config.ball_confidence], w, h)
        if ball is not None:
            result.append(ball)
        output = FrameTracks(self.frame_index, self.segment, bool(cut), tuple(result), tuple(self.trajectory),
                             self.ball_observation_ambiguous)
        self.frame_index += 1
        return output

    def _update_ball(self, candidates, width, height):
        def center(d):
            x1, y1, x2, y2 = d.xyxy
            return np.array([(x1+x2)/2, (y1+y2)/2])
        chosen = None
        self.ball_observation_ambiguous = False
        if self.ball is not None and self.gap >= max(1, round(self.config.ball_gap_seconds * self.fps)):
            self.ball = None
            self.velocity = np.zeros(2)
            self.gap = 0
            self.trajectory = []
        if self.ball is None:
            self.ball_observation_ambiguous = len(candidates) > 1
            chosen = max(candidates, key=lambda d: d.confidence, default=None)
        else:
            predicted = self.ball[:2] + self.velocity
            gate = self.config.ball_gate_fraction * math.hypot(width, height) * (1 + 0.25 * self.gap)
            viable = [(d, float(np.linalg.norm(center(d) - predicted))) for d in candidates]
            viable = [(d, distance) for d, distance in viable if distance <= gate]
            self.ball_observation_ambiguous = len(viable) > 1
            if viable:
                chosen = min(viable, key=lambda item: item[1] / gate + (1 - item[0].confidence))[0]
        if chosen is not None:
            position = center(chosen)
            if self.ball is not None:
                # Velocity is measured from the last predicted state; smoothing limits jitter.
                self.velocity = 0.5 * self.velocity + 0.5 * (position - self.ball[:2])
            x1, y1, x2, y2 = chosen.xyxy
            self.ball = np.array([*position, x2-x1, y2-y1])
            self.gap = 0
            observed, confidence = True, chosen.confidence
        elif self.ball is not None and self.gap < max(1, round(self.config.ball_gap_seconds * self.fps)):
            self.ball[:2] += self.velocity
            self.gap += 1
            if not (0 <= self.ball[0] < width and 0 <= self.ball[1] < height):
                self.ball = None
                self.velocity = np.zeros(2)
                self.gap = 0
                self.trajectory = []
                return None
            observed, confidence = False, None
        else:
            self.ball = None
            self.velocity = np.zeros(2)
            self.gap = 0
            self.trajectory = []
            return None
        x, y, bw, bh = self.ball
        xyxy = (max(0., x-bw/2), max(0., y-bh/2), min(float(width), x+bw/2), min(float(height), y+bh/2))
        self.trajectory.append((float(x), float(y), observed))
        self.trajectory = self.trajectory[-max(1, round(self.config.trajectory_seconds * self.fps)):]
        return Track(tuple(float(v) for v in xyxy), 3, None, confidence, observed)
