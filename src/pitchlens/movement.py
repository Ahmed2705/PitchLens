"""Estimate player movement from consecutive observed, directly mapped contacts."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math

from pitchlens.video_timing import FrameClock, is_finite_number as _finite


@dataclass(frozen=True)
class MovementConfig:
    max_speed_kmh: float = 45.0
    smoothing_seconds: float = 0.2
    min_observations: int = 3

    def __post_init__(self):
        for name in ('max_speed_kmh', 'smoothing_seconds'):
            value = getattr(self, name)
            if not _finite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if type(self.min_observations) is not int or self.min_observations < 3:
            raise ValueError('min_observations must be an integer of at least 3')


@dataclass
class _TrackMovement:
    last_frame: int | None = None
    point: tuple[float, float] | None = None
    window: deque[tuple[float, float]] = field(default_factory=deque)
    speed_kmh: float | None = None
    maximum_speed_kmh: float | None = None
    distance_m: float = 0.0
    valid_seconds: float = 0.0
    observed_seconds: float = 0.0
    accepted: int = 0
    rejected: int = 0
    window_distance: float = 0.0
    window_duration: float = 0.0

    def reset_window(self):
        self.window.clear()
        self.window_distance = self.window_duration = 0.0
        self.speed_kmh = None

    def observe(self, frame_index: int, elapsed: float | None, point: tuple[float, float] | None,
                config: MovementConfig):
        previous = self.point
        if self.last_frame is None or frame_index != self.last_frame + 1:
            elapsed = None
        self.last_frame = frame_index
        self.point = point
        if elapsed is not None:
            self.observed_seconds += elapsed
        if elapsed is None or previous is None or point is None:
            self.reset_window()
            return
        distance = math.dist(previous, point)
        if distance / elapsed * 3.6 > config.max_speed_kmh:
            self.rejected += 1
            self.point = None
            self.reset_window()
            return

        self.accepted += 1
        self.distance_m += distance
        self.valid_seconds += elapsed
        self.window.append((elapsed, distance))
        self.window_duration += elapsed
        self.window_distance += distance
        minimum_segments = config.min_observations - 1
        while (len(self.window) > minimum_segments
               and self.window_duration - self.window[0][0] >= config.smoothing_seconds):
            old_duration, old_distance = self.window.popleft()
            self.window_duration -= old_duration
            self.window_distance -= old_distance
        if len(self.window) >= minimum_segments:
            self.speed_kmh = max(0., self.window_distance) / self.window_duration * 3.6
            if self.maximum_speed_kmh is None or self.speed_kmh > self.maximum_speed_kmh:
                self.maximum_speed_kmh = self.speed_kmh

    def statistics(self, *, current: bool) -> dict:
        return {
            'speed_kmh': self.speed_kmh if current else None,
            'average_speed_kmh': self.distance_m / self.valid_seconds * 3.6 if self.accepted else None,
            'maximum_speed_kmh': self.maximum_speed_kmh,
            'observed_distance_m': self.distance_m if self.accepted else None,
            'valid_measurement_duration_seconds': self.valid_seconds,
            'observed_tracking_duration_seconds': self.observed_seconds,
            'measurement_coverage_pct': (100 * self.valid_seconds / self.observed_seconds
                                         if self.observed_seconds else None),
            'accepted_segments': self.accepted,
            'rejected_segments': self.rejected,
        }


class MovementAnalytics:
    """Annotate player and goalkeeper records and aggregate their shot-scoped statistics."""

    def __init__(self, fps: float, config: MovementConfig | None = None):
        if not _finite(fps) or fps <= 0:
            raise ValueError('fps must be positive and finite')
        self.config = config or MovementConfig()
        self._clock = FrameClock(fps)
        self._tracks: dict[tuple[int, int, int], _TrackMovement] = defaultdict(_TrackMovement)
        self._segment: int | None = None

    def update(self, frame_index: int, segment: int, camera_cut: bool, tracks: list[dict], *,
               provenance: str, timestamp_seconds: float | None = None) -> dict:
        elapsed, source, valid_clock = self._clock.step(frame_index, timestamp_seconds)
        if camera_cut or self._segment != segment:
            elapsed, source = None, None
        self._segment = segment
        for item in tracks:
            class_id, track_id = item['class_id'], item['track_id']
            if class_id not in (0, 1) or type(track_id) is not int or track_id <= 0:
                continue
            state = self._tracks[segment, class_id, track_id]
            observed = item['observed']
            if observed:
                pitch = item.get('pitch', {})
                coords = pitch.get('xy_m')
                point = None
                if (valid_clock and provenance == 'direct' and pitch.get('status') == 'valid'
                        and isinstance(coords, (list, tuple)) and len(coords) == 2 and all(map(_finite, coords))):
                    point = (coords[0], coords[1])
                state.observe(frame_index, elapsed, point, self.config)
            item['movement'] = state.statistics(current=observed)
        return {'elapsed_seconds': elapsed, 'source': source}

    def summary(self) -> list[dict]:
        return [
            {'segment': segment, 'class_id': class_id, 'track_id': track_id,
             **state.statistics(current=(segment == self._segment and state.last_frame == self._clock.frame_index))}
            for (segment, class_id, track_id), state in sorted(self._tracks.items())
        ]
