"""Use increasing source timestamps or validated FPS for adjacent video frames."""

from dataclasses import dataclass
import math


def is_finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass
class FrameClock:
    fps: float
    frame_index: int | None = None
    timestamp: float | None = None
    timestamp_valid: bool = False
    has_positive_timestamp: bool = False

    def step(self, frame_index: int, timestamp: float | None) -> tuple[float | None, str | None, bool]:
        numeric = is_finite_number(timestamp) and timestamp >= 0
        valid = timestamp is None or numeric
        if numeric:
            self.has_positive_timestamp |= timestamp > 0
            # OpenCV can report consecutive zeros when it has no position clock.
            if self.timestamp is not None and timestamp <= self.timestamp and self.has_positive_timestamp:
                valid = False

        elapsed, source = None, None
        if self.frame_index is not None and frame_index == self.frame_index + 1 and valid and self.timestamp_valid:
            if timestamp is not None and self.timestamp is not None and timestamp > self.timestamp:
                elapsed, source = timestamp - self.timestamp, 'video_timestamp'
            else:
                elapsed, source = 1 / self.fps, 'frame_index_fps'
            if not math.isfinite(elapsed) or elapsed <= 0:
                elapsed, source = None, None
        self.frame_index = frame_index
        self.timestamp = timestamp if numeric else None
        self.timestamp_valid = valid
        return elapsed, source, valid
