"""Draw a top-down pitch inset using the frame's calibrated player positions."""

import cv2
import numpy as np

from pitchlens.calibration import PitchConfig


class PitchMinimap:
    def __init__(self, config: PitchConfig, frame_shape: tuple[int, ...]):
        height, width = frame_shape[:2]
        self.padding = 6
        self.scale = min((width * 0.28 - 12) / config.length_m,
                         (height * 0.28 - 12) / config.width_m)
        self.background = None
        if self.scale <= 0:
            return
        x_end, y_end = self._pixel((config.length_m, config.width_m))
        field = np.full((y_end + self.padding + 1, x_end + self.padding + 1, 3), (41, 95, 45), np.uint8)
        line = (219, 229, 219)
        thickness = max(1, round(self.scale / 2.5))
        cv2.rectangle(field, self._pixel((0, 0)), (x_end, y_end), line, thickness, cv2.LINE_AA)
        mid_x, mid_y = config.length_m / 2, config.width_m / 2
        cv2.line(field, self._pixel((mid_x, 0)), self._pixel((mid_x, config.width_m)), line, thickness, cv2.LINE_AA)
        cv2.circle(field, self._pixel((mid_x, mid_y)), round(9.15 * self.scale), line, thickness, cv2.LINE_AA)
        for depth, half_width in ((16.5, 20.16), (5.5, 9.16)):
            cv2.rectangle(field, self._pixel((0, mid_y - half_width)),
                          self._pixel((depth, mid_y + half_width)), line, thickness, cv2.LINE_AA)
            cv2.rectangle(field, self._pixel((config.length_m - depth, mid_y - half_width)),
                          self._pixel((config.length_m, mid_y + half_width)), line, thickness, cv2.LINE_AA)
        self.background = field

    def _pixel(self, xy: tuple[float, float] | list[float]) -> tuple[int, int]:
        return (self.padding + round(xy[0] * self.scale), self.padding + round(xy[1] * self.scale))

    def draw(self, frame: np.ndarray, tracks: list[dict], valid: bool) -> None:
        if self.background is None:
            return
        field = self.background.copy()
        if valid:
            for track in tracks:
                contact = track.get('pitch')
                if contact is None or contact['xy_m'] is None:
                    continue
                color = {1: (255, 120, 0), 2: (220, 0, 220)}.get(track.get('team'), (220, 220, 220))
                thickness = -1 if contact['status'] == 'valid' else 1
                cv2.circle(field, self._pixel(contact['xy_m']), 4, color, thickness, cv2.LINE_AA)
        else:
            field //= 2
            cv2.putText(field, 'Pitch unavailable', (7, max(12, field.shape[0] // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        h, w = field.shape[:2]
        frame[-h:, -w:] = field
