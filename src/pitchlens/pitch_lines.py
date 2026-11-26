"""Validate projected straight markings against local white paint on grass."""

import cv2
import numpy as np


STRAIGHT_MARKINGS = {
    'Side line top': (1, 25), 'Side line bottom': (6, 30),
    'Side line left': (1, 6), 'Side line right': (25, 30), 'Middle line': (14, 17),
    'Big rect. left top': (2, 10), 'Big rect. left main': (10, 13), 'Big rect. left bottom': (5, 13),
    'Big rect. right top': (26, 18), 'Big rect. right main': (18, 21), 'Big rect. right bottom': (29, 21),
    'Small rect. left top': (3, 7), 'Small rect. left main': (7, 8), 'Small rect. left bottom': (4, 8),
    'Small rect. right top': (27, 23), 'Small rect. right main': (23, 24), 'Small rect. right bottom': (28, 24),
}


class PitchLineCheck:
    """Look for narrow bright ridges with green turf on both sides."""

    def __init__(self, landmarks: np.ndarray):
        indices = np.array(list(STRAIGHT_MARKINGS.values())) - 1
        start, end = landmarks[indices].transpose(1, 0, 2)
        delta = end - start
        # Omit the endpoints, where other markings often cross the sampling strip.
        fractions = np.linspace(0.05, 0.95, 48)
        points = start[:, None] + delta[:, None] * fractions[None, :, None]
        self.samples = np.dstack((points, np.ones(points.shape[:2])))
        self.tangents = np.column_stack((delta, np.zeros(len(delta))))

    def rejection(self, frame: np.ndarray, matrix: np.ndarray, depth_sign: float) -> str | None:
        """Require paint support and veto a long marking with coherent off-center paint."""
        homogeneous = self.samples @ matrix.T
        depth = homogeneous[..., 2]
        facing = depth_sign * depth > 1e-8
        if not facing.any():
            return 'insufficient_paint'
        ids, samples = np.nonzero(facing)
        points = homogeneous[ids, samples, :2] / depth[ids, samples, None]
        tangents = self.tangents[ids] @ matrix.T
        direction = tangents[:, :2] - points * tangents[:, 2, None]
        normals = direction[:, ::-1] * [-1, 1]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        height, width = frame.shape[:2]
        # Normal offsets are normalized to a 960 px image diagonal.
        scale = max(0.75, np.hypot(width, height) / 960)
        offsets = np.arange(-14, 15) * scale
        coords = points[:, None] + normals[:, None] * offsets[None, :, None]
        inside = ((coords >= 0) & (coords < [width - 1, height - 1])).all(axis=(1, 2))
        if not inside.any():
            return 'insufficient_paint'
        coords = coords[inside].astype(np.float32)
        ids = ids[inside]
        colors = cv2.remap(frame, coords[..., 0], coords[..., 1], cv2.INTER_LINEAR).astype(np.int16)
        blue, green, red = colors.transpose(2, 0, 1)
        grass = green - np.maximum(blue, red) >= 7
        neutral = np.minimum(blue, red)
        shoulders = grass[:, :-8] & grass[:, 8:]
        contrast = neutral[:, 4:-4] - np.maximum(neutral[:, :-8], neutral[:, 8:])
        ridge = shoulders & (contrast >= 12)
        center = ridge.shape[1] // 2
        eligible = shoulders[:, center - 2:center + 3].any(axis=1)
        aligned = ridge[:, center - 2:center + 3].any(axis=1)
        samples_on_grass = np.bincount(ids[eligible], minlength=len(self.samples))
        matches = np.bincount(ids[aligned], minlength=len(self.samples))
        points = points[inside]

        def spans_marking(selected: np.ndarray) -> bool:
            # Several world samples can collapse onto the same few image pixels.
            # Samples remain ordered along each projected line.
            return len(selected) >= 2 and np.linalg.norm(selected[-1] - selected[0]) >= 24 * scale

        # An isolated player or logo can create a bright peak. A parallel offset
        # repeated along a line is evidence against the projected position.
        for line in np.flatnonzero(samples_on_grass >= 12):
            if matches[line] >= 0.25 * samples_on_grass[line]:
                continue
            selected = (ids == line) & eligible
            shifted = ridge[selected]
            counts = shifted.sum(axis=0)
            peak = counts.argmax()
            if (counts[peak] >= max(8, 0.45 * samples_on_grass[line])
                    and spans_marking(points[selected][shifted[:, peak]])):
                return 'contradictory_paint'
        supported = np.flatnonzero((matches >= 4) & (matches >= 0.25 * samples_on_grass))
        if sum(spans_marking(points[(ids == line) & aligned]) for line in supported) < 2:
            return 'insufficient_paint'
        return None
