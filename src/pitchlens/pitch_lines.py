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


def clip_projected_segments(segments: np.ndarray, camera: np.ndarray,
                            width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip world segments to the camera-facing viewport before dividing by depth."""
    points = segments @ camera.T
    x, y, z = np.moveaxis(points, -1, 0)
    margins = np.stack((z - 1e-9, x, width * z - x, y, height * z - y), axis=-1)
    first, last = margins[:, 0], margins[:, 1]
    entering = (first < 0) & (last >= 0)
    leaving = (first >= 0) & (last < 0)
    crossing = entering | leaving
    intersections = np.divide(first, first - last, out=np.zeros_like(first), where=crossing)
    start = np.where(entering, intersections, 0).max(axis=1)
    end = np.where(leaving, intersections, 1).min(axis=1)
    visible = ~((first < 0) & (last < 0)).any(axis=1) & (start <= end)
    origin = points[visible, 0]
    delta = points[visible, 1] - origin
    clip = np.column_stack((start[visible], end[visible]))
    ends = origin[:, None] + clip[..., None] * delta[:, None]
    return ends[..., :2] / ends[..., 2, None], visible


def near_segments(points: np.ndarray, segments: np.ndarray, radius: float) -> np.ndarray:
    """Test whether each image point is close to any finite image segment."""
    start, end = segments.transpose(1, 0, 2)
    vector = end - start
    distance = np.sum(vector ** 2, axis=1)
    offsets = points[:, None] - start
    along = np.sum(offsets * vector, axis=2)
    fraction = np.clip(np.divide(along, distance, out=np.zeros_like(along), where=distance > 0), 0, 1)
    return (np.linalg.norm(offsets - fraction[..., None] * vector, axis=2) <= radius).any(axis=1)


class PitchLineCheck:
    """Look for narrow bright ridges with green turf on both sides."""

    def __init__(self, landmarks: np.ndarray):
        indices = np.array(list(STRAIGHT_MARKINGS.values())) - 1
        endpoints = landmarks[indices]
        self.segments = np.dstack((endpoints, np.ones(endpoints.shape[:2])))
        start, end = endpoints.transpose(1, 0, 2)
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
        visible_segments = None
        for line in np.flatnonzero(samples_on_grass >= 12):
            if matches[line] >= 0.25 * samples_on_grass[line]:
                continue
            selected = (ids == line) & eligible
            shifted = ridge[selected]
            minimum = max(8, 0.45 * samples_on_grass[line])
            for offset in np.flatnonzero(shifted.sum(axis=0) >= minimum):
                hits = np.flatnonzero(selected)[shifted[:, offset]]
                if not spans_marking(points[hits]):
                    continue
                if visible_segments is None:
                    visible_segments, visible = clip_projected_segments(
                        self.segments, depth_sign * matrix, width, height)
                    visible_ids = np.flatnonzero(visible)
                # Paint near another predicted marking cannot contradict this one.
                others = visible_segments[visible_ids != line]
                aliases = near_segments(coords[hits, offset + 4], others, 2.5 * scale)
                unexplained = hits[~aliases]
                if len(unexplained) >= minimum and spans_marking(points[unexplained]):
                    return 'contradictory_paint'
        supported = np.flatnonzero((matches >= 4) & (matches >= 0.25 * samples_on_grass))
        if sum(spans_marking(points[(ids == line) & aligned]) for line in supported) < 2:
            return 'insufficient_paint'
        return None
