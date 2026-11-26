"""Estimate a fresh image-to-pitch mapping from each frame's pitch landmarks."""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchConfig:
    length_m: float = 105.0
    width_m: float = 68.0
    landmark_confidence: float = 0.5
    min_inliers: int = 6
    min_inlier_ratio: float = 0.65
    reprojection_fraction: float = 0.004
    smoothing: float = 0.35
    extrapolation_fraction: float = 0.06

    def __post_init__(self):
        if not 50 <= self.length_m <= 130 or not 41 <= self.width_m <= 100:
            raise ValueError('Pitch dimensions must fit the modeled markings')
        if type(self.min_inliers) is not int or not 5 <= self.min_inliers <= 32:
            raise ValueError('min_inliers must be an integer between 5 and 32')
        if not (0 < self.landmark_confidence <= 1 and 0.5 <= self.min_inlier_ratio <= 1
                and 0 < self.reprojection_fraction <= 0.02 and 0 <= self.smoothing < 1
                and 0 <= self.extrapolation_fraction <= 0.2):
            raise ValueError('Invalid pitch calibration thresholds')

    def landmarks(self) -> np.ndarray:
        """Roboflow pose indices 1–32 with regulation markings in meters."""
        length, width = self.length_m, self.width_m
        mid = width / 2
        p0, p1 = mid - 20.16, mid + 20.16
        g0, g1 = mid - 9.16, mid + 9.16
        return np.array([
            (0, 0), (0, p0), (0, g0), (0, g1), (0, p1), (0, width),
            (5.5, g0), (5.5, g1), (11, mid),
            (16.5, p0), (16.5, g0), (16.5, g1), (16.5, p1),
            (length/2, 0), (length/2, mid-9.15), (length/2, mid+9.15), (length/2, width),
            (length-16.5, p0), (length-16.5, g0), (length-16.5, g1), (length-16.5, p1),
            (length-11, mid), (length-5.5, g0), (length-5.5, g1),
            (length, 0), (length, p0), (length, g0), (length, g1), (length, p1), (length, width),
            (length/2-9.15, mid), (length/2+9.15, mid),
        ], dtype=np.float64)


def project(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project points without silently mapping the projective horizon to zero."""
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    denominator = homogeneous[:, 2, None]
    return np.divide(homogeneous[:, :2], denominator,
                     out=np.full((len(points), 2), np.nan), where=np.abs(denominator) > 1e-10)


def has_projective_support(world: np.ndarray, image: np.ndarray) -> bool:
    """Check that the normalized correspondence equations constrain eight degrees of freedom."""
    def normalize(points: np.ndarray) -> np.ndarray:
        centered = points - points.mean(axis=0)
        return centered / np.sqrt(np.mean(np.sum(centered ** 2, axis=1)) / 2)

    x, y = normalize(world).T
    u, v = normalize(image).T
    ones, zeros = np.ones(len(x)), np.zeros(len(x))
    equations = np.vstack((np.column_stack((x, y, ones, zeros, zeros, zeros, -u*x, -u*y, -u)),
                           np.column_stack((zeros, zeros, zeros, x, y, ones, -v*x, -v*y, -v))))
    singular = np.linalg.svd(equations, compute_uv=False)
    # Ignore the solution direction. Another near-zero mode makes the fit ambiguous.
    return bool(singular[-2] > 0.01 * singular[0])


@dataclass(frozen=True)
class Calibration:
    config: PitchConfig
    frame_size: tuple[int, int]
    image_to_pitch: np.ndarray | None = None
    hull: np.ndarray | None = None
    candidates: int = 0
    inliers: int = 0
    reprojection_rmse_px: float | None = None
    mean_landmark_confidence: float | None = None
    stabilized: bool = False
    reason: str | None = None

    def metadata(self) -> dict:
        return {
            'valid': self.image_to_pitch is not None, 'reason': self.reason,
            'landmarks': self.candidates, 'inliers': self.inliers,
            'inlier_ratio': self.inliers / self.candidates if self.candidates else None,
            'reprojection_rmse_px': self.reprojection_rmse_px,
            'mean_landmark_confidence': self.mean_landmark_confidence,
            'stabilized': self.stabilized,
        }

    def map_contact(self, xyxy: tuple[float, float, float, float]) -> dict:
        unavailable = {'xy_m': None, 'status': 'unavailable'}
        if self.image_to_pitch is None or self.hull is None:
            return unavailable
        x1, _, x2, y2 = xyxy
        width, height = self.frame_size
        # A clipped box does not locate the feet at its bottom center.
        if not (0 < x1 <= x2 < width and 0 < y2 < height):
            return unavailable
        point = ((x1 + x2) / 2, y2)
        if not all(math.isfinite(value) for value in point):
            return unavailable
        distance = cv2.pointPolygonTest(self.hull, point, True)
        if distance < -max(0.001, self.config.extrapolation_fraction * math.hypot(width, height)):
            return unavailable
        x, y = project(self.image_to_pitch, np.array([point]))[0]
        tolerance = 1e-5  # Round-off at touchlines and goal lines, in meters.
        if not (-tolerance <= x <= self.config.length_m + tolerance
                and -tolerance <= y <= self.config.width_m + tolerance):
            return unavailable
        return {'xy_m': [float(np.clip(x, 0, self.config.length_m)),
                         float(np.clip(y, 0, self.config.width_m))],
                'status': 'valid' if distance >= -0.001 else 'estimated'}


class PitchCalibrator:
    """Robustly fit in pixels, then smooth only against verified current evidence."""

    def __init__(self, config: PitchConfig):
        self.config = config
        self.world = config.landmarks()
        self.previous: np.ndarray | None = None

    def update(self, observations: np.ndarray | None, shape: tuple[int, ...],
               camera_cut: bool = False) -> Calibration:
        previous = None if camera_cut else self.previous
        self.previous = None
        height, width = shape[:2]
        frame_size = (width, height)
        diagonal = math.hypot(width, height)
        config = self.config
        if observations is None:
            return Calibration(config, frame_size, reason='no_single_pitch')
        if observations.shape != (len(self.world), 3):
            raise ValueError('Expected one (x, y, confidence) triplet per landmark')
        x, y, confidence = observations.T
        # Ultralytics clips predictions to the image, so boundary pixels may be off-screen points.
        keep = (np.isfinite(observations).all(axis=1) & (confidence >= config.landmark_confidence)
                & (confidence <= 1) & (x > 0) & (x < width) & (y > 0) & (y < height))
        world, image, scores = self.world[keep], observations[keep, :2], confidence[keep]
        count = len(world)
        if count < config.min_inliers:
            return Calibration(config, frame_size, candidates=count, reason='too_few_landmarks')
        threshold = max(1.0, config.reprojection_fraction * diagonal)
        matrix, mask = cv2.findHomography(world, image, cv2.RANSAC, threshold)
        if matrix is None or mask is None:
            return Calibration(config, frame_size, candidates=count, reason='fit_failed')
        mask = mask.ravel().astype(bool)
        inliers = int(mask.sum())
        if inliers < config.min_inliers or inliers / count < config.min_inlier_ratio:
            return Calibration(config, frame_size, candidates=count, inliers=inliers, reason='too_few_inliers')
        world, image = world[mask], image[mask]
        hull = cv2.convexHull(image.astype(np.float32))
        world_area = cv2.contourArea(cv2.convexHull(world.astype(np.float32)))
        if cv2.contourArea(hull) < 0.003 * width * height or world_area < 0.012 * config.length_m * config.width_m:
            return Calibration(config, frame_size, candidates=count, inliers=inliers, reason='poor_spatial_support')
        if not has_projective_support(world, image):
            return Calibration(config, frame_size, candidates=count, inliers=inliers, reason='ambiguous_geometry')

        def quality(candidate: np.ndarray) -> tuple[np.ndarray, float] | None:
            # A horizon through the support or an ill-conditioned local mapping is unsafe.
            if not np.isfinite(candidate).all():
                return None
            denominators = np.column_stack((world, np.ones(inliers))) @ candidate[2]
            if np.any(np.abs(denominators) < 1e-8) or np.min(denominators) * np.max(denominators) <= 0:
                return None
            projected = project(candidate, world)
            residual = np.linalg.norm(projected - image, axis=1)
            if not np.isfinite(residual).all() or residual.max() > threshold:
                return None
            jacobian_x = project(candidate, world + [0.01, 0]) - projected
            jacobian_y = project(candidate, world + [0, 0.01]) - projected
            jacobian = np.stack((jacobian_x, jacobian_y), axis=-1)
            if not np.isfinite(jacobian).all():
                return None
            try:
                singular = np.linalg.svd(jacobian, compute_uv=False)
                inverse = np.linalg.inv(candidate)
            except np.linalg.LinAlgError:
                return None
            if np.any((singular[:, 0] > 0.01 * diagonal) | (singular[:, 1] < 0.0004)
                      | (singular[:, 0] > 30 * singular[:, 1])):
                return None
            scale = np.max(np.abs(inverse))
            if not np.isfinite(scale) or scale == 0:
                return None
            return inverse / scale, float(np.sqrt(np.mean(residual ** 2)))

        result = quality(matrix)
        if result is None:
            return Calibration(config, frame_size, candidates=count, inliers=inliers, reason='unreliable_geometry')
        stabilized = False
        if previous is not None and config.smoothing:
            current = project(matrix, world)
            prior = project(previous, world)
            if np.all(np.linalg.norm(prior - current, axis=1) <= 2 * threshold):
                blended = current + config.smoothing * (prior - current)
                smoothed, _ = cv2.findHomography(world, blended, 0)
                checked = quality(smoothed) if smoothed is not None else None
                if checked is not None:
                    matrix, result, stabilized = smoothed, checked, True
        self.previous = matrix
        inverse, rmse = result
        return Calibration(config, frame_size, inverse, hull, count, inliers, rmse,
                           float(scores[mask].mean()), stabilized)
