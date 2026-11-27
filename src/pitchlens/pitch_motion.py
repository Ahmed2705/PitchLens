"""Track short-range camera motion using features on the supported pitch."""

import cv2
import numpy as np


def motion_image(frame: np.ndarray, scale: float) -> np.ndarray:
    # Keep an independent copy: the runner draws on the original frame.
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def pitch_features(frame: np.ndarray, gray: np.ndarray, hull: np.ndarray) -> np.ndarray | None:
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 255)
    blue, green, red = cv2.split(frame)
    grass = (green.astype(np.int16) - np.maximum(blue, red) >= 7).astype(np.uint8)
    # Include paint and textured turf, but avoid patches dominated by foreground.
    nearby_grass = cv2.boxFilter(grass.astype(np.float32), -1, (13, 13))
    mask[nearby_grass < 0.65] = 0
    return cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01,
                                   minDistance=8, mask=mask)


def estimate_pitch_motion(anchor: np.ndarray, current: np.ndarray, features: np.ndarray,
                          min_tracks: int, min_inliers: int, min_ratio: float
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return anchor-to-current similarity and the supporting point pairs."""
    if len(features) < min_tracks:
        return None
    options = dict(winSize=(21, 21), maxLevel=3)
    tracked, forward, _ = cv2.calcOpticalFlowPyrLK(anchor, current, features, None, **options)
    if tracked is None or forward is None:
        return None
    source = features[forward.ravel() == 1]
    target = tracked[forward.ravel() == 1]
    height, width = current.shape
    inside = ((target[:, 0] >= 5) & (target[:, 0] < [width - 5, height - 5])).all(axis=1)
    source, target = source[inside], target[inside]
    if len(source) < min_tracks:
        return None
    returned, backward, _ = cv2.calcOpticalFlowPyrLK(current, anchor, target, None, **options)
    if returned is None or backward is None:
        return None
    round_trip = np.linalg.norm((returned - source)[:, 0], axis=1)
    keep = (backward.ravel() == 1) & (round_trip <= 1.0)
    source, target = source[keep, 0], target[keep, 0]
    if len(source) < min_tracks:
        return None
    # A similarity handles short pans, zoom and roll with fewer degrees of
    # freedom than a homography. Current-frame paint must validate the result.
    affine, mask = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
                                               ransacReprojThreshold=2.)
    if affine is None or mask is None or not np.isfinite(affine).all():
        return None
    keep = mask.ravel().astype(bool)
    if keep.sum() < min_inliers or keep.mean() < min_ratio:
        return None
    source, target = source[keep], target[keep]
    scale = np.linalg.norm(affine[:, 0])
    angle = abs(np.arctan2(affine[1, 0], affine[0, 0]))
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    displacement = np.linalg.norm(corners @ affine[:, :2].T + affine[:, 2] - corners, axis=1)
    if not (0.8 <= scale <= 1.25 and angle <= 0.10
            and displacement.max() <= 0.18 * np.hypot(width, height)):
        return None
    return np.vstack((affine, [0, 0, 1])), source, target
