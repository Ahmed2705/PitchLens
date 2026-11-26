"""Detect hard cuts by checking whether neighboring frames share a camera view."""

from __future__ import annotations

import cv2
import numpy as np


_SIZE = (160, 90)
_MIN_CHANGE = .055
_MIN_LUMA_STD = 5.


def thumbnail(frame: np.ndarray) -> np.ndarray:
    """Keep color evidence and discard fine texture and encoding noise."""
    return cv2.resize(frame, _SIZE, interpolation=cv2.INTER_AREA)


def _similar_view(before: np.ndarray, after: np.ndarray, min_correlation: float) -> bool:
    # Operate on pixel rows, including only overlapping pixels after a warp.
    a = cv2.cvtColor(before.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb)[:, 0]
    b = cv2.cvtColor(after.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb)[:, 0]
    if cv2.absdiff(a[:, 1:], b[:, 1:]).mean() > .08 * 255:
        return False
    x, y = a[:, 0].astype(np.float32), b[:, 0].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    scale = np.linalg.norm(x) * np.linalg.norm(y)
    return bool(scale > 0 and np.dot(x, y) >= min_correlation * scale)


def _continuous_motion(before: np.ndarray, after: np.ndarray, gray_before: np.ndarray,
                       gray_after: np.ndarray) -> bool:
    # Fit one translation, rotation and scale from reversible feature tracks.
    points = cv2.goodFeaturesToTrack(gray_before, 96, .015, 6)
    if points is None or len(points) < 8:
        return False
    target, forward, _ = cv2.calcOpticalFlowPyrLK(gray_before, gray_after, points, None)
    back, backward, _ = cv2.calcOpticalFlowPyrLK(gray_after, gray_before, target, None)
    reversible = (forward.ravel().astype(bool) & backward.ravel().astype(bool)
                  & (np.linalg.norm(points - back, axis=2).ravel() <= 1.))
    if np.count_nonzero(reversible) < 8:
        return False
    origins = points[reversible]
    transform, inliers = cv2.estimateAffinePartial2D(origins, target[reversible], ransacReprojThreshold=1.5)
    if transform is None or np.count_nonzero(inliers) < 8:
        return False
    support = cv2.convexHull(origins[inliers.ravel().astype(bool)])
    # Features confined to a scorebug or another small overlay cannot veto a cut.
    if cv2.contourArea(support) < .05 * gray_before.size:
        return False
    if not .65 <= np.linalg.norm(transform[:, 0]) <= 1.55:
        return False
    overlap = cv2.warpAffine(np.full_like(gray_before, 255), transform, _SIZE) == 255
    if overlap.mean() < .55:
        return False
    aligned = cv2.warpAffine(before, transform, _SIZE)
    return _similar_view(aligned[overlap], after[overlap], .65)


def is_hard_cut(before: np.ndarray, after: np.ndarray, low_detail_threshold: float) -> bool:
    """Separate abrupt appearance changes from shared scene structure and motion."""
    change = cv2.absdiff(before, after).mean() / 255
    if change < _MIN_CHANGE:
        return False
    gray_before, gray_after = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in (before, after))
    if max(gray_before.std(), gray_after.std()) < _MIN_LUMA_STD:
        # No spatial evidence is available to distinguish exposure from a new shot.
        return bool(change > low_detail_threshold)
    if _similar_view(before, after, .94):
        return False
    return not _continuous_motion(before, after, gray_before, gray_after)
