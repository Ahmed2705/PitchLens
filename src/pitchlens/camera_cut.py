"""Detect hard cuts by checking whether neighboring frames share a camera view."""

from __future__ import annotations

import cv2
import numpy as np


_SIZE = (160, 90)
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


def _continuous_translation(before: np.ndarray, after: np.ndarray, gray_before: np.ndarray,
                            gray_after: np.ndarray, change: float) -> bool:
    # A view of grass and straight field markings may have no trackable corners.
    # Match the center by translation, then check the full shared area.
    height, width = gray_before.shape
    margin_x, margin_y = width // 5, height // 5
    center = gray_before[margin_y:-margin_y, margin_x:-margin_x]
    if center.std() < _MIN_LUMA_STD:
        return False
    correlation = cv2.matchTemplate(gray_after, center, cv2.TM_CCOEFF_NORMED)
    _, _, _, (x, y) = cv2.minMaxLoc(correlation)
    dx, dy = x - margin_x, y - margin_y
    tx, ty = max(0, dx), max(0, dy)
    sx, sy = max(0, -dx), max(0, -dy)
    w, h = width - abs(dx), height - abs(dy)
    aligned, current = before[sy:sy+h, sx:sx+w], after[ty:ty+h, tx:tx+w]
    return (cv2.absdiff(aligned, current).mean() / 255 <= .55 * change
            and _similar_view(aligned, current, .85))


def is_hard_cut(before: np.ndarray, after: np.ndarray, *, appearance_threshold: float,
                low_detail_threshold: float) -> bool:
    """Separate abrupt appearance changes from shared scene structure and motion."""
    if low_detail_threshold == 1:
        return False
    change = cv2.absdiff(before, after).mean() / 255
    if change <= min(appearance_threshold, low_detail_threshold):
        return False
    gray_before, gray_after = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in (before, after))
    if max(gray_before.std(), gray_after.std()) < _MIN_LUMA_STD:
        # No spatial evidence is available to distinguish exposure from a new shot.
        return bool(change > low_detail_threshold)
    if change <= appearance_threshold or _similar_view(before, after, .94):
        return False
    return not (_continuous_motion(before, after, gray_before, gray_after)
                or _continuous_translation(before, after, gray_before, gray_after, change))
