"""Score ground-plane pitch markings in the SoccerNet annotation vocabulary."""

import math

import cv2
import numpy as np

from pitchlens.pitch_lines import STRAIGHT_MARKINGS


def planar_markings(config):
    """Named world polylines, with x from left to right and y from top to bottom."""
    vertices = config.landmarks()
    markings = {name: vertices[np.array(ids) - 1] for name, ids in STRAIGHT_MARKINGS.items()}
    radius = 9.15
    arc_angle = math.acos((16.5 - 11) / radius)
    mid = np.array([config.length_m, config.width_m]) / 2
    circle = np.linspace(0, 2 * math.pi, 361)
    arc = np.linspace(-arc_angle, arc_angle, 111)
    offsets = radius * np.column_stack((np.cos(arc), np.sin(arc)))
    markings['Circle central'] = mid + radius * np.column_stack((np.cos(circle), np.sin(circle)))
    markings['Circle left'] = vertices[8] + offsets
    markings['Circle right'] = vertices[21] - offsets
    return markings


def read_planar_labels(raw, markings, width, height):
    """Keep distinct finite points within one pixel of the viewport, clamping tiny overflow."""
    labels, excluded = {}, {}
    for name, points in raw.items():
        if name not in markings:
            excluded[name] = {'unmodeled_points': len(points)}
            continue
        xy = np.array([[point['x'], point['y']] for point in points], dtype=float).reshape(-1, 2)
        xy *= [width, height]
        inside = np.isfinite(xy).all(axis=1) & (xy >= -1).all(axis=1) & (xy <= [width + 1, height + 1]).all(axis=1)
        clipped = np.clip(xy[inside], [0, 0], [width, height])
        unique = np.unique(clipped, axis=0)
        counts = {'out_of_bounds_or_nonfinite': int((~inside).sum()),
                  'clamped': int(np.any(clipped != xy[inside], axis=1).sum()),
                  'duplicates': len(clipped) - len(unique)}
        # Even a single retained point can score an already estimated marking.
        if len(unique):
            labels[name] = unique
        if not len(unique) or any(counts.values()):
            excluded[name] = counts
    return labels, excluded


def visible_segments(path, camera, width, height):
    """Clip each world segment in homogeneous image space, before dividing by depth."""
    points = np.column_stack((path, np.ones(len(path)))) @ camera.T
    x, y, z = points.T
    margins = np.column_stack((z - 1e-9, x, width * z - x, y, height * z - y))
    first, last = margins[:-1], margins[1:]
    entering = (first < 0) & (last >= 0)
    leaving = (first >= 0) & (last < 0)
    crossing = entering | leaving
    intersections = np.divide(first, first - last, out=np.zeros_like(first), where=crossing)
    start = np.where(entering, intersections, 0).max(axis=1)
    end = np.where(leaving, intersections, 1).min(axis=1)
    visible = ~((first < 0) & (last < 0)).any(axis=1) & (start <= end)
    a = points[:-1][visible]
    delta = points[1:][visible] - a
    ends = a[:, None, :] + np.column_stack((start[visible], end[visible]))[:, :, None] * delta[:, None, :]
    return ends[:, :, :2] / ends[:, :, 2, None]


def camera_on_support(image_to_pitch, support_point):
    # Only project the camera-facing branch. Inverting preserves an arbitrary scale.
    camera = np.linalg.inv(image_to_pitch)
    camera /= np.max(np.abs(camera))
    depth_sign = np.sign(image_to_pitch[2] @ [*support_point, 1])
    return camera * depth_sign


def point_distances(points, segments):
    if not len(segments):
        return [None] * len(points)
    start, end = segments.transpose(1, 0, 2)
    delta = end - start
    squared = np.sum(delta**2, axis=1)
    offsets = points[:, None, :] - start
    dots = np.sum(offsets * delta, axis=2)
    t = np.clip(np.divide(dots, squared, out=np.zeros_like(dots), where=squared > 0), 0, 1)
    return np.linalg.norm(offsets - t[:, :, None] * delta, axis=2).min(axis=1).tolist()


def score_planar(markings, labels, fit):
    """Use one global orientation per frame, allowing SoccerNet's 180-degree symmetry."""
    if fit.image_to_pitch is None:
        return None
    camera = camera_on_support(fit.image_to_pitch, fit.hull.reshape(-1, 2)[0])
    width, height = fit.frame_size
    size = np.array([fit.config.length_m, fit.config.width_m])

    def score(rotated):
        paths = {name: size - markings[name] if rotated else markings[name] for name in labels}
        return {name: point_distances(points, visible_segments(paths[name], camera, width, height))
                for name, points in labels.items()}

    def orientation_key(result):
        values = [value for distances in result.values() for value in distances]
        # Prefer more 5 px matches, then smaller distance capped at image diagonal.
        diagonal = math.hypot(width, height)
        return (-sum(value is not None and value <= 5 for value in values),
                sum(diagonal if value is None else value for value in values))

    return min((score(False), score(True)), key=orientation_key)


def draw_markings(frame, markings, fit):
    if not fit['valid']:
        return
    inverse = np.array(fit['image_to_pitch'])
    hull = np.array(fit['support_hull_px'])
    camera = camera_on_support(inverse, hull[0])
    height, width = frame.shape[:2]
    for path in markings.values():
        for start, end in np.rint(visible_segments(path, camera, width, height)).astype(int):
            cv2.line(frame, start, end, (255, 100, 0), 2, cv2.LINE_AA)
    cv2.polylines(frame, [np.rint(hull).astype(np.int32)], True, (0, 200, 200), 1, cv2.LINE_AA)
