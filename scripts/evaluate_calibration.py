"""Compare pitch templates with frozen v9 predictions on local inputs."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import warnings
from zipfile import ZipFile

import cv2
import numpy as np

from pitchlens.calibration import PitchCalibrator, PitchConfig, project

from scripts.audit_foundation_datasets import SOCCERNET_SHA
from scripts.score_planar_calibration import (
    draw_markings, planar_markings, read_planar_labels, score_planar,
)

ARC_INDICES = np.array([10, 11, 18, 19])
VARIANTS = ('original', 'corrected')
SOCCERNET_IMAGES = 3212


def compare_calibrators(config):
    original = PitchCalibrator(config)
    original.world[ARC_INDICES, 1] = config.width_m / 2 + np.tile([-9.16, 9.16], 2)
    return {'original': original, 'corrected': PitchCalibrator(config)}


def roboflow_labels(data, width, height):
    values = np.array(data.split(), dtype=float)
    if values.shape != (101,) or not np.isfinite(values).all() or values[0] != 0:
        raise ValueError('Expected one Roboflow pitch with 32 pose triplets')
    points = values[5:].reshape(32, 3)
    if not np.isin(points[:, 2], [0, 1, 2]).all() or ((points[:, :2] < 0) | (points[:, :2] > 1)).any():
        raise ValueError('Invalid Roboflow keypoint coordinate or visibility')
    # Visibility zero often retains coordinates. Score only visibly labeled points.
    return points[:, :2] * [width, height], points[:, 2] == 2


def distances(expected, measured, keep):
    return {
        str(index + 1): float(error) if np.isfinite(error) else None
        for index, error in zip(np.flatnonzero(keep),
                                np.linalg.norm(expected[keep] - measured[keep], axis=1))
    }


def label_diagnostics(points, visible, observations, shape, pair):
    height, width = shape[:2]
    config = pair['corrected'].config
    available = np.zeros(32, dtype=bool)
    raw = {}
    if observations is not None:
        x, y, confidence = observations.T
        available = (np.isfinite(observations).all(axis=1)
                     & (confidence >= config.landmark_confidence) & (confidence <= 1)
                     & (x > 0) & (x < width) & (y > 0) & (y < height))
        raw = distances(points, observations[:, :2], visible & available)

    # Hold out all four arcs together. Both definitions use the same transform.
    control = np.column_stack((points, visible))
    check = np.zeros(32, dtype=bool)
    check[ARC_INDICES] = visible[ARC_INDICES]
    control[ARC_INDICES, 2] = 0
    oracle = PitchCalibrator(config).update(control, shape)
    result = {'visible_ids': (np.flatnonzero(visible) + 1).tolist(), 'raw_px': raw,
              'control': oracle.metadata(), 'held_out_arc_px': {}, 'label_fit': {}}
    annotations = np.column_stack((points, visible))
    for name, calibrator in compare_calibrators(config).items():
        result['label_fit'][name] = calibrator.update(annotations, shape).metadata()
        if oracle.image_to_pitch is not None:
            predicted = project(np.linalg.inv(oracle.image_to_pitch), calibrator.world)
            result['held_out_arc_px'][name] = distances(points, predicted, check)
    return result, available


def compare_frame(detector, frame, pair, cut=False, expected=None, markings=None, labels=None):
    observations = detector.detect(frame)
    evidence = None
    if observations is not None:
        evidence = np.where(np.isfinite(observations), observations.astype(object), None).tolist()
    row = {'shape': list(frame.shape[:2]), 'predictions': evidence, 'cut': cut}
    if expected is not None:
        row['annotation'], available = label_diagnostics(*expected, observations, frame.shape, pair)
    for name, calibrator in pair.items():
        fit = calibrator.update(observations, frame.shape, camera_cut=cut)
        result = row[name] = fit.metadata()
        result['image_to_pitch'] = None
        if fit.image_to_pitch is not None:
            result['image_to_pitch'] = fit.image_to_pitch.tolist()
            result['support_hull_px'] = fit.hull.reshape(-1, 2).tolist()
            if expected is not None:
                points, visible = expected
                predicted = project(np.linalg.inv(fit.image_to_pitch), pair['corrected'].world)
                result['label_px'] = distances(points, predicted, visible)
                result['unobserved_label_px'] = distances(points, predicted, visible & ~available)
        if labels is not None:
            result['planar'] = score_planar(markings, labels, fit)
    return row


def distribution(values):
    return {'count': len(values),
            'finite': sum(value is not None for value in values),
            'p50': percentile(values, 50), 'p90': percentile(values, 90)}


def percentile(values, percent):
    if not values:
        return None
    # Nearest rank keeps failed projections (None) above any finite distance.
    ordered = sorted(values, key=lambda value: float('inf') if value is None else value)
    return ordered[int(np.ceil(len(values) * percent / 100)) - 1]


def summarize(rows, mode):
    def errors(items, field):
        return [value for item in items for value in item.get(field, {}).values()]

    summary = {'frames': len(rows), 'cuts': [row['id'] for row in rows if row['cut']],
               'paired': dict(Counter(
                   f"{int(row['original']['valid'])}/{int(row['corrected']['valid'])}" for row in rows))}
    common = [row for row in rows if all(row[name]['valid'] for name in VARIANTS)]
    for name in VARIANTS:
        items = [row[name] for row in rows]
        accepted = [item for item in items if item['valid']]
        summary[name] = {
            'accepted': len(accepted), 'stabilized': sum(item['stabilized'] for item in accepted),
            'rejections': dict(Counter(item['reason'] for item in items if not item['valid'])),
            'landmark_counts': dict(Counter(item['landmarks'] for item in items)),
            'inlier_rmse_px': distribution([item['reprojection_rmse_px'] for item in accepted]),
        }
        if mode == 'roboflow':
            annotations = [row['annotation'] for row in rows]
            summary[name].update({
                'annotation_fit_accepted': sum(row['label_fit'][name]['valid'] for row in annotations),
                'held_out_arc_px': {str(index + 1): distribution([
                    row['held_out_arc_px'][name][str(index + 1)] for row in annotations
                    if str(index + 1) in row['held_out_arc_px'].get(name, {})]) for index in ARC_INDICES},
                'label_px': distribution(errors(accepted, 'label_px')),
                'common_label_px': distribution(errors([row[name] for row in common], 'label_px')),
                'unobserved_label_px': distribution(errors(accepted, 'unobserved_label_px')),
                'common_unobserved_label_px': distribution(errors([row[name] for row in common],
                                                                   'unobserved_label_px')),
            })
        if mode == 'soccernet':
            def planar(selected):
                return [value for item in selected for points in (item['planar'] or {}).values()
                        for value in points]

            matched = [item for item in accepted if item['planar']]
            values = planar(matched)
            eligible = [row for row in rows if row['usable_markings']]
            total = sum(row['usable_points'] for row in eligible)
            within = sum(value is not None and value <= 5 for value in values)
            summary[name].update({
                'scorable_accepted': len(matched),
                'annotated_frames': len(eligible),
                'distance_px': distribution(values),
                'common_distance_px': distribution(planar([row[name] for row in common])),
                'points_within_5px': within, 'points_in_accepted_frames': len(values),
                'all_eligible_points': total,
                'fraction_within_5px_on_accepted': within / len(values) if values else None,
                'fraction_within_5px_with_rejections': within / total if total else None,
            })
    if mode == 'roboflow':
        annotations = [row['annotation'] for row in rows]
        summary['raw_landmark_px'] = distribution(errors(annotations, 'raw_px'))
        summary['raw_per_id'] = {
            str(index): {'visible': sum(index in row['visible_ids'] for row in annotations),
                         **distribution([row['raw_px'][str(index)] for row in annotations
                                         if str(index) in row['raw_px']])}
            for index in range(1, 33)
        }
    return summary


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def soccernet_rows(args, detector, pair, markings):
    """Read images, annotations and match groups from the publisher-verified ZIP."""
    with args.input.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != SOCCERNET_SHA:
            raise ValueError('SoccerNet archive differs from the publisher SHA-256')
        stream.seek(0)
        with ZipFile(stream) as archive:
            index = json.loads(archive.read('valid/match_info.json'))
            images = {name for name in archive.namelist() if name.startswith('valid/') and name.endswith('.jpg')}
            if images != {f'valid/{image}' for image in index} or len(images) != SOCCERNET_IMAGES:
                raise ValueError('Expected the full SoccerNet validation image set and matching index')
            for image in sorted(index):
                path = f'valid/{image}'
                frame = decode_image(archive.read(path), path)
                height, width = frame.shape[:2]
                text = archive.read(Path(path).with_suffix('.json').as_posix())
                labels, excluded = read_planar_labels(json.loads(text), markings, width, height)
                row = compare_frame(detector, frame, pair, cut=True, markings=markings, labels=labels)
                row.update(id=path, split='valid', cut=False, label_sha256=hashlib.sha256(text).hexdigest(),
                           usable_markings=len(labels), usable_points=sum(map(len, labels.values())),
                           excluded=excluded, event=index[image])
                yield row


def decode_image(data, path):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f'Undecodable image: {path}')
    return frame


def roboflow_rows(args, detector, pair):
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    for item in sorted(audit['rows'], key=lambda row: row['path']):
        path = args.input / item['path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f'Image differs from the audit: {path}')
        frame = decode_image(data, path)
        height, width = frame.shape[:2]
        text = (args.input / item['label']).read_bytes()
        row = compare_frame(detector, frame, pair, cut=True, expected=roboflow_labels(text, width, height))
        # Static samples start fresh. This flag in the report is reserved for video cuts.
        row.update(id=item['path'], split=item['split'], cut=False,
                   label_sha256=hashlib.sha256(text).hexdigest())
        yield row


def video_decode_status(reported, decoded, stop_before):
    if stop_before is not None:
        status = 'frame_limit' if decoded == stop_before else 'early_end'
    elif reported <= 0:
        status = 'unreported_frame_count'
    else:
        status = 'complete' if decoded == reported else 'frame_count_mismatch'
    return {'reported_frames': reported if reported > 0 else None, 'decoded_frames': decoded,
            'stop_before': stop_before, 'status': status}


def video_rows(args, detector, pair, decoding):
    from pitchlens.tracking import Tracker

    capture = cv2.VideoCapture(str(args.input))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open {args.input}')
        reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker = Tracker(capture.get(cv2.CAP_PROP_FPS))
        frame_index = 0
        while args.stop_before is None or frame_index < args.stop_before:
            ok, frame = capture.read()
            if not ok:
                break
            timing = tracker.update(frame, [])
            row = compare_frame(detector, frame, pair, cut=timing.camera_cut)
            row.update(id=frame_index, segment=timing.segment)
            yield row
            frame_index += 1
        decoding.update(video_decode_status(reported, frame_index, args.stop_before))
    finally:
        capture.release()


def video_overlays(path, output, rows, markings):
    """Save a regular sample plus disagreements, sparse fits and cut neighborhoods."""
    selected = set(np.linspace(0, len(rows) - 1, min(12, len(rows)), dtype=int))
    for before, after in [(True, False), (False, True), (True, True), (False, False)]:
        choices = [row['id'] for row in rows if row['original']['valid'] == before
                   and row['corrected']['valid'] == after
                   and (before != after or row['corrected']['landmarks'] <= 7)]
        if choices:
            selected.update(choices[index] for index in np.linspace(0, len(choices) - 1, 6, dtype=int))
    for row in rows:
        if row['cut']:
            selected.update(index for index in (row['id'] - 1, row['id'], row['id'] + 1)
                            if 0 <= index < len(rows))
    capture = cv2.VideoCapture(str(path))
    try:
        for row in rows[:max(selected) + 1]:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f'Unable to re-decode frame {row["id"]}')
            if row['id'] not in selected:
                continue
            panels = []
            for name in VARIANTS:
                item = row[name]
                panel = frame.copy()
                draw_markings(panel, markings, item)
                for index, (x, y, confidence) in enumerate(row['predictions'] or [], 1):
                    if x is not None and y is not None and confidence is not None and confidence >= 0.5:
                        cv2.circle(panel, (round(x), round(y)), 5, (0, 220, 255), -1)
                        cv2.putText(panel, str(index), (round(x) + 5, round(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                panel = cv2.resize(panel, (800, round(800 * frame.shape[0] / frame.shape[1])))
                caption = f'{row["id"]} {name}: {item["inliers"]}/{item["landmarks"]} {item["reason"] or "accepted"}'
                cv2.putText(panel, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
                cv2.putText(panel, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                panels.append(panel)
            ok, data = cv2.imencode('.jpg', np.hstack(panels))
            if not ok:
                raise ValueError('Unable to encode comparison')
            (output / f'{row["id"]:06}.jpg').write_bytes(data.tobytes())
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('roboflow', 'video', 'soccernet'))
    parser.add_argument('--input', type=Path, required=True, help='Roboflow directory, video or publisher valid.zip')
    parser.add_argument('--audit', type=Path, help='Existing dataset-audit-v1 JSON for Roboflow')
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New local output directory')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--stop-before', type=int, help='Exclusive video frame limit')
    args = parser.parse_args()
    if (args.audit is not None) != (args.mode == 'roboflow'):
        parser.error('--audit is required in roboflow mode and valid only in that mode')
    if args.stop_before is not None and (args.mode != 'video' or args.stop_before < 1):
        parser.error('--stop-before must be a positive frame index in video mode')

    # Configure local Ultralytics paths before loading the same model as the runner.
    from pitchlens.detector import Detector  # noqa: F401
    from pitchlens.pitch_landmarks import CHECKPOINT_SHA256, PitchLandmarkDetector

    config = PitchConfig()
    pair = compare_calibrators(config)
    markings = planar_markings(config)
    detector = PitchLandmarkDetector(args.weights, args.device)
    args.output.mkdir()
    rows = []
    decoding = {}
    if args.mode == 'video':
        source = video_rows(args, detector, pair, decoding)
    elif args.mode == 'roboflow':
        source = roboflow_rows(args, detector, pair)
    else:
        source = soccernet_rows(args, detector, pair, markings)
    with (args.output / 'frames.jsonl').open('x', encoding='utf-8') as stream:
        for row in source:
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    if not rows:
        raise ValueError('No input frames decoded')
    if args.mode == 'video':
        if decoding['status'] == 'early_end':
            raise ValueError(f'Video ended before the requested frame limit: {decoding}')
        if decoding['status'] not in ('complete', 'frame_limit'):
            warnings.warn(f'Video coverage is unverified: {decoding}', stacklevel=1)
        video_overlays(args.input, args.output, rows, markings)
    report = summarize(rows, args.mode)
    if args.mode == 'video':
        report['video_decode'] = decoding
    if args.mode == 'soccernet':
        matches = sorted({(row['event']['league'], row['event']['season'], row['event']['match'])
                          for row in rows})
        report['by_match'] = {
            '/'.join(match): summarize(
                [row for row in rows if tuple(row['event'][key] for key in ('league', 'season', 'match')) == match],
                args.mode) for match in matches
        }
    report.update(mode=args.mode, config=asdict(config), device=args.device, opencv=cv2.__version__,
                  weights_sha256=CHECKPOINT_SHA256, source=str(args.input),
                  input_sha256=SOCCERNET_SHA if args.mode == 'soccernet' else sha256(args.audit or args.input))
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
