"""Run detector and tracker on a video, writing MP4 and JSONL without overwrite."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from pitchlens.ball_detector import BallDetector, add_detection_arguments, combine_detections, video_detection_settings
from pitchlens.detector import BALL_CLASS_ID, CLASS_NAMES, Detector
from pitchlens.pass_detection import PassConfig, PassDetector
from pitchlens.possession import PossessionAnalytics, PossessionConfig
from pitchlens.tracking import Tracker, TrackingConfig
from pitchlens.team_classification import TeamClassifier, TeamConfig


def draw_possession(frame: np.ndarray, result: dict):
    owner = result['owner']
    label = 'Possession unknown'
    if owner is not None:
        team = f"Team {owner['team']}" if owner['team'] is not None else 'Team unknown'
        label = f"{CLASS_NAMES[owner['class_id']].capitalize()} {owner['track_id']} has ball | {team}"
    totals = result['totals']
    coverage = totals['team_possession_coverage_pct']
    detail = f'Team coverage {coverage:.1f}%' if coverage is not None else 'Team coverage unavailable'
    if totals['known_team_duration_seconds']:
        shares = ' / '.join(f"T{team['team']} {team['possession_pct']:.0f}%" for team in totals['teams'])
        detail = f'{shares} | {detail}'
    height, width = frame.shape[:2]
    for line, y in ((label, height - 30), (detail, height - 10)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_width = cv2.getTextSize(line, font, .52, 1)[0][0]
        scale = .52 * min(1., (width - 20) / max(1, text_width)) if width > 20 else .25
        cv2.putText(frame, line, (10, max(0, y)), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (10, max(0, y)), font, scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_pass_event(frame: np.ndarray, event: dict, teams: list[dict]):
    labels = {'likely_successful_pass': 'Likely pass', 'possible_turnover': 'Possible turnover',
              'unclassified_transfer': 'Unclassified transfer'}
    source, receiver = event['source']['track_id'], event['receiver']['track_id']
    label = f"{labels[event['classification']]} #{source} > #{receiver}"
    counts = ' / '.join(f"T{team['team']} {team['detected_passes']}" for team in teams)
    for line, offset in ((label, 70), (f'Detected passes: {counts}', 50)):
        origin = (10, max(0, frame.shape[0] - offset))
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, .52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New MP4 path; JSONL and summary use the same stem')
    add_detection_arguments(p)
    p.add_argument('--config', type=Path, default=Path('configs/tracking.json'))
    p.add_argument('--team-config', type=Path, default=None)
    p.add_argument('--team-references', type=Path, help='JSON mapping team 1 and 2 to paths of labeled crop images')
    p.add_argument('--siglip-snapshot', type=Path, help='Local pretrained SigLIP snapshot directory')
    p.add_argument('--pitch-weights', type=Path, help='Verified local Roboflow v9 pitch checkpoint for evaluation')
    p.add_argument('--pitch-config', type=Path, help='JSON overrides for the pitch dimensions and fit thresholds')
    p.add_argument('--pitch-temporal', action='store_true',
                   help='Enable temporal pitch recovery for local evaluation')
    p.add_argument('--movement-config', type=Path, help='JSON overrides for player movement estimation')
    p.add_argument('--possession-config', type=Path, help='JSON overrides for ball possession estimation')
    p.add_argument('--pass-config', type=Path, help='JSON overrides for ball transfer detection')
    p.add_argument('--max-frames', type=int, default=None, help='Optional limit for a short smoke check')
    args = p.parse_args()
    if (args.team_references is None) != (args.siglip_snapshot is None):
        p.error('--team-references and --siglip-snapshot must be supplied together')
    if args.max_frames is not None and args.max_frames < 1:
        p.error('--max-frames must be positive')
    if args.team_references and args.team_config:
        p.error('--team-config applies only to automatic mode')
    if args.pitch_config and not args.pitch_weights:
        p.error('--pitch-config requires --pitch-weights')
    if args.pitch_temporal and not args.pitch_weights:
        p.error('--pitch-temporal requires --pitch-weights')
    if args.movement_config and not args.pitch_weights:
        p.error('--movement-config requires --pitch-weights')
    if args.output.suffix.lower() != '.mp4':
        p.error('--output must be an MP4 path')
    paths = [args.output, args.output.with_suffix('.jsonl'), args.output.with_suffix('.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Output video or sidecar already exists')
    if not args.output.parent.is_dir():
        raise ValueError('Output directory must exist')
    config = TrackingConfig(**json.loads(args.config.read_text(encoding='utf-8-sig')))
    try:
        detection = video_detection_settings(args)
    except ValueError as exc:
        p.error(str(exc))
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise ValueError('Cannot open input video')
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError('Invalid video properties')
        if width % 2 or height % 2:
            raise ValueError('Video dimensions must be even for MP4 output')
        dedicated = detection.ball_weights is not None
        candidate_confidence = min(config.person_low, detection.people_confidence)
        detector = Detector(detection.people_weights, confidence=detection.people_confidence,
                            candidate_confidence=candidate_confidence, people_only=dedicated,
                            class_confidence=None if dedicated else {BALL_CLASS_ID: config.ball_confidence},
                            image_size=detection.people_image_size, device=args.device)
        ball_detector = BallDetector(detection.ball_weights, detection.ball_config, device=args.device) if dedicated else None
        tracker = Tracker(fps, config)
        team_colors = {1: (255, 120, 0), 2: (220, 0, 220)}
        possession_overrides = (
            json.loads(args.possession_config.read_text(encoding='utf-8-sig')) if args.possession_config else {}
        )
        if not isinstance(possession_overrides, dict):
            raise ValueError('Possession configuration must be a JSON object')
        possession = PossessionAnalytics(fps, tuple(team_colors), PossessionConfig(**possession_overrides))
        pass_overrides = json.loads(args.pass_config.read_text(encoding='utf-8-sig')) if args.pass_config else {}
        if not isinstance(pass_overrides, dict):
            raise ValueError('Pass configuration must be a JSON object')
        passes = PassDetector(fps, tuple(team_colors), PassConfig(**pass_overrides))
        if args.team_references:
            from pitchlens.assisted_team import AssistedTeamClassifier, SiglipEncoder
            reference_paths = json.loads(args.team_references.read_text(encoding='utf-8-sig'))
            if not isinstance(reference_paths, dict) or set(reference_paths) != {'1', '2'} or any(not isinstance(reference_paths[k], list) or not reference_paths[k] for k in ('1', '2')):
                raise ValueError('team references must contain nonempty lists for teams 1 and 2')
            references = {}
            for key in ('1', '2'):
                references[int(key)] = []
                for name in reference_paths[key]:
                    try:
                        payload = (args.team_references.parent / name).read_bytes()
                        img = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    except (OSError, TypeError, ValueError, cv2.error):
                        img = None
                    if img is None or not img.size:
                        raise ValueError(f'Cannot read reference crop: {name}')
                    references[int(key)].append(img)
            teams = AssistedTeamClassifier(SiglipEncoder(args.siglip_snapshot, detector.device), references)
        else:
            team_path = Path(__file__).resolve().parent.parent / 'configs' / 'team_classification.json'
            settings = json.loads(team_path.read_text(encoding='utf-8-sig'))
            if args.team_config:
                settings.update(json.loads(args.team_config.read_text(encoding='utf-8-sig')))
            team_config = TeamConfig(**settings)
            teams = TeamClassifier(team_config)
        if args.pitch_weights:
            from pitchlens.calibration import PitchCalibrator, PitchConfig
            from pitchlens.movement import MovementAnalytics, MovementConfig
            from pitchlens.pitch_landmarks import CHECKPOINT_SHA256, PitchLandmarkDetector
            from pitchlens.pitch_minimap import PitchMinimap

            overrides = json.loads(args.pitch_config.read_text(encoding='utf-8-sig')) if args.pitch_config else {}
            if not isinstance(overrides, dict):
                raise ValueError('Pitch configuration must be a JSON object')
            pitch = PitchConfig(**overrides)
            movement_overrides = (
                json.loads(args.movement_config.read_text(encoding='utf-8-sig')) if args.movement_config else {}
            )
            if not isinstance(movement_overrides, dict):
                raise ValueError('Movement configuration must be a JSON object')
            movement = MovementAnalytics(fps, MovementConfig(**movement_overrides))
            landmark_detector = PitchLandmarkDetector(args.pitch_weights, detector.device)
            calibrator = PitchCalibrator(pitch, fps=fps if args.pitch_temporal else None)
            minimap = PitchMinimap(pitch, (height, width))
            pitch_frames = dict.fromkeys(('direct', 'temporal', 'unavailable'), 0)
            pitch_counts = dict.fromkeys(('valid', 'estimated', 'unavailable'), 0)
        # Reserve the output name atomically before OpenCV opens it for writing.
        with args.output.open('xb'):
            pass
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise ValueError('Cannot open video writer')
        frames = cuts = observed = predicted = 0
        possession_elapsed = 0.
        pass_elapsed = 0.
        ball_elapsed = 0.
        detection_counts = Counter(dict.fromkeys(CLASS_NAMES, 0))
        recovery_candidates = 0
        displayed_event = None
        display_seconds = 0.
        start = time.perf_counter()
        with paths[1].open('x', encoding='utf-8') as records:
            while args.max_frames is None or frames < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                general_detections = detector.detect(frame)
                ball_detections = None
                if ball_detector is not None:
                    ball_start = time.perf_counter()
                    ball_detections = ball_detector.detect(frame)
                    ball_elapsed += time.perf_counter() - ball_start
                detections = combine_detections(general_detections, ball_detections)
                for item in detections:
                    if item.class_id != BALL_CLASS_ID and item.confidence < detection.people_confidence:
                        recovery_candidates += 1
                    else:
                        detection_counts[CLASS_NAMES[item.class_id]] += 1
                tracked = tracker.update(frame, detections)
                assignments = teams.update(frame, tracked)
                record = asdict(tracked)
                if ball_detections is not None:
                    record['raw_ball_detections'] = [asdict(detection) for detection in ball_detections]
                for item, team in zip(record['tracks'], assignments):
                    item['team'] = team
                possession_start = time.perf_counter()
                result = record['possession'] = possession.update(
                    tracked.frame_index, tracked.segment, tracked.camera_cut, record['tracks'],
                    timestamp_seconds=timestamp_seconds, ambiguous_ball=tracked.ball_observation_ambiguous,
                )
                possession_elapsed += time.perf_counter() - possession_start
                owner = result['owner']
                pass_start = time.perf_counter()
                pass_result = record['passes'] = passes.update(
                    tracked.frame_index, tracked.segment, tracked.camera_cut, record['tracks'],
                    timestamp_seconds=timestamp_seconds, ambiguous_ball=tracked.ball_observation_ambiguous,
                    possession=result,
                )
                pass_elapsed += time.perf_counter() - pass_start
                interval = result['timing']['elapsed_seconds']
                if tracked.camera_cut or interval is None:
                    displayed_event = None
                else:
                    display_seconds -= interval
                    if display_seconds <= 0:
                        displayed_event = None
                if pass_result['events']:
                    displayed_event = pass_result['events'][0]
                    display_seconds = 1.25
                if args.pitch_weights:
                    calibration = calibrator.update(landmark_detector.detect(frame), frame.shape,
                                                    camera_cut=tracked.camera_cut, frame=frame)
                    metadata = calibration.metadata()
                    record['calibration'] = metadata
                    pitch_frames[calibration.provenance] += 1
                    for item in record['tracks']:
                        if item['class_id'] in (0, 1):
                            item['pitch'] = calibration.map_contact(item['xyxy'])
                            pitch_counts[item['pitch']['status']] += 1
                    record['movement_timing'] = movement.update(
                        tracked.frame_index, tracked.segment, tracked.camera_cut, record['tracks'],
                        provenance=calibration.provenance, timestamp_seconds=timestamp_seconds,
                    )
                    minimap.draw(frame, record['tracks'], metadata['valid'])
                records.write(json.dumps(record, allow_nan=False) + '\n')
                cuts += tracked.camera_cut
                for item, team, output_track in zip(tracked.tracks, assignments, record['tracks']):
                    x1, y1, x2, y2 = map(round, item.xyxy)
                    color = (0, 220, 255) if item.class_id != 3 else ((0, 255, 0) if item.observed else (0, 140, 255))
                    if item.class_id in (0, 1):
                        color = team_colors.get(team, (180, 180, 180))
                    label = f'{CLASS_NAMES[item.class_id]} {item.track_id}' if item.class_id != 3 else ('ball observed' if item.observed else 'ball predicted')
                    if item.class_id in (0, 1):
                        label += f' team {team}' if team else ' unknown'
                        speed = output_track.get('movement', {}).get('speed_kmh')
                        if speed is not None:
                            label += f' {speed:.1f} km/h'
                    owns_ball = (owner is not None and item.class_id == owner['class_id']
                                 and item.track_id == owner['track_id'])
                    if owns_ball:
                        label += ' | ball'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if owns_ball else 2)
                    cv2.putText(frame, label, (x1, max(15, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                    if item.class_id == 3:
                        observed += item.observed
                        predicted += not item.observed
                for a, b in zip(tracked.ball_trajectory, tracked.ball_trajectory[1:]):
                    color = (0, 255, 0) if a[2] and b[2] else (0, 140, 255)
                    cv2.line(frame, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), color, 2)
                draw_possession(frame, result)
                if displayed_event is not None:
                    draw_pass_event(frame, displayed_event, pass_result['running']['teams'])
                writer.write(frame)
                frames += 1
        elapsed = time.perf_counter() - start
        if not frames:
            raise ValueError('No frames decoded')
        summary = {'input': str(args.input), 'frames': frames, 'source_fps': fps, 'width': width, 'height': height,
                   'elapsed_seconds': elapsed, 'processing_fps': frames / elapsed, 'camera_cuts': cuts,
                   'ball_observed_frames': observed, 'ball_predicted_frames': predicted, 'device': detector.device,
                   'image_size': detector.image_size, 'config': asdict(config), 'team_config': asdict(teams.config), 'team_mode': 'assisted' if args.team_references else 'automatic', 'team_references': str(args.team_references) if args.team_references else None, 'siglip_snapshot': str(args.siglip_snapshot) if args.siglip_snapshot else None}
        summary['possession'] = {
            'config': asdict(possession.config), 'analysis_ms_per_frame': 1000 * possession_elapsed / frames,
            **possession.summary(),
        }
        summary['passes'] = {
            'config': asdict(passes.config), 'analysis_ms_per_frame': 1000 * pass_elapsed / frames,
            **passes.finish(),
        }
        summary['detection'] = {
            'profile': detection.profile, 'people_weights': str(detection.people_weights),
            'people_image_size': detection.people_image_size, 'people_confidence': detection.people_confidence,
            'people_candidate_confidence': candidate_confidence,
            'operational_detection_counts': dict(detection_counts),
            'people_candidates_below_output_confidence': recovery_candidates,
        }
        if ball_detector is not None:
            summary['dedicated_ball'] = {
                'weights': str(detection.ball_weights),
                'config': asdict(detection.ball_config),
                'tracker_confidence': config.ball_confidence,
                'inference_ms_per_frame': 1000 * ball_elapsed / frames,
            }
        if detector.device.startswith('cuda'):
            summary['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(detector.device)
            summary['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(detector.device)
        if args.pitch_weights:
            summary['pitch'] = {'checkpoint_sha256': CHECKPOINT_SHA256, 'config': asdict(pitch),
                                'temporal_enabled': args.pitch_temporal and pitch.recovery_seconds > 0,
                                'valid_frames': pitch_frames['direct'] + pitch_frames['temporal'],
                                'provenance_frames': pitch_frames, 'coordinates': pitch_counts}
            summary['movement'] = {'config': asdict(movement.config), 'tracks': movement.summary()}
        with paths[2].open('x', encoding='utf-8') as file:
            json.dump(summary, file, indent=2, allow_nan=False)
        print(json.dumps(summary, indent=2))
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()
