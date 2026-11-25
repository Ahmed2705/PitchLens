"""Run detector and tracker on a video, writing MP4 and JSONL without overwrite."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import cv2
from pitchlens.detector import CLASS_NAMES, Detector
from pitchlens.tracking import Tracker, TrackingConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New MP4 path; JSONL and summary use the same stem')
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/tracking.json'))
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    p.add_argument('--max-frames', type=int, default=None, help='Optional limit for a short smoke check')
    args = p.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        p.error('--max-frames must be positive')
    if args.output.suffix.lower() != '.mp4':
        p.error('--output must be an MP4 path')
    paths = [args.output, args.output.with_suffix('.jsonl'), args.output.with_suffix('.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Output video or sidecar already exists')
    if not args.output.parent.is_dir():
        raise ValueError('Output directory must exist')
    config = TrackingConfig(**json.loads(args.config.read_text(encoding='utf-8-sig')))
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
        detector = Detector(args.weights, confidence=config.person_low, class_confidence={3: config.ball_confidence}, image_size=960, device=args.device)
        tracker = Tracker(fps, config)
        # Reserve the output name atomically before OpenCV opens it for writing.
        with args.output.open('xb'):
            pass
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise ValueError('Cannot open video writer')
        frames = cuts = observed = predicted = 0
        start = time.perf_counter()
        with paths[1].open('x', encoding='utf-8') as records:
            while args.max_frames is None or frames < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                tracked = tracker.update(frame, detector.detect(frame))
                records.write(json.dumps(asdict(tracked), allow_nan=False) + '\n')
                cuts += tracked.camera_cut
                for item in tracked.tracks:
                    x1, y1, x2, y2 = map(round, item.xyxy)
                    color = (0, 220, 255) if item.class_id != 3 else ((0, 255, 0) if item.observed else (0, 140, 255))
                    label = f'{CLASS_NAMES[item.class_id]} {item.track_id}' if item.class_id != 3 else ('ball observed' if item.observed else 'ball predicted')
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, max(15, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
                    if item.class_id == 3:
                        observed += item.observed
                        predicted += not item.observed
                for a, b in zip(tracked.ball_trajectory, tracked.ball_trajectory[1:]):
                    color = (0, 255, 0) if a[2] and b[2] else (0, 140, 255)
                    cv2.line(frame, (round(a[0]), round(a[1])), (round(b[0]), round(b[1])), color, 2)
                writer.write(frame)
                frames += 1
        elapsed = time.perf_counter() - start
        if not frames:
            raise ValueError('No frames decoded')
        summary = {'input': str(args.input), 'frames': frames, 'source_fps': fps, 'width': width, 'height': height,
                   'elapsed_seconds': elapsed, 'processing_fps': frames / elapsed, 'camera_cuts': cuts,
                   'ball_observed_frames': observed, 'ball_predicted_frames': predicted, 'device': detector.device,
                   'image_size': detector.image_size, 'config': asdict(config)}
        with paths[2].open('x', encoding='utf-8') as file:
            json.dump(summary, file, indent=2, allow_nan=False)
        print(json.dumps(summary, indent=2))
    finally:
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()
