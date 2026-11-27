"""Annotate one video with the trained frame detector."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from pitchlens.ball_detector import BallDetector, add_detection_arguments, combine_detections, video_detection_settings
from pitchlens.detector import CLASS_NAMES, Detector


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    add_detection_arguments(p)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        settings = video_detection_settings(args)
    except ValueError as exc:
        p.error(str(exc))
    dedicated = settings.ball_weights is not None
    detector = Detector(settings.people_weights, image_size=settings.people_image_size,
                        confidence=settings.people_confidence, device=args.device, people_only=dedicated)
    ball_detector = BallDetector(settings.ball_weights, settings.ball_config, device=args.device) if dedicated else None
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise ValueError('Cannot open video')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError('Invalid video properties')
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise ValueError('Cannot open MP4 writer')
    counts = Counter()
    ball_frames = 0
    frames = 0
    samples = []
    start = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            people = detector.detect(frame)
            ball = ball_detector.detect(frame) if ball_detector is not None else None
            detections = combine_detections(people, ball)
            ball_frames += int(any(d.class_id == 3 for d in detections))
            for d in detections:
                counts[CLASS_NAMES[d.class_id]] += 1
                x1, y1, x2, y2 = map(round, d.xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(frame, f'{CLASS_NAMES[d.class_id]} {d.confidence:.2f}', (x1, max(15, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1)
            writer.write(frame)
            if expected and frames in {0, expected//4, expected//2, 3*expected//4, expected-1}:
                samples.append(cv2.resize(frame, (640, 360)))
            frames += 1
    finally:
        cap.release()
        writer.release()
    elapsed = time.perf_counter() - start
    if not frames:
        raise ValueError('No frames decoded')
    if samples:
        cv2.imwrite(str(args.output.with_suffix('.jpg')), np.vstack(samples))
    result = {'frames': frames, 'reported_input_frames': expected, 'source_fps': fps, 'width': width, 'height': height,
              'elapsed_seconds': elapsed, 'processing_fps': frames/elapsed, 'detections': dict(counts), 'frames_with_ball_detection': ball_frames,
              'device': detector.device, 'image_size': detector.image_size, 'confidence': detector.confidence,
              'detector_profile': settings.profile, 'weights': str(settings.people_weights)}
    if settings.ball_config is not None:
        result['dedicated_ball'] = {'weights': str(settings.ball_weights), 'image_size': settings.ball_config.image_size,
                                    'confidence': settings.ball_config.confidence}
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
