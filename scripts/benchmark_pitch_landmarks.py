"""Sample an existing video uniformly and save landmark overlays and timings."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from pitchlens.pitch_landmarks import PitchLandmarkDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('--samples must be at least 2')
    detector = PitchLandmarkDetector(args.weights, args.device)
    cap = cv2.VideoCapture(str(args.input))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or count < 1:
            raise ValueError('Cannot open video or determine frame count')
        args.output.mkdir()
        samples = set(np.linspace(0, count - 1, args.samples).astype(int).tolist())
        measurements = []
        for index in range(count):
            ok, frame = cap.read()
            if not ok:
                break
            if index not in samples:
                continue
            start = time.perf_counter()
            points = detector.detect(frame)
            elapsed = (time.perf_counter() - start) * 1000
            visible = []
            if points is not None:
                for landmark, (x, y, score) in enumerate(points, 1):
                    if not np.isfinite((x, y, score)).all() or score < 0.5:
                        continue
                    point = (round(float(x)), round(float(y)))
                    visible.append([landmark, float(x), float(y), float(score)])
                    cv2.circle(frame, point, 5, (0, 220, 255), -1)
                    cv2.putText(frame, str(landmark), (point[0] + 5, point[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            ok, encoded = cv2.imencode('.jpg', frame)
            if not ok:
                raise ValueError(f'Cannot encode sample frame {index}')
            (args.output / f'{index:06}.jpg').write_bytes(encoded.tobytes())
            measurements.append({'frame': index, 'inference_ms': elapsed, 'landmarks': visible})
        if not measurements:
            raise ValueError('No representative frames decoded')
        (args.output / 'samples.json').write_text(json.dumps(measurements, indent=2, allow_nan=False), encoding='utf-8')
        print(f'{len(measurements)} samples, visible counts {[len(m["landmarks"]) for m in measurements]}, '
              f'warm median {np.median([m["inference_ms"] for m in measurements[1:] or measurements]):.1f} ms')
    finally:
        cap.release()


if __name__ == '__main__':
    main()
