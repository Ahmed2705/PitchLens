"""Evaluate a frozen checkpoint and benchmark single-frame inference."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import ConfusionMatrix

from pitchlens.detector import CLASS_NAMES, Detector


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def targets_for(path: Path, width: int, height: int) -> dict[str, torch.Tensor]:
    label = path.parent.parent / "labels" / f"{path.stem}.txt"
    classes, boxes = [], []
    for line in label.read_text(encoding="utf-8").splitlines():
        category, x, y, w, h = map(float, line.split())
        classes.append(int(category))
        boxes.append([(x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height])
    return {"cls": torch.tensor(classes, dtype=torch.float32),
            "bboxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)}


def confusion_at_default_threshold(detector: Detector, images: list[Path]) -> list[list[int]]:
    """Compute class-agnostic IoU 0.5 matching at the fixed detection threshold."""
    matrix = ConfusionMatrix(names=dict(enumerate(CLASS_NAMES)))
    for path in images:
        frame = read_image(path)
        detections = detector.detect(frame)
        predictions = {"cls": torch.tensor([d.class_id for d in detections], dtype=torch.float32),
                       "conf": torch.tensor([d.confidence for d in detections], dtype=torch.float32),
                       "bboxes": torch.tensor([d.xyxy for d in detections], dtype=torch.float32).reshape(-1, 4)}
        height, width = frame.shape[:2]
        matrix.process_batch(predictions, targets_for(path, width, height), conf=detector.confidence, iou_thres=.5)
    return matrix.matrix.astype(int).tolist()


def fixed_threshold_scores(matrix: list[list[int]]) -> dict[str, dict[str, float | int]]:
    """Calculate class-wise precision and recall without tuning on the test set."""
    scores = {}
    for index, name in enumerate(CLASS_NAMES):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[index]) - true_positive
        false_negative = sum(row[index] for row in matrix) - true_positive
        scores[name] = {"precision": true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.,
                        "recall": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.,
                        "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative}
    return scores


def annotate(frame: np.ndarray, detector: Detector, destination: Path) -> None:
    canvas = frame.copy()
    for detection in detector.detect(frame):
        x1, y1, x2, y2 = (round(value) for value in detection.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
        label = f"{CLASS_NAMES[detection.class_id]} {detection.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 200, 255), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", canvas)
    if not ok:
        raise ValueError(f"Cannot encode sample output: {destination}")
    encoded.tofile(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="New directory for evaluation artifacts")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=.25, help="Fixed operating threshold for confusion counts")
    parser.add_argument("--benchmark-frames", type=int, default=50)
    args = parser.parse_args()
    data, weights, run = args.data.resolve(), args.weights.resolve(), args.run.resolve()
    if not data.is_file() or not weights.is_file():
        parser.error("Dataset YAML and checkpoint must exist")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this GPU evaluation")
    if args.benchmark_frames < 1:
        parser.error("benchmark-frames must be positive")
    if not 0 <= args.confidence <= 1 or args.image_size < 32 or args.image_size % 32:
        parser.error("Invalid confidence or image size")
    dataset = check_det_dataset(data, autodownload=False, split=args.split)
    images = sorted(Path(dataset[args.split]).glob("*.jpg"))
    validation_images = sorted(Path(dataset["val"]).glob("*.jpg"))
    if not images or not validation_images:
        parser.error("Evaluation and validation image directories must contain images")
    run.mkdir(parents=True, exist_ok=False)
    info = {"split": args.split, "data": str(data), "data_sha256": digest(data), "weights": str(weights),
            "weights_sha256": digest(weights), "image_size": args.image_size, "confidence": args.confidence,
            "note": "AP uses a confidence sweep; precision, recall, and confusion counts use the fixed threshold and IoU 0.5."}
    manifest = data.parent / "manifest.json"
    if manifest.exists():
        info["manifest_sha256"] = digest(manifest)
    (run / "evaluation_input.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    model = YOLO(str(weights))
    metrics = model.val(data=str(data), split=args.split, imgsz=args.image_size, device=0, batch=1,
                        workers=0, plots=False, project=str(run.parent), name=run.name, exist_ok=True, verbose=False)
    overall = {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)}
    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        _, _, ap50, ap = metrics.box.class_result(index)
        per_class[CLASS_NAMES[int(class_id)]] = {
            "ap50": float(ap50), "ap50_95": float(ap),
            "instances": int(metrics.nt_per_class[int(class_id)]), "images": int(metrics.nt_per_image[int(class_id)])}
    info.update(overall=overall, per_class=per_class, validation_speed_ms=metrics.speed)
    del model, metrics
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark the user-facing frame API on validation imagery, with no file IO
    # or drawing in the timed loop. Test images are never used to tune speed.
    torch.cuda.reset_peak_memory_stats()
    detector = Detector(weights, confidence=args.confidence, image_size=args.image_size, device="cuda")
    frame = read_image(validation_images[0])
    for _ in range(5):
        detector.detect(frame)
    torch.cuda.synchronize()
    durations = []
    for _ in range(args.benchmark_frames):
        start = time.perf_counter()
        detector.detect(frame)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    info["benchmark"] = {"frames": len(durations), "frame_shape": list(frame.shape),
                         "mean_ms": 1000 * sum(durations) / len(durations),
                         "median_ms": 1000 * float(np.median(durations)),
                         "fps": len(durations) / sum(durations),
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                         "gpu": torch.cuda.get_device_name(0)}
    matrix = confusion_at_default_threshold(detector, images)
    fixed_scores = fixed_threshold_scores(matrix)
    for name, values in fixed_scores.items():
        per_class.setdefault(name, {}).update(values)
    overall["precision"] = sum(values["precision"] for values in fixed_scores.values()) / len(CLASS_NAMES)
    overall["recall"] = sum(values["recall"] for values in fixed_scores.values()) / len(CLASS_NAMES)
    info["confusion_matrix"] = {"rows": "predicted class plus background", "columns": "true class plus background",
                                "classes": [*CLASS_NAMES, "background"], "iou_match": .5,
                                "confidence": args.confidence, "matrix": matrix}
    for index, path in enumerate(validation_images[:3], 1):
        annotate(read_image(path), detector, run / f"validation_sample_{index}.jpg")
    (run / "evaluation.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
