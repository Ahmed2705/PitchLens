"""Evaluate a frozen checkpoint and benchmark single-frame inference."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import os
import re
from tempfile import TemporaryDirectory
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix

from pitchlens.detector import CLASS_NAMES, Detector
from pitchlens.detection_manifest import Frame, label_for, labels_match, manifest_frames, sha256, verified_bytes


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def targets_for(path: Path, width: int, height: int) -> dict[str, torch.Tensor]:
    label = label_for(path)
    classes, boxes = [], []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
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


def group_frames(frames: list[Frame]) -> dict[str, list[Frame]]:
    groups = defaultdict(list)
    for frame in frames:
        groups[frame.group].append(frame)
    return dict(groups)


def average_precision(model: YOLO, frames: list[Frame], image_size: int, run: Path, *, device=0) -> dict:
    """Score explicit paths and check that the validator consumed each frame exactly once."""
    expected = {frame.image: frame for frame in frames}
    seen = Counter()

    class VerifiedValidator(DetectionValidator):
        def build_dataset(self, *args, **kwargs):
            dataset = super().build_dataset(*args, **kwargs)
            if Counter(map(Path, dataset.im_files)) != Counter(expected.keys()):
                raise ValueError("AP loader population differs from the verified image list")
            for label in dataset.labels:
                frame = expected[Path(label["im_file"])]
                _, raw = frame.verified_payload()
                if not labels_match(raw, label):
                    raise ValueError(f"AP loader changed annotations for {frame.image}")
            return dataset

        def update_metrics(self, preds, batch):
            paths = batch["im_file"]
            if len(preds) != len(paths):
                raise ValueError("AP prediction count differs from the verified batch")
            super().update_metrics(preds, batch)
            seen.update(map(Path, paths))

    run.mkdir()
    image_list = [str(path) for path in expected]
    data = frames[0].image.parent.parent / "data.yaml"
    data.write_text(yaml.safe_dump({"train": image_list, "val": image_list,
                                   "names": list(CLASS_NAMES)}), encoding="utf-8")
    if model.task != "detect" or model.names != dict(enumerate(CLASS_NAMES)):
        raise ValueError("AP checkpoint does not match the approved detection classes")
    metrics = model.val(validator=VerifiedValidator, data=str(data), split="val", imgsz=image_size,
                        device=device, batch=1, workers=0, plots=False, save_json=False, save_txt=False,
                        fraction=1., classes=None, single_cls=False, cache=False,
                        project=str(run.parent), name=run.name, exist_ok=True, verbose=False)
    if seen != Counter(expected.keys()):
        raise ValueError("AP metrics population differs from the verified image list")
    for frame in frames:
        frame.verified_payload()
    per_class = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        _, _, ap50, ap = metrics.box.class_result(index)
        per_class[CLASS_NAMES[int(class_id)]] = {
            "ap50": float(ap50), "ap50_95": float(ap),
            "instances": int(metrics.nt_per_class[int(class_id)]), "images": int(metrics.nt_per_image[int(class_id)])}
    return {"overall": {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)},
            "per_class": per_class, "validation_speed_ms": metrics.speed}


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


def metric_summaries(args, run: Path, checkpoint: Path, frames: list[Frame],
                     validation: list[Frame], info: dict) -> None:
    groups = group_frames(frames) if args.by_source_group else {}
    model = YOLO(str(checkpoint))
    info.update(average_precision(model, frames, args.image_size, run / "aggregate"))
    overall, per_class = info["overall"], info["per_class"]
    if groups:
        info["source_groups"] = {}
    for index, (group, paths) in enumerate(groups.items(), 1):
        group_run = run / f"group-{index}"
        copies = [frame.snapshot(checkpoint.parent / f"group-{index}") for frame in paths]
        summary = average_precision(model, copies, args.image_size, group_run)
        info["source_groups"][group] = {"images": len(paths), **summary}
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark the user-facing frame API on validation imagery, with no file IO
    # or drawing in the timed loop. Test images are never used to tune speed.
    torch.cuda.reset_peak_memory_stats()
    detector = Detector(checkpoint, confidence=args.confidence, image_size=args.image_size, device="cuda")
    frame = read_image(validation[0].image)
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
    group_matrices = {group: confusion_at_default_threshold(detector, [frame.image for frame in paths])
                      for group, paths in groups.items()}
    matrix = (np.sum(list(group_matrices.values()), axis=0).tolist() if group_matrices
              else confusion_at_default_threshold(detector, [frame.image for frame in frames]))
    fixed_scores = fixed_threshold_scores(matrix)
    for name, values in fixed_scores.items():
        per_class.setdefault(name, {}).update(values)
    overall["precision"] = sum(values["precision"] for values in fixed_scores.values()) / len(CLASS_NAMES)
    overall["recall"] = sum(values["recall"] for values in fixed_scores.values()) / len(CLASS_NAMES)
    info["confusion_matrix"] = {"rows": "predicted class plus background", "columns": "true class plus background",
                                "classes": [*CLASS_NAMES, "background"], "iou_match": .5,
                                "confidence": args.confidence, "matrix": matrix}
    for group, group_matrix in group_matrices.items():
        summary = info["source_groups"][group]
        summary["confusion_matrix"] = group_matrix
        for name, values in fixed_threshold_scores(group_matrix).items():
            summary["per_class"].setdefault(name, {}).update(values)
    for index, sample in enumerate(validation[:3], 1):
        annotate(read_image(sample.image), detector, run / f"validation_sample_{index}.jpg")
    for frame in [*frames, *([] if validation is frames else validation)]:
        frame.verified_payload()


def evaluate(args) -> dict:
    data, weights, run = args.data.resolve(), args.weights.resolve(), args.run.resolve()
    root = data.parent
    if run.is_relative_to(root):
        raise ValueError("Write evaluation artifacts outside the prepared dataset")
    if weights.suffix != ".pt" or not weights.is_file():
        raise ValueError("A local .pt checkpoint is required")
    if not re.fullmatch(r"[0-9a-f]{64}", args.manifest_sha256):
        raise ValueError("Supply the frozen manifest's SHA-256")
    rows = json.loads(verified_bytes(root / "manifest.json", args.manifest_sha256))
    data_bytes = data.read_bytes()
    config = yaml.safe_load(data_bytes)
    if config.get("names") not in (list(CLASS_NAMES), dict(enumerate(CLASS_NAMES))):
        raise ValueError("Dataset YAML does not match the approved detection classes")
    splits = {args.split, "val"}
    for split in splits:
        base = Path(config.get("path") or root)
        directory = root / base / config[split]
        if directory.resolve() != root / split / "images":
            raise ValueError(f"Dataset YAML does not match the manifest's {split} directory")
    sources = manifest_frames(root, rows, args.split)
    validation_sources = sources if args.split == "val" else manifest_frames(root, rows, "val")
    run.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="verified-", dir=run) as directory:
        snapshot = Path(directory)
        frames = [source.snapshot(snapshot / "aggregate") for source in sources]
        validation = (frames if args.split == "val" else
                      [source.snapshot(snapshot / "validation") for source in validation_sources])
        checkpoint = snapshot / "checkpoint.pt"
        weight_bytes = weights.read_bytes()
        checkpoint.write_bytes(weight_bytes)
        info = {"split": args.split, "data": str(data), "data_sha256": sha256(data_bytes),
                "manifest_sha256": args.manifest_sha256, "weights": str(weights),
                "weights_sha256": sha256(weight_bytes), "image_size": args.image_size, "confidence": args.confidence,
                "images": len(frames),
                "population": [source.image.relative_to(root).as_posix() for source in sources],
                "note": "AP uses a confidence sweep. Precision, recall, and confusion counts use the fixed threshold and IoU 0.5."}
        (run / "evaluation_input.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        metric_summaries(args, run, checkpoint, frames, validation, info)
    (run / "evaluation.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True, help="Previously recorded SHA-256 of manifest.json")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="New directory for evaluation artifacts")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--confidence", type=float, default=.25, help="Fixed operating threshold for confusion counts")
    parser.add_argument("--benchmark-frames", type=int, default=50)
    parser.add_argument("--by-source-group", action="store_true",
                        help="Report separate development metrics for each source group in manifest.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this GPU evaluation")
    if args.benchmark_frames < 1:
        parser.error("benchmark-frames must be positive")
    if args.by_source_group and args.split != "val":
        parser.error("Source grouping requires the development validation split")
    if not 0 <= args.confidence <= 1 or args.image_size < 32 or args.image_size % 32:
        parser.error("Invalid confidence or image size")
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
