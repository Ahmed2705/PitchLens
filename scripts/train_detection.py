"""Run a versioned YOLO26s detection experiment on the local CUDA GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import torch
from ultralytics import YOLO


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared data.yaml path")
    parser.add_argument("--weights", type=Path, required=True, help="Local pretrained yolo26s.pt")
    parser.add_argument("--run", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "detection_train.json")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch using 5% of training images")
    args = parser.parse_args()
    data, weights, run = args.data.resolve(), args.weights.resolve(), args.run.resolve()
    for path in (data, weights, args.config):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this training configuration requires the approved NVIDIA GPU")
    parameters = json.loads(args.config.read_text(encoding="utf-8"))
    if args.smoke:
        parameters.update(epochs=1, fraction=0.05, save_period=-1)
    run.mkdir(parents=True, exist_ok=False)
    info = {"weights": str(weights), "weights_sha256": digest(weights), "data": str(data),
            "data_sha256": digest(data), "parameters": parameters, "smoke": args.smoke,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)}
    for filename in ("manifest.json", "report.json"):
        artifact = data.parent / filename
        if artifact.is_file():
            info[f"{filename}_sha256"] = digest(artifact)
    (run / "pitchlens_run.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    model = YOLO(str(weights))
    epoch_state = {"active": False}

    def start_epoch(trainer) -> None:
        torch.cuda.reset_peak_memory_stats(trainer.device)
        epoch_state["active"] = True

    def log_memory(trainer) -> None:
        # Ultralytics also fires on_fit_epoch_end during final best-checkpoint
        # validation; that pass must not be reported as a training epoch.
        if not epoch_state["active"]:
            return
        epoch_state["active"] = False
        torch.cuda.synchronize(trainer.device)
        free, total = torch.cuda.mem_get_info(trainer.device)
        path = run / "gpu_memory.csv"
        needs_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            if needs_header:
                writer.writerow(["epoch", "peak_allocated_bytes", "peak_reserved_bytes", "free_bytes", "total_bytes"])
            writer.writerow([trainer.epoch + 1, torch.cuda.max_memory_allocated(trainer.device),
                             torch.cuda.max_memory_reserved(trainer.device), free, total])

    model.add_callback("on_train_epoch_start", start_epoch)
    model.add_callback("on_fit_epoch_end", log_memory)
    model.train(data=str(data), project=str(run.parent), name=run.name, exist_ok=True, **parameters)


if __name__ == "__main__":
    main()
