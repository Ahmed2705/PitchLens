"""Run a versioned YOLO26s detection experiment on the local CUDA GPU."""

from __future__ import annotations

import argparse
import csv
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

PROJECT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT / "outputs" / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "outputs" / "matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import unwrap_model

from pitchlens.detection_manifest import labels_match, manifest_frames, verified_bytes
from pitchlens.detector import CLASS_NAMES


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_resumable(trainer) -> None:
    # Preserve the complete snapshot before Ultralytics strips last.pt.
    path = trainer.last.with_name("last-resumable.pt")
    temporary = path.with_suffix(".tmp")
    shutil.copyfile(trainer.last, temporary)
    temporary.replace(path)


def checkpoint_manifest(checkpoint: dict, expected: str) -> None:
    recorded = checkpoint.get("pitchlens_manifest_sha256")
    if not isinstance(recorded, str) or not re.fullmatch(r"[0-9a-f]{64}", recorded):
        raise ValueError("Checkpoint has no verifiable PitchLens training manifest provenance")
    if recorded != expected:
        raise ValueError("Supplied manifest SHA-256 differs from the checkpoint's training manifest")


class StrictEarlyStopping:
    """Use one best epoch for early stopping and checkpoint selection."""

    def __init__(self, patience: int):
        self.best_fitness = None
        self.best_epoch = 0
        self.patience = patience
        self.epochs_without_improvement = 0
        self.possible_stop = False

    def __call__(self, epoch: int, fitness: float | None) -> bool:
        if fitness is not None and math.isfinite(fitness):
            score = float(fitness)
            if self.best_fitness is None or score > self.best_fitness:
                self.best_fitness, self.best_epoch = score, epoch
        if self.best_epoch:
            self.epochs_without_improvement = epoch - self.best_epoch
        self.possible_stop = bool(self.best_epoch and self.patience > 0
                                  and self.epochs_without_improvement >= self.patience - 1)
        return self.exhausted

    @property
    def exhausted(self) -> bool:
        return self.patience > 0 and self.epochs_without_improvement >= self.patience


def saved_stopper(checkpoint: dict) -> dict:
    state = checkpoint.get("stopper")
    fields = vars(StrictEarlyStopping(0))
    if not isinstance(state, dict) or state.keys() != fields.keys():
        raise ValueError("Checkpoint lacks complete early-stopping state")
    epoch, wait, patience = (state[key] for key in ("best_epoch", "epochs_without_improvement", "patience"))
    fitness = state["best_fitness"]
    if (type(epoch) is not int or not 0 <= epoch <= checkpoint["epoch"] + 1
            or type(wait) is not int or wait != (checkpoint["epoch"] + 1 - epoch if epoch else 0)
            or type(patience) is not int or patience < 0 or patience != checkpoint["train_args"]["patience"]
            or type(state["possible_stop"]) is not bool
            or state["possible_stop"] != bool(epoch and patience > 0 and wait >= patience - 1)):
        raise ValueError("Checkpoint has inconsistent early-stopping state")
    if epoch:
        if type(fitness) is not float or not math.isfinite(fitness) or fitness != checkpoint["best_fitness"]:
            raise ValueError("Checkpoint has inconsistent early-stopping fitness")
    elif fitness is not None or checkpoint["best_fitness"] is not None or checkpoint["train_args"].get("val", True):
        raise ValueError("Checkpoint has inconsistent pre-validation state")
    return state


def stage_saved_best(checkpoint: dict, local_weights: Path, original_weights: Path, snapshot: Path,
                     *, best_source: Path | None = None) -> Path | None:
    """Resolve the exact selected model, including an older best for a periodic checkpoint."""
    state = saved_stopper(checkpoint)
    epoch = state["best_epoch"]
    expected = checkpoint["pitchlens_best_sha256"]
    if epoch in (0, checkpoint["epoch"] + 1):
        if expected is not None:
            raise ValueError("Checkpoint has an inconsistent best-model reference")
        return local_weights if epoch else None
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Checkpoint has no verifiable historical best-model reference")
    if best_source is None:
        archived = original_weights.with_name(f"best-{expected}.pt")
        best_source = archived if archived.exists() else original_weights.with_name("best.pt")
    if not best_source.is_file() or best_source.suffix != ".pt":
        raise ValueError("Historical best checkpoint is missing. Supply its path with --best-weights.")
    payload = best_source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Historical best checkpoint SHA-256 mismatch. Supply the matching --best-weights.")
    local_best = snapshot / "historical-best.pt"
    local_best.write_bytes(payload)
    _, best = load_checkpoint(local_best)
    checkpoint_manifest(best, checkpoint["pitchlens_manifest_sha256"])
    best_state = saved_stopper(best)
    if (best["epoch"] + 1 != epoch or best_state["best_epoch"] != epoch
            or best_state["best_fitness"] != state["best_fitness"] or best.get("pitchlens_best_sha256", "") is not None):
        raise ValueError("Historical best checkpoint state does not match the saved selection")
    return local_best


class PitchLensTrainer(DetectionTrainer):
    def __init__(self, *args, manifest_sha256: str, **kwargs):
        self.manifest_sha256 = manifest_sha256
        self.best_sha256 = None
        super().__init__(*args, **kwargs)
        self.add_callback("on_pretrain_routine_end", self.verify_training_snapshot)

    def verify_training_snapshot(self, trainer):
        root = Path(trainer.args.data).parent
        rows = json.loads(verified_bytes(root / "manifest.json", self.manifest_sha256))
        for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
            frames = manifest_frames(root, rows, split)
            fraction = loader.dataset.fraction
            count = fraction if isinstance(fraction, int) else max(1, round(len(frames) * fraction))
            expected = {frame.image for frame in frames[:count]}
            loaded = list(map(Path, loader.dataset.im_files))
            if len(loaded) != len(expected) or set(loaded) != expected:
                raise ValueError(f"Training loader population differs from the verified {split} snapshot")
            annotations = {Path(label["im_file"]): label for label in loader.dataset.labels}
            # Reject loader repairs, including corrections held only in memory.
            for frame in frames:
                _, raw = frame.verified_payload()
                if frame.image in expected and not labels_match(raw, annotations[frame.image]):
                    raise ValueError(f"Training loader changed annotations for {frame.image}")

    def check_resume(self, overrides):
        super().check_resume(overrides)
        # BaseTrainer writes args.yaml and creates weights immediately after this hook.
        if not self.args.save:
            raise ValueError("Checkpoint saving must stay enabled to preserve every selected best")
        if getattr(self.args, "save_dir", None) != overrides["save_dir"] or (
            overrides.get("resume") and (not self.resume or self.args.resume != overrides["resume"])
        ):
            raise RuntimeError("Ultralytics did not honor the isolated run or resume arguments")

    def resume_training(self, ckpt):
        self.stopper = StrictEarlyStopping(self.args.patience)
        super().resume_training(ckpt)

    def _load_checkpoint_state(self, ckpt):
        checkpoint_manifest(ckpt, self.manifest_sha256)
        saved_stopper(ckpt)
        super()._load_checkpoint_state(ckpt)

    def save_model(self):
        """Publish the native checkpoint with PitchLens provenance and stopping state."""
        last, best, directory = self.last, self.best, self.wdir
        self.best_fitness = self.stopper.best_fitness
        improved = self.stopper.best_epoch == self.epoch + 1
        period = self.save_period
        with TemporaryDirectory(prefix="checkpoint-", dir=directory) as temporary:
            staging = Path(temporary)
            self.last, self.best, self.wdir = staging / "last.pt", staging / "best.pt", staging
            self.save_period = -1
            try:
                if not super().save_model():
                    return False
                payload = torch.load(self.last, map_location="cpu", weights_only=False)
            finally:
                self.last, self.best, self.wdir, self.save_period = last, best, directory, period
            payload["pitchlens_manifest_sha256"] = self.manifest_sha256
            payload["stopper"] = vars(self.stopper).copy()
            saved_stopper(payload)
            # A checkpoint that was selected at this epoch is its own best.
            # Later checkpoints record the exact bytes of that earlier file.
            previous_best = self.stopper.best_epoch and not improved
            if previous_best and self.best_sha256 is None:
                raise RuntimeError("Previously selected best checkpoint is unverified")
            payload["pitchlens_best_sha256"] = self.best_sha256 if previous_best else None
            serialized = staging / "state.pt"
            torch.save(payload, serialized)
            if improved and best.exists():
                self.archive_best(best, staging)
            destinations = [best] if improved else []
            if period > 0 and self.epoch % period == 0:
                destinations.append(directory / f"epoch{self.epoch}.pt")
            destinations.append(last)
            for target in destinations:
                pending = staging / "publish.pt"
                shutil.copyfile(serialized, pending)
                pending.replace(target)
            if improved:
                self.best_sha256 = digest(serialized)
        return True

    def archive_best(self, best: Path, staging: Path) -> None:
        """Keep the superseded best available to earlier periodic checkpoints."""
        if self.best_sha256 is None or digest(best) != self.best_sha256:
            raise ValueError("Existing best checkpoint does not match its recorded SHA-256")
        archive = best.with_name(f"best-{self.best_sha256}.pt")
        if archive.exists():
            if digest(archive) != self.best_sha256:
                raise ValueError("Historical best checkpoint does not match its recorded SHA-256")
            return
        pending = staging / "archive.pt"
        shutil.copyfile(best, pending)
        pending.rename(archive)

    def final_eval(self):
        # Ultralytics strips and updates the checkpoint it validates. Preserve
        # the selected checkpoint's exact bytes and metadata during that pass.
        selected = self.best
        if not selected.exists():
            return super().final_eval()
        with TemporaryDirectory(prefix="final-validation-", dir=self.wdir) as temporary:
            self.best = Path(temporary) / selected.name
            try:
                shutil.copyfile(selected, self.best)
                return super().final_eval()
            finally:
                self.best = selected


def compare_state(actual, expected, field: str) -> None:
    if isinstance(expected, torch.Tensor):
        if not torch.equal(actual.detach().cpu(), expected.to(dtype=actual.dtype, device="cpu")):
            raise RuntimeError(f"Resume state mismatch in {field}")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError(f"Resume state keys mismatch in {field}")
        for key, value in expected.items():
            compare_state(actual[key], value, f"{field}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise RuntimeError(f"Resume state length mismatch in {field}")
        for index, (item, value) in enumerate(zip(actual, expected)):
            compare_state(item, value, f"{field}.{index}")
    elif actual != expected:
        raise RuntimeError(f"Resume state mismatch in {field}")


def verify_resume_state(trainer, checkpoint: dict) -> dict:
    saved_model = checkpoint.get("ema") or checkpoint["model"]
    compare_state(unwrap_model(trainer.model).state_dict(), saved_model.state_dict(), "model")
    compare_state(trainer.ema.ema.state_dict(), saved_model.state_dict(), "EMA")
    compare_state(trainer.ema.updates, checkpoint["updates"], "EMA updates")
    actual, expected = trainer.optimizer.state_dict(), checkpoint["optimizer"]
    # Ultralytics selects the fused optimizer kernel for the current device.
    for group in actual["param_groups"]:
        group.pop("fused", None)
    expected = {**expected, "param_groups": [{key: value for key, value in group.items() if key != "fused"}
                                            for group in expected["param_groups"]]}
    compare_state(actual, expected, "optimizer")
    compare_state(trainer.scaler.state_dict(), checkpoint["scaler"], "AMP scaler")
    compare_state(trainer.scaler.is_enabled(), bool(checkpoint["scaler"]), "AMP scaler enabled")
    compare_state(trainer.best_fitness, checkpoint["best_fitness"], "best fitness")
    stopper_state = saved_stopper(checkpoint)
    compare_state(vars(trainer.stopper), stopper_state, "early stopping")

    # Ultralytics saves no scheduler object. Reconstruct LambdaLR at the
    # previous epoch so the first scheduler.step() uses the next epoch's LR.
    start = checkpoint["epoch"] + 1
    config = checkpoint["train_args"]
    compare_state(trainer.start_epoch, start, "start epoch")
    compare_state(trainer.scheduler.last_epoch, start - 1, "scheduler epoch")
    compare_state(trainer.epochs, config["epochs"], "schedule length")
    base_lrs = [group["initial_lr"] for group in expected["param_groups"]]
    compare_state(trainer.scheduler.base_lrs, base_lrs, "scheduler base LRs")
    progress = start / config["epochs"]
    if config["cos_lr"]:
        progress = (1 - math.cos(math.pi * progress)) / 2
    factor = 1 + progress * (config["lrf"] - 1)
    if not all(math.isclose(schedule(start), factor) for schedule in trainer.scheduler.lr_lambdas):
        raise RuntimeError("Resume scheduler would use the wrong learning rate")
    return {"saved_epoch": start, "next_epoch": start + 1, "scheduler_last_epoch": start - 1,
            "next_epoch_lrs": [rate * factor for rate in base_lrs], "amp_scaler": bool(checkpoint["scaler"]),
            "optimizer_states": len(actual["state"]), "ema_updates": trainer.ema.updates,
            "early_stopping": stopper_state}


def stage_training_data(data: Path, expected: str, snapshot: Path) -> Path:
    manifest = verified_bytes(data.parent / "manifest.json", expected)
    rows = json.loads(manifest)
    for split in ("train", "val"):
        for frame in manifest_frames(data.parent, rows, split):
            frame.snapshot(snapshot / split)
    (snapshot / "manifest.json").write_bytes(manifest)
    local_data = snapshot / "data.yaml"
    names = ", ".join(CLASS_NAMES)
    local_data.write_text(f"train: train/images\nval: val/images\nnames: [{names}]\n", encoding="utf-8")
    return local_data


def resume_detection(data: Path, weights: Path, run: Path, manifest_sha256: str, *, smoke: bool,
                     best_weights: Path | None = None) -> dict:
    root = data.parent
    checkpoint_dir = weights.parent.parent if weights.parent.name == "weights" else weights.parent
    if run.is_relative_to(root) or run.is_relative_to(checkpoint_dir):
        raise ValueError("Use a new run outside the dataset and the original checkpoint directory")
    if run.exists():
        raise FileExistsError(run)
    if weights.suffix != ".pt":
        raise ValueError("Resume requires a local stateful .pt checkpoint")
    run.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix="resume-input-", dir=run) as directory:
        snapshot = Path(directory)
        local_weights = snapshot / "checkpoint.pt"
        shutil.copyfile(weights, local_weights)
        weights_sha256 = digest(local_weights)
        model, checkpoint = load_checkpoint(local_weights)
        checkpoint_manifest(checkpoint, manifest_sha256)
        if (model.task != "detect" or model.names != dict(enumerate(CLASS_NAMES))
                or checkpoint.get("epoch", -1) < 0 or not checkpoint.get("optimizer")
                or "scaler" not in checkpoint or checkpoint.get("ema") is None):
            raise ValueError("Checkpoint lacks the required detection continuation state")
        config = checkpoint["train_args"]
        if not config.get("save", True):
            raise ValueError("Continuation requires checkpoint saving enabled")
        if checkpoint["epoch"] + 1 >= config["epochs"]:
            raise ValueError("Checkpoint has already reached its configured epoch limit")
        stopper = StrictEarlyStopping(config["patience"])
        vars(stopper).update(saved_stopper(checkpoint))
        if stopper.exhausted:
            raise ValueError("Checkpoint has already exhausted its early-stopping patience")
        if "pitchlens_best_sha256" not in checkpoint:
            raise ValueError("Checkpoint has no verifiable best-model reference")
        local_best = stage_saved_best(checkpoint, local_weights, weights, snapshot, best_source=best_weights)
        local_data = stage_training_data(data, manifest_sha256, snapshot)
        trainer = PitchLensTrainer(manifest_sha256=manifest_sha256,
                                   overrides={"model": str(local_weights), "resume": str(local_weights),
                                              "data": str(local_data), "save_dir": str(run), "task": "detect",
                                              "device": 0, "workers": 0, "cache": False, "plots": False})
        if local_best is not None:
            shutil.copyfile(local_best, trainer.best)
            trainer.best_sha256 = digest(trainer.best)
        report = {"checkpoint": str(weights), "checkpoint_sha256": weights_sha256,
                  "data": str(data), "manifest_sha256": manifest_sha256, "setup_only": smoke,
                  "best_checkpoint_sha256": trainer.best_sha256}

        def verify(trainer) -> None:
            report.update(verify_resume_state(trainer, checkpoint))
            (run / "resume_state.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        trainer.add_callback("on_pretrain_routine_end", verify)
        if smoke:
            trainer._setup_train()
        else:
            trainer.add_callback("on_model_save", save_resumable)
            trainer.train()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared data.yaml path")
    parser.add_argument("--weights", type=Path, required=True, help="Local pretrained yolo26s.pt")
    parser.add_argument("--run", type=Path, required=True, help="New experiment directory")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs" / "detection_train.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Run one epoch using 5% of training images")
    mode.add_argument("--resume", action="store_true", help="Continue a stateful checkpoint in an isolated run")
    mode.add_argument("--resume-smoke", action="store_true", help="Verify resumed state without running any epochs")
    parser.add_argument("--manifest-sha256", required=True, help="Previously recorded training manifest SHA-256")
    parser.add_argument("--best-weights", type=Path, help="Matching best checkpoint if the original best.pt is elsewhere")
    parser.add_argument("--save-resumable", action="store_true",
                        help="Keep the latest checkpoint with optimizer and scaler state")
    args = parser.parse_args()
    data, weights, run = args.data.resolve(), args.weights.resolve(), args.run.resolve()
    for path in (data, weights, args.config):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; this training configuration requires the approved NVIDIA GPU")
    if args.resume or args.resume_smoke:
        best_weights = args.best_weights.resolve() if args.best_weights else None
        print(json.dumps(resume_detection(data, weights, run, args.manifest_sha256, smoke=args.resume_smoke,
                                          best_weights=best_weights), indent=2))
        return
    if args.best_weights:
        parser.error("--best-weights requires --resume or --resume-smoke")
    parameters = json.loads(args.config.read_text(encoding="utf-8"))
    if {"resume", "save_dir", "project", "name", "exist_ok"} & parameters.keys():
        parser.error("Run destinations and resume mode must be set through the launcher arguments")
    if str(parameters.get("device")) != "0":
        parser.error("Detection training requires the approved single GPU, device 0")
    if not parameters.get("save", True):
        parser.error("Checkpoint saving must stay enabled to preserve every selected best")
    if run.is_relative_to(data.parent):
        parser.error("Use a new run outside the prepared dataset")
    if args.smoke:
        parameters.update(epochs=1, fraction=0.05, save_period=-1)
    run.mkdir(parents=True, exist_ok=False)
    info = {"weights": str(weights), "weights_sha256": digest(weights), "data": str(data),
            "data_sha256": digest(data), "parameters": parameters, "smoke": args.smoke,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0)}
    info["manifest.json_sha256"] = args.manifest_sha256
    if (data.parent / "report.json").is_file():
        info["report.json_sha256"] = digest(data.parent / "report.json")
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
    if args.save_resumable:
        model.add_callback("on_model_save", save_resumable)
    with TemporaryDirectory(prefix="training-input-", dir=run) as directory:
        local_data = stage_training_data(data, args.manifest_sha256, Path(directory))
        model.train(trainer=partial(PitchLensTrainer, manifest_sha256=args.manifest_sha256),
                    data=str(local_data), save_dir=str(run), exist_ok=True, **parameters)


if __name__ == "__main__":
    main()
