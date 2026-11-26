"""Exercise native checkpoint restoration and isolated run selection without training."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from ultralytics.cfg import DEFAULT_CFG, get_cfg, get_save_dir
from ultralytics.utils.torch_utils import ModelEMA

from pitchlens.detection_manifest import sha256
from pitchlens.detector import CLASS_NAMES
from scripts.train_detection import (
    PitchLensTrainer, StrictEarlyStopping, checkpoint_manifest, digest, resume_detection,
    saved_stopper, stage_saved_best, stage_training_data, verify_resume_state,
)


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(2, 2))


def make_dataset(root, *, category=0, duplicate=False):
    rows = []
    for index, split in enumerate(("train", "train", "val", "val")):
        _, image = cv2.imencode(".jpg", np.full((64, 64, 3), index, dtype=np.uint8))
        image = image.tobytes()
        label = f"{category} 0.5 0.5 0.2 0.2\n".encode()
        if duplicate and split == "train":
            label *= 2
        image_hash = sha256(image)
        image_path = f"{split}/images/{image_hash}.jpg"
        for path, payload in ((image_path, image), (f"{split}/labels/{image_hash}.txt", label)):
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        rows.append({"split": split, "sha256": image_hash, "label_sha256": sha256(label),
                     "output": image_path, "group": split})
    manifest = root / "manifest.json"
    manifest.write_bytes(json.dumps(rows).encode())
    data = root / "data.yaml"
    data.write_text("train: train/images\nval: val/images\nnames: [player, goalkeeper, referee, ball]\n",
                    encoding="utf-8")
    return data, digest(manifest)


class ResumeTests(unittest.TestCase):
    def test_legacy_saved_destination_is_overridden_before_artifact_setup(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            old = root / "original"
            new = root / "new"
            old.mkdir()
            (old / "args.yaml").write_bytes(b"original args")
            snapshot = old / "last-resumable.pt"
            snapshot.write_bytes(b"stub")
            trainer = object.__new__(PitchLensTrainer)
            trainer.args = get_cfg(DEFAULT_CFG, {"resume": str(snapshot), "data": "isolated.yaml"})
            original_args = {**vars(DEFAULT_CFG), "save_dir": str(old), "data": "previous.yaml"}
            with patch("ultralytics.engine.trainer.load_checkpoint",
                       return_value=(SimpleNamespace(args=original_args), {})):
                trainer.check_resume({"save_dir": str(new), "resume": str(snapshot)})
            self.assertTrue(trainer.resume)
            self.assertEqual(get_save_dir(trainer.args), new)
            self.assertEqual(trainer.args.data, "isolated.yaml")
            self.assertEqual((old / "args.yaml").read_bytes(), b"original args")

    def test_existing_run_and_original_checkpoint_are_preserved(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            original = root / "original/weights"
            original.mkdir(parents=True)
            weights = original / "last-resumable.pt"
            weights.write_bytes(b"original checkpoint")
            existing = root / "existing"
            existing.mkdir()
            report = existing / "resume_state.json"
            report.write_bytes(b"original report")
            with self.assertRaises(FileExistsError):
                resume_detection(root / "dataset/data.yaml", weights, existing, "", smoke=True)
            with self.assertRaisesRegex(ValueError, "outside"):
                resume_detection(root / "dataset/data.yaml", weights, original.parent / "nested", "", smoke=True)
            self.assertEqual(report.read_bytes(), b"original report")
            self.assertEqual(weights.read_bytes(), b"original checkpoint")

    def test_effective_save_disabled_configuration_is_rejected_before_setup(self):
        with TemporaryDirectory() as folder:
            new = Path(folder).resolve() / "new"
            with self.assertRaisesRegex(ValueError, "Checkpoint saving must stay enabled"):
                PitchLensTrainer(manifest_sha256=sha256(b"fixture"),
                                 overrides={"task": "detect", "save_dir": str(new), "save": False})
            self.assertFalse(new.exists())

    def test_model_optimizer_scheduler_scaler_and_stopper_restoration(self):
        saved = FixtureModel()
        optimizer = torch.optim.AdamW(saved.parameters(), lr=.01)
        for parameter in saved.parameters():
            optimizer.state[parameter].update(step=torch.tensor(12.), exp_avg=torch.full_like(parameter, .25),
                                              exp_avg_sq=torch.full_like(parameter, .5))
        optimizer.param_groups[0]["initial_lr"] = .01
        optimizer.param_groups[0]["lr"] = .007
        config = {"epochs": 10, "cos_lr": False, "lrf": .1, "patience": 4}
        scaler = torch.amp.GradScaler("cpu", init_scale=4096.)
        stopper = StrictEarlyStopping(4)
        for epoch, fitness in enumerate((.2, .4, .4, .3), 1):
            stopper(epoch, fitness)
        training_hash = sha256(b"fixture")
        checkpoint = {"ema": saved, "model": None, "updates": 12, "epoch": 3, "best_fitness": .4,
                      "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "train_args": config,
                      "stopper": vars(stopper), "pitchlens_manifest_sha256": training_hash}
        trainer = object.__new__(PitchLensTrainer)
        trainer.manifest_sha256 = training_hash
        trainer.resume, trainer.epochs = True, config["epochs"]
        trainer.args = SimpleNamespace(model="fixture.pt", **config)
        trainer.model = deepcopy(saved)  # Native setup_model initializes weights from saved EMA.
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.01)
        trainer.scaler = torch.amp.GradScaler("cpu")
        trainer.ema = ModelEMA(trainer.model)
        trainer._setup_scheduler()
        trainer.resume_training(deepcopy(checkpoint))
        trainer.scheduler.last_epoch = trainer.start_epoch - 1  # Native setup_train restores the index.
        verified = verify_resume_state(trainer, checkpoint)
        self.assertEqual(verified["next_epoch"], 5)
        self.assertEqual(verified["optimizer_states"], 2)
        self.assertTrue(verified["amp_scaler"])
        self.assertAlmostEqual(verified["next_epoch_lrs"][0], .0064)
        self.assertEqual(verified["early_stopping"], vars(stopper))
        self.assertFalse(trainer.stopper(5, .4))
        self.assertTrue(trainer.stopper(6, .4))
        self.assertEqual((trainer.stopper.best_epoch, trainer.stopper.best_fitness), (2, .4))
        vars(trainer.stopper).update(vars(stopper))
        for field, break_state in (
            ("scheduler", lambda: setattr(trainer.scheduler, "last_epoch", 0)),
            ("AMP scaler", lambda: trainer.scaler.load_state_dict({**checkpoint["scaler"], "scale": 2048.})),
        ):
            with self.subTest(field=field):
                break_state()
                with self.assertRaisesRegex(RuntimeError, field):
                    verify_resume_state(trainer, checkpoint)
                trainer.scheduler.last_epoch = trainer.start_epoch - 1

    def test_manifest_binding_and_incomplete_checkpoints_fail_closed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            data, training_hash = make_dataset(root / "dataset")
            checkpoint = {"pitchlens_manifest_sha256": training_hash}
            checkpoint_manifest(checkpoint, training_hash)
            snapshot = root / "snapshot"
            self.assertTrue(stage_training_data(data, training_hash, snapshot).is_file())
            manifest = data.parent / "manifest.json"
            manifest.write_bytes(manifest.read_bytes() + b"\n")
            alternative = digest(manifest)
            self.assertNotEqual(alternative, training_hash)
            with self.assertRaisesRegex(ValueError, "differs from the checkpoint"):
                checkpoint_manifest(checkpoint, alternative)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                stage_training_data(data, training_hash, snapshot)
            for value in ({}, {"pitchlens_manifest_sha256": "not a SHA"}):
                with self.assertRaisesRegex(ValueError, "no verifiable"):
                    checkpoint_manifest(value, training_hash)
            for state in ({}, {"stopper": {"best_fitness": .4, "best_epoch": 1}}):
                with self.assertRaisesRegex(ValueError, "lacks complete"):
                    saved_stopper(state)

    def test_checkpoint_saves_only_strict_improvements_and_preserves_metadata(self):
        with TemporaryDirectory() as folder:
            trainer = object.__new__(PitchLensTrainer)
            trainer.wdir = Path(folder)
            trainer.last, trainer.best = trainer.wdir / "last.pt", trainer.wdir / "best.pt"
            trainer.save_period = 1
            trainer.args = SimpleNamespace(patience=5)
            trainer.manifest_sha256 = sha256(b"fixture")
            trainer.stopper = StrictEarlyStopping(5)
            trainer.best_sha256 = None

            def native_save(instance):
                # Mirror the upstream tie write in a disposable staging directory.
                payload = {"train_args": vars(instance.args), "epoch": instance.epoch,
                           "best_fitness": instance.best_fitness, "fixture_epoch": instance.epoch}
                torch.save(payload, instance.last)
                if instance.best_fitness == instance.fitness:
                    torch.save(payload, instance.best)
                return True

            with patch("ultralytics.engine.trainer.BaseTrainer.save_model", new=native_save):
                previous = None
                for epoch, score in enumerate((0., 0., -.01, 0., .4, .4, .4)):
                    trainer.epoch, trainer.fitness = epoch, score
                    trainer.stopper(epoch + 1, score)
                    trainer.save_model()
                    selected = trainer.best.read_bytes()
                    if epoch in (0, 4):
                        self.assertNotEqual(selected, previous)
                    else:
                        self.assertEqual(selected, previous)
                    previous = selected
                    best = torch.load(trainer.best, weights_only=True)
                    last = torch.load(trainer.last, weights_only=True)
                    self.assertEqual(best["fixture_epoch"], 0 if epoch < 4 else 4)
                    self.assertEqual(last["stopper"]["best_epoch"], best["stopper"]["best_epoch"])
                    self.assertEqual(last["best_fitness"], best["best_fitness"])
                    self.assertEqual(last["pitchlens_manifest_sha256"], trainer.manifest_sha256)
                    self.assertEqual(last["pitchlens_best_sha256"], None if epoch in (0, 4) else sha256(selected))
                    self.assertEqual((trainer.wdir / f"epoch{epoch}.pt").read_bytes(), trainer.last.read_bytes())

            def native_final_eval(instance):
                instance.best.write_bytes(b"stripped for validation")

            with patch("ultralytics.engine.trainer.BaseTrainer.final_eval", new=native_final_eval):
                trainer.final_eval()
            self.assertEqual(trainer.best.read_bytes(), previous)

    def test_training_snapshot_rejects_loader_annotation_repair(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            data, training_hash = make_dataset(root / "dataset", duplicate=True)
            snapshot = root / "snapshot"
            staged = stage_training_data(data, training_hash, snapshot)
            images = sorted((snapshot / "train/images").iterdir())
            labels = [{"im_file": str(image), "cls": np.array([[0.]], dtype=np.float32),
                       "bboxes": np.array([[.5, .5, .2, .2]], dtype=np.float32)}
                      for image in images]
            loader = SimpleNamespace(dataset=SimpleNamespace(fraction=1., im_files=images, labels=labels))
            trainer = object.__new__(PitchLensTrainer)
            trainer.manifest_sha256 = training_hash
            staged_trainer = SimpleNamespace(args=SimpleNamespace(data=staged), train_loader=loader, test_loader=None)
            with self.assertRaisesRegex(ValueError, "Training loader changed annotations"):
                trainer.verify_training_snapshot(staged_trainer)

    def test_complete_prevalidation_state_requires_no_historical_best(self):
        stopper = StrictEarlyStopping(1)
        self.assertFalse(stopper(1, None))
        self.assertFalse(stopper.possible_stop)
        checkpoint = {"epoch": 0, "train_args": {"patience": 1, "val": False}, "best_fitness": None,
                      "stopper": vars(stopper), "pitchlens_best_sha256": None}
        self.assertEqual(saved_stopper(checkpoint), vars(stopper))
        self.assertIsNone(stage_saved_best(checkpoint, Path("checkpoint.pt"), Path("best.pt"), Path(".")))
        self.assertFalse(stopper(2, 0.))
        self.assertEqual(stopper.best_epoch, 2)
        self.assertTrue(stopper(3, 0.))

    def test_incomplete_or_exhausted_stopper_prevents_trainer_start(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            model_directory = root / "old/weights"
            model_directory.mkdir(parents=True)
            weights = model_directory / "last.pt"
            weights.write_bytes(b"checkpoint fixture")
            training_hash = sha256(b"fixture")
            stopper = StrictEarlyStopping(1)
            stopper(1, .4)
            self.assertTrue(stopper(2, .4))
            checkpoint = {"pitchlens_manifest_sha256": training_hash, "epoch": 1, "optimizer": {"fixture": True},
                          "scaler": {}, "ema": object(), "best_fitness": .4, "train_args": {"epochs": 10, "patience": 1}}
            model = SimpleNamespace(task="detect", names=dict(enumerate(CLASS_NAMES)))
            for state, save, message in (({}, True, "lacks complete"), (vars(stopper), True, "exhausted"),
                                         (vars(stopper), False, "checkpoint saving enabled")):
                with (self.subTest(state=state), patch("scripts.train_detection.PitchLensTrainer") as trainer,
                      patch("scripts.train_detection.load_checkpoint",
                            return_value=(model, {**checkpoint, "stopper": state,
                                                  "train_args": {**checkpoint["train_args"], "save": save}})),
                      self.assertRaisesRegex(ValueError, message)):
                    run = root / ("disabled" if not save else "exhausted" if state else "incomplete")
                    resume_detection(root / "dataset/data.yaml", weights, run, training_hash, smoke=True)
                trainer.assert_not_called()
                self.assertFalse((run / "resume_state.json").exists())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA resume smoke requires an available GPU")
    def test_cuda_resume_smoke_with_disposable_synthetic_checkpoint(self):
        with TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            data, training_hash = make_dataset(root / "dataset")
            initial = root / "fixture"
            initial.mkdir()
            local_data = stage_training_data(data, training_hash, initial)
            trainer = PitchLensTrainer(manifest_sha256=training_hash,
                                       overrides={"model": "yolo26s.yaml", "task": "detect", "data": str(local_data),
                                                  "save_dir": str(initial), "device": 0, "workers": 0,
                                                  "batch": 2, "imgsz": 64, "amp": True, "epochs": 10,
                                                  "patience": 2, "optimizer": "AdamW", "plots": False,
                                                  "save_period": 2})
            trainer._setup_train()
            parameter = trainer.optimizer.param_groups[0]["params"][0]
            trainer.optimizer.state[parameter].update(
                step=torch.tensor(12., device=parameter.device), exp_avg=torch.full_like(parameter, .25),
                exp_avg_sq=torch.full_like(parameter, .5))
            trainer.scaler.load_state_dict({**trainer.scaler.state_dict(), "scale": 4096., "_growth_tracker": 12})
            trainer.ema.updates = 12
            trainer.metrics = {}
            for epoch, score in enumerate((.2, .4, .4, .5)):
                trainer.epoch, trainer.fitness = epoch, score
                self.assertFalse(trainer.stopper(epoch + 1, score))
                self.assertTrue(trainer.save_model())
                if epoch == 1:
                    selected_hash = digest(trainer.best)
                    self.assertFalse((trainer.wdir / "epoch1.pt").exists())
                if epoch == 2:
                    self.assertEqual(digest(trainer.best), selected_hash)
            weights = trainer.wdir / "epoch2.pt"
            archive = weights.with_name(f"best-{selected_hash}.pt")
            newer_best_hash = digest(trainer.best)
            self.assertNotEqual(newer_best_hash, selected_hash)
            self.assertEqual(digest(archive), selected_hash)
            original_hash = digest(weights)
            del trainer
            other_data, other_hash = make_dataset(root / "other-dataset", category=1)
            moved = root / "moved/weights"
            moved.mkdir(parents=True)
            copied = moved / weights.name
            shutil.copyfile(weights, copied)
            # A different sibling checkpoint must never stand in for the selected best.
            shutil.copyfile(weights.with_name("best.pt"), moved / "best.pt")

            with patch.object(PitchLensTrainer, "train", side_effect=AssertionError("Training is forbidden")):
                with self.assertRaisesRegex(ValueError, "differs from the checkpoint"):
                    resume_detection(other_data, weights, root / "wrong-manifest", other_hash, smoke=True)
                with self.assertRaisesRegex(ValueError, "Historical best checkpoint SHA-256 mismatch"):
                    resume_detection(data, copied, root / "wrong-best", training_hash, smoke=True)
                report = resume_detection(data, weights, root / "resume", training_hash, smoke=True)
            self.assertEqual(digest(weights), original_hash)
            self.assertEqual(digest(weights.with_name("best.pt")), newer_best_hash)
            self.assertEqual(digest(archive), selected_hash)
            self.assertEqual(digest(root / "resume/weights/best.pt"), selected_hash)
            self.assertEqual(report["best_checkpoint_sha256"], selected_hash)
            self.assertEqual(report["next_epoch"], 4)
            self.assertEqual(report["optimizer_states"], 1)
            self.assertTrue(report["amp_scaler"])
            self.assertEqual(report["early_stopping"], {"best_fitness": .4, "best_epoch": 2, "patience": 2,
                                                        "epochs_without_improvement": 1, "possible_stop": True})
            self.assertFalse((root / "resume/results.csv").exists())
            print("CUDA setup-only resume evidence:", json.dumps(report))


if __name__ == "__main__":
    unittest.main()
