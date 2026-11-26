import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pitchlens.pitch_landmarks import PitchLandmarkDetector


class PitchLandmarkLoadingTests(unittest.TestCase):
    def test_security_flag_and_digest_fail_before_loading(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'external.pt'
            path.write_bytes(b'unapproved')
            with patch('pitchlens.pitch_landmarks.YOLO') as loader:
                with patch('pitchlens.pitch_landmarks.SAFE_LOAD', False), self.assertRaises(RuntimeError):
                    PitchLandmarkDetector(path, 'cpu')
                with patch('pitchlens.pitch_landmarks.SAFE_LOAD', True), self.assertRaisesRegex(ValueError, 'SHA-256'):
                    PitchLandmarkDetector(path, 'cpu')
                loader.assert_not_called()

    def test_loader_reads_the_verified_snapshot(self):
        with TemporaryDirectory() as folder:
            original = Path(folder) / 'external.pt'
            payload = b'verified checkpoint payload'
            original.write_bytes(payload)

            def load(snapshot, *, task):
                original.write_bytes(b'replacement')
                self.assertEqual(Path(snapshot).read_bytes(), payload)
                self.assertEqual(task, 'pose')
                return SimpleNamespace(task='pose', model=SimpleNamespace(model=[SimpleNamespace(kpt_shape=(32, 3))]))

            with (patch('pitchlens.pitch_landmarks.SAFE_LOAD', True),
                  patch('pitchlens.pitch_landmarks.CHECKPOINT_SHA256', hashlib.sha256(payload).hexdigest()),
                  patch('pitchlens.pitch_landmarks.YOLO', side_effect=load) as loader):
                PitchLandmarkDetector(original, 'cpu')
            self.assertFalse(Path(loader.call_args.args[0]).exists())
