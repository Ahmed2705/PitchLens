"""Ball checkpoint and tracking-runner contracts without pretrained weights."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from pitchlens.ball_detector import BallDetector, BallDetectorConfig, combine_detections
from pitchlens.detector import Detection
from scripts import track_video


class BallDetectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'ball.pt'
        self.path.touch()
        self.model = Mock(task='detect', names={0: 'ball'})
        patcher = patch('pitchlens.detector.YOLO', return_value=self.model)
        self.loader = patcher.start()
        self.addCleanup(patcher.stop)

    def test_checkpoint_class_remap_and_original_frame_coordinates(self):
        class Boxes:
            xyxy = torch.tensor([[-5., 12., 130., 55.], [1., 1., 5., 5.]])
            cls = torch.tensor([0., 0.])
            conf = torch.tensor([.75, .125])

            def __len__(self):
                return 2

        self.model.predict.return_value = [SimpleNamespace(boxes=Boxes())]
        model = BallDetector(self.path, BallDetectorConfig(image_size=1280), device='cpu')
        result = model.detect(np.zeros((60, 100, 3), dtype=np.uint8))
        self.assertEqual(result, [Detection((0., 12., 100., 55.), 3, .75)])
        self.assertEqual(self.model.predict.call_args.kwargs['imgsz'], 1280)
        self.assertEqual(self.model.predict.call_args.kwargs['conf'], .15)
        self.model.predict.return_value = [SimpleNamespace(boxes=None)]
        self.assertEqual(model.detect(np.zeros((60, 100, 3), dtype=np.uint8)), [])

    def test_invalid_inference_configuration(self):
        for overrides in ({'confidence': -1}, {'confidence': True}, {'confidence': float('nan')},
                          {'image_size': 961}, {'image_size': 960.0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                BallDetectorConfig(**overrides)

    def test_missing_and_incompatible_checkpoints_fail_clearly(self):
        config = BallDetectorConfig()
        with self.assertRaisesRegex(FileNotFoundError, 'Dedicated ball checkpoint not found'):
            BallDetector(self.path.with_name('missing.pt'), config, device='cpu')
        self.loader.assert_not_called()
        self.model.task = 'segment'
        with self.assertRaisesRegex(RuntimeError, 'one-class detection model'):
            BallDetector(self.path, config, device='cpu')
        self.model.task, self.model.names = 'detect', {0: 'player'}
        with self.assertRaisesRegex(RuntimeError, 'one-class detection model'):
            BallDetector(self.path, config, device='cpu')
        self.loader.side_effect = OSError('invalid checkpoint archive')
        with self.assertRaisesRegex(RuntimeError, 'invalid checkpoint archive'):
            BallDetector(self.path, config, device='cpu')

    def test_combining_preserves_people_and_drops_general_ball_even_if_dedicated_is_empty(self):
        general = [Detection((1., 2., 3., 4.), cls, .75) for cls in range(4)]
        dedicated = Detection((10., 10., 14., 14.), 3, .875)
        self.assertIs(combine_detections(general, None), general)
        self.assertEqual(combine_detections(general, []), general[:3])
        self.assertEqual(combine_detections(general, [dedicated]), general[:3] + [dedicated])


class BallTrackingRunnerTests(unittest.TestCase):
    def test_runner_provenance_and_downstream_analytics(self):
        people = [Detection((x - 5., 30., x + 5., 70.), 0, .9) for x in (50., 150.)]
        general_ball = Detection((298., 68., 302., 72.), 3, .99)
        observed_samples = (50., 50., 50., 50., 50., 76., 98., 121., 150., 150.)
        samples = observed_samples + (None,) * 6
        dedicated = [[Detection((x - 2., 68., x + 2., 72.), 3, .875)] if x else [] for x in samples]
        frame = np.zeros((200, 480, 3), dtype=np.uint8)
        properties = {cv2.CAP_PROP_FPS: 20., cv2.CAP_PROP_FRAME_WIDTH: 480, cv2.CAP_PROP_FRAME_HEIGHT: 200,
                      cv2.CAP_PROP_POS_MSEC: 0.}
        for enable_ball in (False, True):
            with self.subTest(enable_ball=enable_ball), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / 'sample.mp4'
                args = ['track_video.py', '--input', 'input.mp4', '--output', str(output),
                        '--weights', 'general.pt', '--device', 'cpu']
                if enable_ball:
                    args += ['--ball-weights', 'ball.pt']
                with (patch('sys.argv', args), patch.object(track_video.cv2, 'VideoCapture') as capture,
                      patch.object(track_video.cv2, 'VideoWriter'), patch.object(track_video, 'Detector') as general,
                      patch.object(track_video, 'BallDetector') as ball,
                      patch.object(track_video.TeamClassifier, 'update', side_effect=lambda _, tracked:
                                   [1 if item.class_id == 0 else None for item in tracked.tracks]),
                      redirect_stdout(StringIO())):
                    capture.return_value.get.side_effect = properties.__getitem__
                    capture.return_value.read.side_effect = [(True, frame.copy()) for _ in samples] + [(False, None)]
                    general.return_value.device = 'cpu'
                    general.return_value.image_size = 960
                    general.return_value.detect.return_value = people + [general_ball]
                    ball.return_value.device = 'cpu'
                    ball.return_value.detect.side_effect = dedicated
                    track_video.main()
                records = [json.loads(line) for line in output.with_suffix('.jsonl').read_text().splitlines()]
                summary = json.loads(output.with_suffix('.json').read_text())
                balls = [next((track for track in record['tracks'] if track['class_id'] == 3), None)
                         for record in records]
                if enable_ball:
                    ball.assert_called_once()
                    self.assertEqual(records[0]['raw_ball_detections'][0]['class_id'], 3)
                    self.assertEqual(balls[0]['xyxy'], [48., 68., 52., 72.])
                    self.assertTrue(balls[0]['observed'])
                    gap_start = len(observed_samples)
                    self.assertEqual(records[gap_start]['raw_ball_detections'], [])
                    self.assertFalse(balls[gap_start]['observed'])
                    self.assertIsNone(balls[gap_start]['confidence'])
                    self.assertIsNone(balls[-1])
                    self.assertEqual(summary['passes']['likely_successful_passes'], 1)
                    self.assertGreater(summary['possession']['team_possession_coverage_pct'], 0)
                else:
                    ball.assert_not_called()
                    self.assertNotIn('raw_ball_detections', records[0])
                    self.assertNotIn('dedicated_ball', summary)
                    self.assertEqual(balls[0]['xyxy'], list(general_ball.xyxy))
                    self.assertEqual(summary['passes']['likely_successful_passes'], 0)
