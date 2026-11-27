"""Ball checkpoint and tracking-runner contracts without pretrained weights."""

import argparse
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

from pitchlens.ball_detector import (
    BallDetector, BallDetectorConfig, add_detection_arguments, combine_detections, video_detection_settings,
)
from pitchlens.detector import CLASS_NAMES, Detection, Detector
from pitchlens.tracking import Tracker
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
    def test_profiles_and_independent_overrides(self):
        parser = argparse.ArgumentParser()
        add_detection_arguments(parser)
        defaults = video_detection_settings(parser.parse_args([]))
        self.assertEqual(defaults.profile, 'final')
        self.assertTrue(defaults.people_weights.as_posix().endswith('/detection-candidate-v2-640/weights/best.pt'))
        self.assertEqual((defaults.people_image_size, defaults.people_confidence), (640, .25))
        self.assertEqual(defaults.ball_weights.name, 'yolo-sn-ball-opt.pt')
        self.assertEqual((defaults.ball_config.image_size, defaults.ball_config.confidence), (960, .15))
        legacy = video_detection_settings(parser.parse_args(['--weights', 'original.pt']))
        self.assertEqual((legacy.profile, legacy.people_image_size, legacy.ball_weights), ('legacy', 960, None))
        self.assertEqual(legacy.people_weights, Path('original.pt').resolve())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = root / 'ball.json'
            config.write_text('{"image_size": 1280, "confidence": 0.20}', encoding='utf-8')
            custom = video_detection_settings(parser.parse_args([
                '--detector-profile', 'gsr', '--storage-root', str(root), '--people-image-size', '704',
                '--people-confidence', '.3', '--ball-config', str(config),
            ]))
            self.assertEqual(custom.people_weights, root / 'runs/gsr-yolo26s-640-v1/weights/best.pt')
            self.assertEqual(custom.ball_weights, root / 'models/pretrained/yolo-sn-ball-opt.pt')
            self.assertEqual((custom.people_image_size, custom.people_confidence), (704, .3))
            self.assertEqual((custom.ball_config.image_size, custom.ball_config.confidence), (1280, .2))
            config.write_text('[]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Invalid dedicated ball configuration'):
                video_detection_settings(parser.parse_args(['--ball-config', str(config)]))
        with self.assertRaisesRegex(ValueError, 'Invalid people detector settings'):
            video_detection_settings(parser.parse_args(['--people-image-size', '641']))

    def test_dual_inference_feeds_bytetrack_recovery_and_only_dedicated_ball(self):
        class Boxes:
            def __init__(self, classes, scores):
                self.cls = torch.tensor(classes)
                self.conf = torch.tensor(scores)
                self.xyxy = torch.tensor([[10. + 20*i, 10., 20. + 20*i, 30.] for i in range(len(classes))])

            def __len__(self):
                return len(self.cls)

        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        people = Mock(task='detect', names=dict(enumerate(CLASS_NAMES)))
        people.predict.side_effect = [
            [SimpleNamespace(boxes=Boxes([0, 1, 2, 3], [.9, .8, .7, .99]))],
            [SimpleNamespace(boxes=Boxes([0], [.19]))],
            [],
        ]
        ball = Mock(task='detect', names={0: 'ball'})
        ball.predict.side_effect = [[SimpleNamespace(boxes=Boxes([0], [.8]))],
                                    [SimpleNamespace(boxes=Boxes([0], [.2]))], []]
        with tempfile.TemporaryDirectory() as tmp, patch('pitchlens.detector.YOLO', side_effect=[people, ball]) as loader:
            path = Path(tmp) / 'local.pt'
            path.touch()
            person_detector = Detector(path, image_size=640, confidence=.25, candidate_confidence=.1,
                                       people_only=True, device='cpu')
            ball_detector = BallDetector(path, BallDetectorConfig(), device='cpu')
            tracker = Tracker(30.)
            first, second, empty = [combine_detections(person_detector.detect(frame), ball_detector.detect(frame))
                                    for _ in range(3)]
            self.assertEqual(loader.call_count, 2)
        self.assertEqual([item.class_id for item in first], [0, 1, 2, 3])
        self.assertEqual(first[-1].xyxy, (10., 10., 20., 30.))
        self.assertEqual([item.class_id for item in second], [0, 3])
        self.assertEqual(empty, [])
        first_tracks = tracker.update(frame, first).tracks
        second_tracks = tracker.update(frame, second).tracks
        self.assertEqual([item.class_id for item in first_tracks], [0, 1, 2, 3])
        self.assertEqual([item.class_id for item in second_tracks], [0, 3])
        self.assertEqual(first_tracks[0].track_id, second_tracks[0].track_id)
        self.assertLess(second_tracks[0].confidence, person_detector.confidence)
        self.assertFalse(second_tracks[-1].observed)
        people_args, ball_args = people.predict.call_args.kwargs, ball.predict.call_args.kwargs
        self.assertEqual((people_args['imgsz'], people_args['conf'], people_args['classes']), (640, .1, [0, 1, 2]))
        self.assertEqual((ball_args['imgsz'], ball_args['conf']), (960, .15))

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
                        '--weights', 'general.pt', '--device', 'cpu', '--team-mode', 'color']
                if enable_ball:
                    args += ['--ball-weights', 'ball.pt']
                else:
                    args += ['--people-tracker', 'bytetrack']
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
                self.assertEqual(summary['detection']['profile'], 'legacy')
                self.assertEqual(summary['team_classifier']['backend'], 'color')
                self.assertEqual(general.call_args.kwargs['candidate_confidence'], .1)
                self.assertEqual(summary['people_tracking'], {
                    'backend': 'BOTSORT' if enable_ball else 'BYTETracker',
                    'gmc_method': 'sparseOptFlow' if enable_ball else None,
                    'with_reid': False,
                })
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
