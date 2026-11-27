"""Automatic team fitting and runner contracts without model weights."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from pitchlens.automatic_team import (
    AutomaticTeamClassifier, AutomaticTeamConfig, fit_crops, player_crop, prepare_automatic_teams,
)
from pitchlens.detector import Detection
from pitchlens.tracking import FrameTracks, Track
from scripts import track_video


class AutomaticTeamTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((120, 240, 3), dtype=np.uint8)
        self.player = Track((20., 15., 50., 86.), 0, 1, .9, True)

    def test_crop_filters_small_or_invalid_boxes_and_nonplayers(self):
        self.frame[:] = (20, 90, 170)
        crop = player_crop(self.frame, self.player.xyxy)
        self.assertEqual(crop.shape, (34, 22, 3))
        for box in ((0, 0, 15, 15), (-1, 12, 40, 65), (1, 50, 45, 40), (1, 2, 82, 42)):
            self.assertIsNone(player_crop(self.frame, box))
        people = [Detection(self.player.xyxy, 0, .8), Detection((70, 15, 100, 86), 1, .9),
                  Detection((110, 15, 140, 86), 2, .9), Detection((150, 15, 180, 86), 0, .3)]
        selected = fit_crops(self.frame, people, AutomaticTeamConfig())
        self.assertEqual(len(selected), 1)
        self.assertFalse(np.shares_memory(selected[0], self.frame))
        self.assertEqual(fit_crops(self.frame, people + [Detection(self.player.xyxy, 1, .9)], AutomaticTeamConfig()), [])

    def test_fit_uses_umap_then_two_clusters_in_siglip_space(self):
        vectors = np.array([[1., .1]] * 8 + [[.1, 1.]] * 8, np.float32)
        projected = np.column_stack([vectors, np.zeros(16)])
        reducer = Mock()
        reducer.return_value.fit_transform.return_value = projected
        kmeans = Mock()
        kmeans.return_value.fit_predict.return_value = np.array([0] * 8 + [1] * 8)
        classifier = AutomaticTeamClassifier(Mock(return_value=vectors))
        with patch.dict('sys.modules', {'umap': SimpleNamespace(UMAP=reducer),
                                        'sklearn.cluster': SimpleNamespace(KMeans=kmeans)}):
            classifier.fit([self.frame] * 16)
        self.assertEqual(reducer.call_args.kwargs['n_components'], 3)
        self.assertEqual(kmeans.call_args.kwargs['n_clusters'], 2)
        self.assertIs(kmeans.return_value.fit_predict.call_args.args[0], projected)
        self.assertEqual(classifier.cluster_sizes, (8, 8))
        self.assertEqual(classifier.prototypes.argmax(axis=1).tolist(), [0, 1])

    def test_ambiguous_tracks_retry_then_cache_and_cut_clears_only_track_evidence(self):
        encoder = Mock(side_effect=[[[1, 1]], [[1, 0]], [[1, 0]], [[0, 1]], [[0, 1]]])
        classifier = AutomaticTeamClassifier(encoder, AutomaticTeamConfig(retry_interval=2))
        classifier.prototypes = np.eye(2, dtype=np.float32)
        tracks = (self.player, replace(self.player, class_id=1), replace(self.player, class_id=2),
                  replace(self.player, track_id=2, confidence=.15))

        def update(index, segment=0, cut=False):
            return classifier.update(self.frame, FrameTracks(index, segment, cut, tracks, ()))

        self.assertEqual(update(0), [None] * 4)
        self.assertEqual(update(1), [None] * 4)
        self.assertEqual(update(2), [None] * 4)
        self.assertEqual(update(4), [1, None, None, None])
        self.assertEqual(update(5), [1, None, None, None])
        self.assertEqual(update(6, 1, True), [None] * 4)
        self.assertEqual(update(8, 1), [2, None, None, None])
        self.assertEqual(encoder.call_count, 5)
        np.testing.assert_array_equal(classifier.prototypes, np.eye(2))

    def test_insufficient_samples_and_missing_checkpoint_fail(self):
        encoder = Mock()
        with self.assertRaisesRegex(ValueError, 'at least 16 reliable player crops'):
            AutomaticTeamClassifier(encoder).fit([self.frame])
        encoder.assert_not_called()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, 'Missing SigLIP snapshot'):
                prepare_automatic_teams(Path('input.mp4'), Mock(), Path(temp))

    def test_default_runner_preserves_numeric_team_and_track_schema(self):
        classifier = AutomaticTeamClassifier(Mock(return_value=[[1, 0], [0, 1]]),
                                             AutomaticTeamConfig(min_observations=1))
        classifier.prototypes = np.eye(2, dtype=np.float32)
        properties = {cv2.CAP_PROP_FPS: 30., cv2.CAP_PROP_FRAME_WIDTH: 240,
                      cv2.CAP_PROP_FRAME_HEIGHT: 120, cv2.CAP_PROP_POS_MSEC: 0.}
        boxes = [Detection(self.player.xyxy, 0, .9), Detection((120., 15., 150., 86.), 0, .9),
                 Detection((180., 20., 185., 25.), 3, .9)]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'test.mp4'
            argv = ['track_video.py', '--input', 'fixture.mp4', '--output', str(output), '--weights', 'fixture.pt']
            with (patch('sys.argv', argv), patch.object(track_video, 'Detector') as detector,
                  patch.object(track_video.cv2, 'VideoCapture') as capture, patch.object(track_video.cv2, 'VideoWriter'),
                  patch('pitchlens.automatic_team.prepare_automatic_teams', return_value=classifier) as fit,
                  patch('builtins.print')):
                detector.return_value.device = 'cpu'
                detector.return_value.image_size = 960
                detector.return_value.detect.return_value = boxes
                capture.return_value.get.side_effect = properties.__getitem__
                capture.return_value.read.side_effect = [(True, self.frame.copy()), (False, None)]
                track_video.main()
            fit.assert_called_once()
            row = json.loads(output.with_suffix('.jsonl').read_text())
            summary = json.loads(output.with_suffix('.json').read_text())
        self.assertEqual([track['team'] for track in row['tracks']], [1, 2, None])
        self.assertEqual({track['class_id'] for track in row['tracks']}, {0, 3})
        self.assertTrue({'possession', 'passes', 'ball_trajectory'} <= row.keys())
        self.assertEqual(summary['team_mode'], 'automatic')
        self.assertEqual(summary['team_classifier']['backend'], 'siglip')
