import hashlib
from io import BytesIO
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

import cv2
import numpy as np

from pitchlens.calibration import PitchCalibrator, PitchConfig, project
from scripts.evaluate_calibration import (
    compare_calibrators, compare_frame, roboflow_labels, soccernet_rows, summarize, video_decode_status,
)
from scripts.score_planar_calibration import planar_markings, read_planar_labels, score_planar, visible_segments


class CalibrationEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = PitchConfig()
        self.markings = planar_markings(self.config)
        self.frame = np.zeros((540, 960, 3), dtype=np.uint8)
        self.camera = np.array([[6., 0.2, 90.], [0.1, 5., 50.], [0.001, 0.002, 1.]])
        self.world = self.config.landmarks()

    def test_original_template_and_single_inference_for_both_fits(self):
        pair = compare_calibrators(self.config)
        differences = (pair['original'].world != pair['corrected'].world).any(axis=1)
        self.assertEqual(np.flatnonzero(differences).tolist(), [10, 11, 18, 19])
        left, right = pair['original'].world[[10, 11], 1]
        np.testing.assert_allclose([left, right], [24.84, 43.16])
        points = project(self.camera, self.world)
        observations = np.column_stack((points, np.ones(32)))
        calls = []

        class Detector:
            def detect(self, frame):
                calls.append(frame.shape)
                return observations

        row = compare_frame(Detector(), self.frame, pair, expected=(points, np.ones(32, dtype=bool)))
        self.assertEqual(calls, [self.frame.shape])
        self.assertEqual(len(row['annotation']['raw_px']), 32)
        self.assertEqual(max(row['annotation']['raw_px'].values()), 0.)
        self.assertLess(max(row['annotation']['held_out_arc_px']['corrected'].values()), 1e-4)
        self.assertGreater(min(row['annotation']['held_out_arc_px']['original'].values()), 5.)

    def test_roboflow_excludes_occluded_and_unlabeled_points(self):
        coords = np.tile([0.5, 0.75, 0], (32, 1))
        coords[:2, 2] = [1, 2]
        data = '0 0.5 0.5 1 1 ' + ' '.join(map(str, coords.ravel()))
        points, visible = roboflow_labels(data, 960, 540)
        np.testing.assert_array_equal(points[1], [480, 405])
        self.assertEqual(np.flatnonzero(visible).tolist(), [1])
        with self.assertRaises(ValueError):
            roboflow_labels('0 1', 960, 540)

    def test_planar_labels_clamp_deduplicate_and_exclude_unmodeled_points(self):
        raw = {
            'Middle line': [{'x': -0.001, 'y': 0.5}, {'x': 0, 'y': 0.5},
                            {'x': 1.01, 'y': 0.5}, {'x': float('nan'), 'y': 0}],
            'Goal left crossbar': [{'x': 0.5, 'y': 0.4}],
        }
        labels, excluded = read_planar_labels(raw, self.markings, 960, 540)
        np.testing.assert_array_equal(labels['Middle line'], [[0, 270]])
        self.assertEqual(excluded['Middle line'], {
            'out_of_bounds_or_nonfinite': 2, 'clamped': 1, 'duplicates': 1,
        })
        self.assertEqual(excluded['Goal left crossbar'], {'unmodeled_points': 1})

    def test_planar_score_accepts_one_global_half_turn_including_arcs(self):
        points = project(self.camera, self.world)
        observations = np.column_stack((points, np.ones(32)))
        fit = PitchCalibrator(self.config).update(observations, self.frame.shape)
        size = np.array([self.config.length_m, self.config.width_m])
        names = ['Side line top', 'Big rect. right main', 'Circle left']
        labels = {name: project(self.camera, size - self.markings[name]) for name in names}
        distances = score_planar(self.markings, labels, fit)
        self.assertLess(max(value for values in distances.values() for value in values), 1e-4)
        labels['Side line top'] = project(self.camera, self.markings['Side line top'])
        distances = score_planar(self.markings, labels, fit)
        self.assertGreater(max(distances['Side line top']), 100)
        self.assertIsNone(score_planar(self.markings, labels,
                                       PitchCalibrator(self.config).update(None, self.frame.shape)))

    def test_clipping_excludes_pitch_behind_projective_horizon(self):
        camera = np.array([[1., 0, 0], [0, 1., 0], [1., 0, 1.]])
        path = np.array([[-2., 0.5], [3., 0.5]])
        np.testing.assert_allclose(visible_segments(path, camera, 5, 5), [[[0, 0.5], [0.75, 0.125]]])
        self.assertEqual(len(visible_segments(path, -camera, 5, 5)), 0)

    def test_rejections_and_unprojectable_points_count_against_coverage(self):
        accepted = dict(valid=True, stabilized=False, landmarks=6, reprojection_rmse_px=1,
                        planar={'Middle line': [2., None]})
        rejected = dict(valid=False, landmarks=4, reason='too_few_landmarks', planar=None)
        rows = [dict(id='a', cut=False, original=accepted, corrected=rejected, usable_markings=1, usable_points=2),
                dict(id='b', cut=False, original=rejected, corrected=accepted, usable_markings=1, usable_points=2)]
        report = summarize(rows, 'soccernet')
        original = report['original']
        self.assertEqual(original['fraction_within_5px_on_accepted'], 0.5)
        self.assertEqual(original['fraction_within_5px_with_rejections'], 0.25)
        self.assertEqual(original['distance_px'], dict(count=2, finite=1, p50=2., p90=None))

    def test_soccernet_reads_only_pinned_archive_and_rejects_changed_labels_or_index(self):
        image = '00000.jpg'
        ok, data = cv2.imencode('.jpg', self.frame)
        self.assertTrue(ok)
        index = {image: {'match': 'fixture'}}
        labels = {'Middle line': [{'x': 0.5, 'y': 0.4}]}

        def make_archive(match_index, annotations):
            stream = BytesIO()
            with ZipFile(stream, 'w') as archive:
                archive.writestr(f'valid/{image}', data.tobytes())
                archive.writestr('valid/00000.json', json.dumps(annotations))
                archive.writestr('valid/match_info.json', json.dumps(match_index))
            return stream.getvalue()

        trusted = make_archive(index, labels)
        detector = Mock()
        points = project(self.camera, self.world)
        detector.detect.return_value = np.column_stack((points, np.ones(32)))
        path = Mock()
        args = SimpleNamespace(input=path)
        with patch('scripts.evaluate_calibration.SOCCERNET_SHA', hashlib.sha256(trusted).hexdigest()), \
             patch('scripts.evaluate_calibration.SOCCERNET_IMAGES', 1):
            path.open.return_value = BytesIO(trusted)
            rows = list(soccernet_rows(args, detector, compare_calibrators(self.config), self.markings))
            self.assertEqual(rows[0]['event'], index[image])
            self.assertEqual(rows[0]['usable_points'], 1)
            for altered in (make_archive(index, {}), make_archive({image: {'match': 'edited'}}, labels)):
                path.open.return_value = BytesIO(altered)
                with self.assertRaisesRegex(ValueError, 'publisher SHA-256'):
                    list(soccernet_rows(args, detector, compare_calibrators(self.config), self.markings))
        detector.detect.assert_called_once()

    def test_decode_status_distinguishes_early_end_and_intentional_limit(self):
        self.assertEqual(video_decode_status(391, 390, None), {
            'reported_frames': 391, 'decoded_frames': 390, 'stop_before': None,
            'status': 'frame_count_mismatch',
        })
        self.assertEqual(video_decode_status(928, 928, None)['status'], 'complete')
        self.assertEqual(video_decode_status(928, 545, 545)['status'], 'frame_limit')
        self.assertEqual(video_decode_status(928, 540, 545)['status'], 'early_end')
        self.assertEqual(video_decode_status(0, 390, None)['status'], 'unreported_frame_count')
