import json
import unittest

from pitchlens.movement import MovementAnalytics, MovementConfig


def player(x, y=10., *, track_id=1, class_id=0, status='valid', observed=True):
    return {'class_id': class_id, 'track_id': track_id, 'observed': observed,
            'pitch': {'xy_m': [x, y] if x is not None else None, 'status': status}}


class MovementTests(unittest.TestCase):
    def setUp(self):
        self.analytics = MovementAnalytics(10)

    def frame(self, index, tracks, *, segment=0, cut=False, provenance='direct', timestamp=None):
        return self.analytics.update(index, segment, cut, tracks, provenance=provenance,
                                     timestamp_seconds=timestamp)

    def observe(self, index, x, **kwargs):
        item = player(x)
        self.frame(index, [item], **kwargs)
        return item['movement']

    def test_stationary_speed_requires_consecutive_measurements(self):
        first = self.observe(0, 5)
        self.assertIsNone(first['speed_kmh'])
        self.assertIsNone(first['observed_distance_m'])
        self.assertIsNone(first['measurement_coverage_pct'])
        second = self.observe(1, 5)
        self.assertEqual(second['observed_distance_m'], 0.)
        self.assertIsNone(second['speed_kmh'])
        self.assertIsNone(second['maximum_speed_kmh'])
        third = self.observe(2, 5)
        for name in ('speed_kmh', 'average_speed_kmh', 'maximum_speed_kmh', 'observed_distance_m'):
            self.assertEqual(third[name], 0.)
        self.assertEqual(third['measurement_coverage_pct'], 100.)

    def test_ten_metres_over_two_seconds_at_different_frame_rates(self):
        for fps in (10, 24, 60):
            with self.subTest(fps=fps):
                self.analytics = MovementAnalytics(fps)
                for frame in range(2 * fps + 1):
                    result = self.observe(frame, 5 * frame / fps)
                self.assertAlmostEqual(result['observed_distance_m'], 10.)
                for name in ('speed_kmh', 'average_speed_kmh', 'maximum_speed_kmh'):
                    self.assertAlmostEqual(result[name], 18.)
                self.assertAlmostEqual(result['valid_measurement_duration_seconds'], 2.)
                self.assertAlmostEqual(result['observed_tracking_duration_seconds'], 2.)
                self.assertEqual(result['accepted_segments'], 2 * fps)

    def test_timestamps_use_actual_elapsed_time_including_first_interval(self):
        for index, timestamp in enumerate((0., .06, .22, .31)):
            track = player(5 * timestamp)
            timing = self.frame(index, [track], timestamp=timestamp)
            if index:
                self.assertEqual(timing['source'], 'video_timestamp')
        result = track['movement']
        self.assertAlmostEqual(result['observed_distance_m'], 1.55)
        self.assertAlmostEqual(result['valid_measurement_duration_seconds'], .31)
        self.assertAlmostEqual(result['speed_kmh'], 18.)
        self.assertAlmostEqual(result['average_speed_kmh'], 18.)

    def test_absent_and_constant_zero_timestamp_fall_back_to_fps(self):
        for timestamp in (None, 0.):
            with self.subTest(timestamp=timestamp):
                self.analytics = MovementAnalytics(10)
                self.observe(0, 0., timestamp=timestamp)
                item = player(.5)
                timing = self.frame(1, [item], timestamp=timestamp)
                self.assertEqual(timing, {'elapsed_seconds': .1, 'source': 'frame_index_fps'})
                result = self.observe(2, 1., timestamp=timestamp)
                self.assertAlmostEqual(result['speed_kmh'], 18.)

    def test_invalid_timestamps_break_both_adjacent_intervals(self):
        for bad in (.1, .01, 0., -1., float('nan'), float('inf')):
            with self.subTest(bad=bad):
                self.analytics = MovementAnalytics(10)
                self.observe(0, 0., timestamp=0.)
                self.observe(1, .5, timestamp=.1)
                for index, x, stamp in ((2, 8., bad), (3, 1., .3)):
                    self.assertIsNone(self.observe(index, x, timestamp=stamp)['speed_kmh'])
                result = self.observe(4, 1.5, timestamp=.4)
                self.assertEqual(result['observed_distance_m'], 1.)
                self.assertEqual(result['accepted_segments'], 2)
                self.assertEqual(result['rejected_segments'], 0)
                self.assertAlmostEqual(result['observed_tracking_duration_seconds'], .2)

    def test_player_and_goalkeeper_class_scoped_ids_and_excluded_classes(self):
        for index in range(3):
            tracks = [player(.5 * index), player(.25 * index, track_id=2),
                      player(index, class_id=1), player(index, class_id=2),
                      player(index, class_id=3), player(index, track_id=None),
                      player(index, track_id=0), player(index, track_id=True)]
            self.frame(index, tracks)
        for track, expected in zip(tracks[:3], (18., 9., 36.)):
            self.assertAlmostEqual(track['movement']['speed_kmh'], expected)
        for track in tracks[3:]:
            self.assertNotIn('movement', track)
        self.assertEqual([(t['class_id'], t['track_id']) for t in self.analytics.summary()],
                         [(0, 1), (0, 2), (1, 1)])

    def test_temporal_missing_estimated_and_invalid_coordinates_are_gaps(self):
        cases = [('temporal', player(0.8)), ('unavailable', player(0.8)),
                 ('direct', player(None)), ('direct', player(.8, status='estimated')),
                 ('direct', player(float('nan'))), ('direct', player(.8, float('inf'))),
                 ('direct', player('0.8'))]
        for provenance, track in cases:
            with self.subTest(provenance=provenance, track=track):
                self.analytics = MovementAnalytics(10)
                self.observe(0, 0.)
                self.observe(1, .5)
                self.frame(2, [track], provenance=provenance)
                self.assertIsNone(track['movement']['speed_kmh'])
                self.assertIsNone(self.observe(3, 20.)['speed_kmh'])
                result = self.observe(4, 20.5)
                self.assertIsNone(result['speed_kmh'])
                self.assertEqual(result['observed_distance_m'], 1.)
                self.assertAlmostEqual(result['measurement_coverage_pct'], 50.)
                self.assertEqual(result['rejected_segments'], 0)

    def test_missing_and_predicted_tracks_do_not_bridge_observations(self):
        self.observe(0, 0.)
        self.observe(1, .5)
        self.frame(2, [])
        self.assertIsNone(self.analytics.summary()[0]['speed_kmh'])
        self.observe(3, 20.)
        prediction = player(20.5, observed=False)
        self.frame(4, [prediction])
        self.assertIsNone(prediction['movement']['speed_kmh'])
        self.observe(5, 10.)
        result = self.observe(6, 10.5)
        self.assertEqual(result['observed_distance_m'], 1.)
        self.assertAlmostEqual(result['observed_tracking_duration_seconds'], .2)
        self.assertAlmostEqual(result['measurement_coverage_pct'], 100.)

    def test_changed_identity_and_skipped_frame_do_not_bridge(self):
        self.observe(0, 0.)
        self.observe(1, .5)
        different = player(.8, track_id=2)
        self.frame(2, [different])
        self.assertIsNone(different['movement']['observed_distance_m'])
        result = self.observe(4, 30.)
        self.assertEqual(result['observed_distance_m'], .5)
        self.assertIsNone(result['speed_kmh'])

    def test_camera_cut_keeps_track_ids_and_totals_separate(self):
        self.observe(0, 0.)
        self.observe(1, .5)
        self.observe(2, 1.)
        cut = self.observe(3, 40., segment=1, cut=True)
        self.assertIsNone(cut['observed_distance_m'])
        self.assertIsNone(cut['speed_kmh'])
        self.observe(4, 40.25, segment=1)
        self.observe(5, 40.5, segment=1)
        before, after = self.analytics.summary()
        self.assertEqual((before['segment'], after['segment']), (0, 1))
        self.assertEqual(before['track_id'], after['track_id'])
        self.assertIsNone(before['speed_kmh'])
        self.assertEqual(before['observed_distance_m'], 1.)
        self.assertEqual(after['observed_distance_m'], .5)
        self.assertAlmostEqual(after['speed_kmh'], 9.)

    def test_rejected_speed_breaks_window_and_discards_bad_endpoint(self):
        for index, x in enumerate((0., .5, 1.)):
            previous = self.observe(index, x)
        self.assertAlmostEqual(previous['speed_kmh'], 18.)
        rejected = self.observe(3, 50.)
        self.assertEqual(rejected['observed_distance_m'], 1.)
        self.assertEqual(rejected['rejected_segments'], 1)
        self.assertIsNone(rejected['speed_kmh'])
        self.assertEqual(self.observe(4, 1.5)['observed_distance_m'], 1.)
        result = self.observe(5, 2.)
        self.assertEqual(result['observed_distance_m'], 1.5)
        self.assertIsNone(result['speed_kmh'])
        self.assertAlmostEqual(result['maximum_speed_kmh'], 18.)
        self.assertAlmostEqual(result['measurement_coverage_pct'], 60.)

    def test_average_and_peak_use_accepted_duration_and_smoothed_speed(self):
        self.analytics = MovementAnalytics(8)
        for index, x in enumerate((0., .25, 1.25, 1.5, 1.5)):
            result = self.observe(index, x)
        self.assertAlmostEqual(result['speed_kmh'], 3.6)
        self.assertAlmostEqual(result['maximum_speed_kmh'], 18.)
        self.assertAlmostEqual(result['average_speed_kmh'], 10.8)
        self.assertEqual(result['observed_distance_m'], 1.5)

    def test_configurable_minimum_and_speed_limit(self):
        self.analytics = MovementAnalytics(10, MovementConfig(max_speed_kmh=20., min_observations=4))
        for index, x in enumerate((0., .25, .5, .75, 1.5)):
            result = self.observe(index, x)
            if index < 3:
                self.assertIsNone(result['speed_kmh'])
            elif index == 3:
                self.assertAlmostEqual(result['speed_kmh'], 9.)
        self.assertEqual(result['rejected_segments'], 1)
        self.assertIsNone(result['speed_kmh'])

    def test_json_uses_null_without_valid_measurements(self):
        first = self.observe(0, None)
        second = self.observe(1, None)
        serialized = json.loads(json.dumps([first, second, self.analytics.summary()], allow_nan=False))
        for state in serialized[:2]:
            for key in ('speed_kmh', 'maximum_speed_kmh', 'average_speed_kmh', 'observed_distance_m'):
                self.assertIsNone(state[key])
        self.assertEqual(serialized[1]['valid_measurement_duration_seconds'], 0.)
        self.assertEqual(serialized[1]['measurement_coverage_pct'], 0.)

    def test_invalid_fps_and_configuration(self):
        for fps in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                MovementAnalytics(fps)
        for settings in ({'max_speed_kmh': 0}, {'smoothing_seconds': float('inf')}, {'min_observations': 2},
                         {'min_observations': 3.5}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                MovementConfig(**settings)
