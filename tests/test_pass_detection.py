import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from pitchlens.pass_detection import PassConfig, PassDetector
from pitchlens.possession import PossessionAnalytics


def player(x=50., track_id=1, team='A', cls=0, observed=True):
    return {'class_id': cls, 'track_id': track_id, 'team': team, 'observed': observed,
            'xyxy': [x - 5, 30., x + 5, 70.]}


def ball(x, y=70., observed=True):
    return {'class_id': 3, 'track_id': None, 'observed': observed, 'xyxy': [x - 2, y - 2, x + 2, y + 2]}


class PassDetectionTests(unittest.TestCase):
    def setUp(self):
        self.passes = PassDetector(20, ('A', 'B'))
        self.players = [player(), player(150., 2)]

    def sequence(self, balls, *, start=0, players=None, segment=0):
        group = self.players if players is None else players
        return [self.passes.update(start + index, segment, False, [*group, *([sample] if sample else [])])
                for index, sample in enumerate(balls)]

    def standard(self):
        return [ball(x) for x in (50., 50., 76., 98., 121., 150., 150.)]

    def test_same_team_pass_and_statistics_without_pitch(self):
        outputs = self.sequence(self.standard())
        self.assertTrue(all(not frame['events'] for frame in outputs[:-1]))
        event, = outputs[-1]['events']
        self.assertEqual(event['classification'], 'likely_successful_pass')
        self.assertEqual((event['source']['track_id'], event['receiver']['track_id']), (1, 2))
        self.assertEqual((event['release_frame'], event['reception_frame'], event['confirmation_frame']), (2, 5, 6))
        self.assertEqual(event['observed_travel_samples'], 3)
        self.assertAlmostEqual(event['duration_seconds'], .15)
        self.assertAlmostEqual(event['max_observation_gap_seconds'], .05)
        self.assertEqual([sample['frame_index'] for sample in event['evidence']['observed_trajectory']], [1, 2, 3, 4, 5])
        self.assertEqual(event['evidence']['reception_contact_distance'], 0.)
        report = json.loads(json.dumps(self.passes.finish(), allow_nan=False))
        self.assertEqual(report['events'], [event])
        self.assertEqual(report['teams'][0]['detected_passes'], 1)
        self.assertEqual(report['tracks'][0]['detected_passes'], 1)
        self.assertEqual(report['tracks'][1]['received_passes'], 1)
        self.assertEqual(report['observed_ball_coverage_pct'], 100.)

    def test_opponent_turnover_and_unknown_team(self):
        for other, expected in (('B', 'possible_turnover'), (None, 'unclassified_transfer')):
            with self.subTest(team=other):
                self.setUp()
                self.players[1]['team'] = other
                event, = self.sequence(self.standard())[-1]['events']
                self.assertEqual(event['classification'], expected)
                self.assertEqual(event['receiver']['team'], other)
                stats = self.passes.finish()
                self.assertEqual(stats['likely_successful_passes'], 0)
                self.assertEqual(stats['tracks'][0][f'{expected}s_from'], 1)
                if other == 'B':
                    self.assertEqual(stats['teams'][1]['possible_turnovers_to'], 1)

    def test_inconsistent_team_on_supported_contact_is_unclassified(self):
        for index, sample in enumerate(self.standard()):
            first_team = 'B' if index == 1 else 'A'
            result = self.passes.update(index, 0, False, [player(team=first_team), self.players[1], sample])
        event, = result['events']
        self.assertEqual(event['classification'], 'unclassified_transfer')
        self.assertIsNone(event['source']['team'])

    def test_goalkeeper_eligible_and_referee_excluded(self):
        self.players[0]['class_id'] = 1
        event, = self.sequence(self.standard())[-1]['events']
        self.assertEqual(event['source']['class_id'], 1)
        self.setUp()
        self.players[0]['class_id'] = 2
        self.assertFalse(any(frame['events'] for frame in self.sequence(self.standard())))

    def test_no_transfer_during_possession_dribble_or_camera_pan(self):
        for frame in range(15):
            offset = 8. * frame
            tracks = [player(50 + offset), player(150 + offset, 2), ball(50 + offset)]
            result = self.passes.update(frame, 0, False, tracks)
            self.assertFalse(result['events'])
        self.assertEqual(self.passes.finish()['rejected_candidates'], 0)
        self.setUp()
        # The player moves away from a stationary image point and another arrives.
        for index, source_x in enumerate((50, 50, 75, 85, 100, 120, 130)):
            receiver_x = 150 if index < 5 else 50
            result = self.passes.update(index, 0, False, [player(source_x), player(receiver_x, 2), ball(50.)])
            self.assertFalse(result['events'])
        self.assertEqual(self.passes.finish()['rejection_reasons'], {'insufficient_travel': 1})

    def test_unsupported_ambiguous_or_predicted_source_cannot_launch(self):
        samples = self.standard()
        for source, first in (
            (self.players, [None, samples[1]]),
            (self.players, [ball(50., observed=False)] * 2),
            ([*self.players, player(52., 3, 'B')], samples[:2]),
            ([player(observed=False), self.players[1]], samples[:2]),
            ([self.players[1]], samples[:2]),
        ):
            with self.subTest(source=source, first=first):
                self.passes = PassDetector(20, ('A', 'B'))
                self.sequence(first, players=source)
                outputs = self.sequence(samples[2:], start=2)
                self.assertFalse(any(frame['events'] for frame in outputs))

    def test_confirmed_possession_can_corroborate_but_not_override_ambiguity(self):
        possession = {'state': 'confirmed', 'owner': {'segment': 0, 'class_id': 0, 'track_id': 1}}
        self.passes.update(0, 0, False, [*self.players, ball(50.)], possession=possession)
        result = self.sequence([ball(x) for x in (76., 98., 121., 150., 150.)], start=1)[-1]
        self.assertEqual(len(result['events']), 1)
        self.setUp()
        self.passes.update(0, 0, False, [*self.players, player(52., 3), ball(50.)], possession=possession)
        result = self.sequence([ball(x) for x in (76., 98., 121., 150., 150.)], start=1)[-1]
        self.assertFalse(result['events'])

    def test_ambiguous_or_fleeting_receiver_rejects_flight(self):
        for tracks, reason in (([*self.players, player(151., 3)], 'ambiguous_owner'),
                               ([*self.players], 'receiver_contact_lost')):
            with self.subTest(reason=reason):
                self.passes = PassDetector(20, ('A', 'B'))
                self.sequence(self.standard()[:5])
                run = self.sequence([ball(150.), None], start=5, players=tracks)
                self.assertIn(reason, [frame['rejection_reason'] for frame in run])
                self.assertFalse(any(frame['events'] for frame in run))

    def test_same_player_recontact_and_direct_nearest_switch_do_not_count(self):
        outputs = self.sequence([ball(x) for x in (50., 50., 76., 98., 50., 50.)])
        self.assertEqual(outputs[4]['rejection_reason'], 'source_recontact')
        self.assertFalse(any(frame['events'] for frame in outputs))
        self.setUp()
        outputs = self.sequence([ball(x) for x in (50., 50., 150., 150.)])
        self.assertFalse(any(frame['events'] for frame in outputs))

    def test_insufficient_travel_and_too_few_intermediate_samples(self):
        close = [player(), player(72., 2)]
        samples = [ball(50.), ball(50.), ball(60., 100.), ball(62., 112.), ball(60., 120.), ball(72.)]
        self.assertEqual(self.sequence(samples, players=close)[-1]['rejection_reason'], 'insufficient_travel')
        self.passes = PassDetector(10, ('A', 'B'))
        result = self.sequence([ball(x) for x in (50., 50., 80., 150.)])[-1]
        self.assertEqual(result['rejection_reason'], 'insufficient_travel_samples')

    def test_short_gap_needs_observed_travel_on_both_sides(self):
        samples = [ball(50.), ball(50.), ball(76.), ball(95., observed=False), ball(102.),
                   ball(121.), ball(150.), ball(150.)]
        event, = self.sequence(samples)[-1]['events']
        self.assertTrue(event['evidence']['bridged_missing_ball'])
        self.assertEqual(event['observed_travel_samples'], 3)
        self.assertAlmostEqual(event['max_observation_gap_seconds'], .10)
        self.assertNotIn(3, [p['frame_index'] for p in event['evidence']['observed_trajectory']])
        self.setUp()
        samples = [ball(50.), ball(50.), ball(76.), None, ball(121.), ball(150.)]
        self.assertEqual(self.sequence(samples)[-1]['rejection_reason'], 'insufficient_travel')
        self.setUp()
        samples = [ball(50.), ball(50.), ball(76.), ball(98.), None, ball(150.)]
        self.assertEqual(self.sequence(samples)[-1]['rejection_reason'], 'observation_gap_at_reception')

    def test_excessive_gap_and_expired_flight_are_counted_once(self):
        self.sequence([ball(50.), ball(50.), ball(76.), *([None] * 6), ball(121.), ball(150.), ball(150.)])
        self.assertEqual(self.passes.finish()['rejection_reasons'], {'observation_gap': 1})
        self.passes = PassDetector(20, ('A', 'B'), PassConfig(max_transfer_seconds=.25))
        self.sequence([ball(50.), ball(50.), *([ball(100.)] * 8), ball(150.), ball(150.)])
        self.assertEqual(self.passes.finish()['rejection_reasons'], {'transfer_timeout': 1})
        self.setUp()
        self.sequence(self.standard()[:4])
        self.assertEqual(self.passes.finish()['rejection_reasons'], {'clip_end': 1})
        self.assertEqual(self.passes.finish()['rejection_reasons'], {'clip_end': 1})

    def test_cuts_cancel_and_reused_ids_stay_separate(self):
        self.sequence(self.standard()[:4])
        cut = self.passes.update(4, 1, True, [*self.players, ball(121.)])
        self.assertEqual(cut['rejection_reason'], 'camera_cut')
        self.assertFalse(self.sequence(self.standard()[5:], start=5, segment=1)[-1]['events'])
        self.sequence(self.standard(), start=7, segment=1)
        self.sequence(self.standard(), start=14, segment=2)
        report = self.passes.finish()
        self.assertEqual([e['segment'] for e in report['events']], [1, 2])
        self.assertEqual([t['segment'] for t in report['tracks']], [1, 1, 2, 2])

    def test_consecutive_transfers_and_duplicate_prevention(self):
        self.players.append(player(250., 3))
        self.sequence(self.standard())
        second = self.sequence([ball(x) for x in (176., 198., 221., 250., 250., 250., 250.)], start=7)
        event, = second[4]['events']
        self.assertEqual(event['event_id'], 2)
        self.assertEqual((event['source']['track_id'], event['receiver']['track_id']), (2, 3))
        self.assertFalse(any(frame['events'] for frame in second[5:]))

    def test_actual_timestamps_and_frame_rates(self):
        for fps in (12, 24, 60):
            with self.subTest(fps=fps):
                self.passes = PassDetector(fps, ('A', 'B'))
                locations = [50. if t < .12 else 74. + (t - .12) * 250 if t < .32 else 150.
                             for t in (index / fps for index in range(fps))]
                self.sequence([ball(x) for x in locations])
                self.assertEqual(self.passes.finish()['likely_successful_passes'], 1)
        self.setUp()
        for index, (sample, stamp) in enumerate(zip(self.standard(), (7., 7.06, 7.10, 7.19, 7.23, 7.28, 7.36))):
            output = self.passes.update(index, 0, False, [*self.players, sample], timestamp_seconds=stamp)
        event, = output['events']
        self.assertAlmostEqual(event['release_timestamp_seconds'], 7.10)
        self.assertAlmostEqual(event['reception_timestamp_seconds'], 7.28)
        self.assertAlmostEqual(event['duration_seconds'], .18)

    def test_invalid_inputs_and_tracking_discontinuity(self):
        for override in ({'max_normalized_contact_distance': math.nan}, {'min_contact_margin': .6},
                         {'min_transfer_seconds': 5.}, {'min_travel_samples': 1}):
            with self.assertRaises(ValueError):
                PassConfig(**override)
        with self.assertRaises(ValueError):
            PassDetector(0, ('A', 'B'))
        self.sequence(self.standard()[:4])
        output = self.passes.update(4, 0, False, [*self.players, ball(121.)], timestamp_seconds=math.nan)
        self.assertEqual(output['rejection_reason'], 'invalid_timing')
        self.setUp()
        self.sequence(self.standard()[:4])
        self.assertEqual(self.sequence([ball(math.inf)], start=4)[0]['rejection_reason'], 'invalid_ball')
        self.setUp()
        self.sequence(self.standard()[:4])
        self.assertEqual(self.sequence([ball(121.)], start=4, players=self.players[1:])[0]['rejection_reason'],
                         'source_track_lost')

    def test_observed_ambiguity_flag_rejects_and_running_possession_is_preserved(self):
        possession = PossessionAnalytics(20, ('A', 'B'))
        control = PossessionAnalytics(20, ('A', 'B'))
        for index, sample in enumerate(self.standard()):
            tracks = [*self.players, sample]
            ambiguous = index == 4
            shared = possession.update(index, 0, False, tracks, ambiguous_ball=ambiguous)
            result = self.passes.update(index, 0, False, tracks, ambiguous_ball=ambiguous, possession=shared)
            self.assertEqual(shared, control.update(index, 0, False, tracks, ambiguous_ball=ambiguous))
            if ambiguous:
                self.assertEqual(result['rejection_reason'], 'ambiguous_ball')
        report = self.passes.finish()
        self.assertEqual(report['detected_transfers'], 0)
        self.assertEqual(report['observed_ball_frames'], 7)
        self.assertEqual(report['usable_ball_frames'], 6)

    def test_runner_exports_pass_without_pitch_and_clears_overlay(self):
        import cv2
        import numpy as np
        from pitchlens.detector import Detection
        from scripts import track_video

        path = [50., 50., 76., 98., 121., 150., 150., 150.]
        locations = path + path + [150.] * 30
        capture = Mock()
        capture.get.side_effect = {cv2.CAP_PROP_FPS: 20., cv2.CAP_PROP_FRAME_WIDTH: 320.,
                                   cv2.CAP_PROP_FRAME_HEIGHT: 180., cv2.CAP_PROP_POS_MSEC: 0.}.__getitem__
        capture.read.side_effect = [(True, np.full((180, 320, 3), 20 if i < 8 else 220, dtype=np.uint8))
                                    for i in range(len(locations))] + [(False, None)]
        detector = Mock(device='cpu', image_size=960)
        current = []

        def detect(_frame):
            index = len(current)
            current.append(index)
            items = [*self.players, ball(locations[index])]
            return [Detection(tuple(item['xyxy']), item['class_id'], .95) for item in items]

        detector.detect.side_effect = detect
        team_classifier = Mock(config=track_video.TeamConfig())
        team_classifier.update.side_effect = lambda _, tracked: [1 if t.class_id == 0 else None for t in tracked.tracks]
        shown = []
        draw = track_video.draw_pass_event

        def capture_overlay(frame, event, teams):
            shown.append((current[-1], event['event_id']))
            draw(frame, event, teams)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'transfers.mp4'
            args = ['track_video.py', '--input', 'fixture.mp4', '--output', str(output), '--weights', 'fixture.pt']
            with (patch.object(sys, 'argv', args), patch.object(cv2, 'VideoCapture', return_value=capture),
                  patch.object(cv2, 'VideoWriter', return_value=Mock()),
                  patch.object(track_video, 'Detector', return_value=detector),
                  patch.object(track_video, 'TeamClassifier', return_value=team_classifier),
                  patch.object(track_video, 'draw_pass_event', side_effect=capture_overlay), patch('builtins.print')):
                track_video.main()
            rows = [json.loads(line) for line in output.with_suffix('.jsonl').read_text().splitlines()]
            summary = json.loads(output.with_suffix('.json').read_text())
        self.assertNotIn('pitch', summary)
        self.assertTrue(rows[8]['camera_cut'])
        events = [event for row in rows for event in row['passes']['events']]
        self.assertEqual(events, summary['passes']['events'])
        self.assertEqual([(e['segment'], e['confirmation_frame']) for e in events], [(0, 6), (1, 14)])
        self.assertEqual([(frame, event) for frame, event in shown if event == 1], [(6, 1), (7, 1)])
        self.assertEqual([(frame, event) for frame, event in shown if event == 2],
                         [(i, 2) for i in range(14, 39)])
