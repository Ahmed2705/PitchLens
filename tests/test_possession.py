import json
import math
import unittest

from pitchlens.possession import PossessionAnalytics, PossessionConfig


def person(x=50., *, height=40., width=10., track_id=1, class_id=0, team='A', observed=True):
    return {'class_id': class_id, 'track_id': track_id, 'team': team, 'observed': observed,
            'xyxy': [x - width / 2, 70. - height, x + width / 2, 70.]}


def ball(x=50., y=70., *, observed=True):
    return {'class_id': 3, 'track_id': None, 'observed': observed, 'xyxy': [x - 2, y - 2, x + 2, y + 2]}


class PossessionTests(unittest.TestCase):
    def setUp(self):
        self.analytics = PossessionAnalytics(10, ('A', 'B'))

    def update(self, frame, tracks, *, segment=0, cut=False, timestamp=None, ambiguous_ball=False):
        return self.analytics.update(frame, segment, cut, tracks, timestamp_seconds=timestamp,
                                     ambiguous_ball=ambiguous_ball)

    def test_single_near_player_confirms_then_earns_forward_time(self):
        tracks = [person(), ball(56., 62.)]
        start, confirmed, credited = [self.update(i, tracks) for i in range(3)]
        self.assertEqual(start['state'], 'pending')
        self.assertIsNone(start['owner'])
        self.assertEqual(start['pending_candidate']['track_id'], 1)
        self.assertEqual(start['normalized_distance'], .025)
        self.assertEqual(confirmed['state'], 'confirmed')
        self.assertEqual(confirmed['owner'], {'segment': 0, 'class_id': 0, 'track_id': 1, 'team': 'A'})
        self.assertEqual(confirmed['timing']['credited_player_seconds'], 0.)
        self.assertEqual(credited['timing']['credited_player_seconds'], .1)
        self.assertEqual(credited['timing']['credited_team'], 'A')
        self.assertNotIn('pitch', tracks[0])

    def test_selects_by_normalized_distance_and_preserves_image_scale(self):
        # First region is closer in pixels, second is closer after normalization.
        original = [person(54., height=8., width=4.), person(59., height=40., track_id=2), ball()]
        for scale in (1., 2.5):
            with self.subTest(scale=scale):
                self.analytics = PossessionAnalytics(10, ('A', 'B'))
                scaled = [{**item, 'xyxy': [value * scale for value in item['xyxy']]} for item in original]
                result = self.update(0, scaled)
                self.assertEqual(result['pending_candidate']['track_id'], 2)
                self.assertAlmostEqual(result['normalized_distance'], .1)

    def test_threshold_and_margin_only_among_nearby_candidates(self):
        result = self.update(0, [person(32.), person(68., track_id=2), ball()])
        self.assertEqual(result['reason'], 'no_nearby_player')
        result = self.update(1, [person(33.), person(68., track_id=2), ball()])
        self.assertEqual(result['state'], 'pending')
        self.assertEqual(result['normalized_distance'], .3)
        # A margin of exactly 0.1 is sufficient.
        tracks = [person(41.), person(63., track_id=2), ball()]
        self.analytics = PossessionAnalytics(10, ('A', 'B'))
        self.update(0, tracks)
        result = self.update(1, tracks)
        self.assertEqual(result['owner']['track_id'], 1)

    def test_relative_motion_rejects_a_ball_traveling_through_the_lower_region(self):
        self.analytics = PossessionAnalytics(20, ('A', 'B'))
        results = [self.update(i, [person(), ball(x)]) for i, x in enumerate((35., 40., 45., 50., 55., 60., 65.))]
        self.assertEqual(results[0]['state'], 'pending')
        self.assertTrue(all(result['reason'] == 'relative_ball_travel' for result in results[1:]))
        self.assertTrue(all(result['owner'] is None for result in results))
        self.assertEqual(self.analytics.summary()['confirmed_player_duration_seconds'], 0.)

    def test_shared_image_motion_preserves_control(self):
        self.analytics = PossessionAnalytics(20, ('A', 'B'))
        for i in range(4):
            x = 50. + 12 * i
            result = self.update(i, [person(x), ball(x + 4., 67.)])
        self.assertEqual(result['owner']['track_id'], 1)
        self.assertEqual(result['totals']['known_team_duration_seconds'], .05)

    def test_short_tracking_interruptions_restore_owner_without_time_credit(self):
        for missing in ([person()], [person(), ball(observed=False)], [ball()]):
            with self.subTest(missing=missing):
                self.analytics = PossessionAnalytics(20, ('A', 'B'))
                tracks = [person(), ball()]
                for i in range(4):
                    self.update(i, tracks)
                interrupted = self.update(4, missing)
                self.assertIsNone(interrupted['owner'])
                resumed = self.update(5, tracks)
                self.assertEqual(resumed['owner']['track_id'], 1)
                self.assertEqual(resumed['timing']['credited_player_seconds'], 0.)
                self.assertEqual(self.update(6, tracks)['timing']['credited_player_seconds'], .05)
                self.assertEqual(self.analytics.summary()['episodes'], 1)

    def test_predicted_ball_cannot_establish_owner_and_sustained_separation_expires_hold(self):
        self.analytics = PossessionAnalytics(20, ('A', 'B'))
        for i in range(4):
            self.assertIsNone(self.update(i, [person(), ball(observed=False)])['owner'])
        close = [person(), ball()]
        for i in range(4, 7):
            result = self.update(i, close)
        self.assertEqual(result['state'], 'confirmed')
        for i in range(7, 10):
            self.assertIsNone(self.update(i, [person(), ball(85.)])['owner'])
        self.assertEqual(self.update(10, close)['state'], 'pending')

    def test_contest_suspends_public_owner_and_blocks_credit(self):
        self.analytics = PossessionAnalytics(20, ('A', 'B'))
        tracks = [person(), ball()]
        for i in range(4):
            self.update(i, tracks)
        result = self.update(4, [*tracks, person(track_id=2, team='B')])
        self.assertEqual(result['reason'], 'ambiguous_owner')
        self.assertIsNone(result['owner'])
        self.assertEqual(result['timing']['credited_player_seconds'], 0.)

    def test_contested_ball_is_unknown_for_same_and_different_teams(self):
        for team in ('A', 'B'):
            with self.subTest(team=team):
                self.analytics = PossessionAnalytics(10, ('A', 'B'))
                for frame in range(4):
                    result = self.update(frame, [person(49.), person(52., track_id=2, team=team), ball()])
                    self.assertEqual(result['state'], 'unknown')
                    self.assertEqual(result['reason'], 'ambiguous_owner')
                    self.assertIsNone(result['owner'])
                self.assertEqual(result['totals']['unknown_player_duration_seconds'], .1 * 3)

    def test_goalkeeper_is_eligible_and_referee_is_ignored(self):
        tracks = [person(50., class_id=2), person(55., class_id=1, team='B'), ball()]
        for frame in range(3):
            result = self.update(frame, tracks)
        self.assertEqual(result['owner']['class_id'], 1)
        self.assertEqual(result['owner']['team'], 'B')
        self.assertEqual(self.analytics.summary()['tracks'][0]['class_id'], 1)

    def test_unknown_team_preserves_player_credit(self):
        tracks = [person(class_id=1, team=None), ball()]
        for frame in range(3):
            result = self.update(frame, tracks)
        self.assertIsNone(result['owner']['team'])
        self.assertEqual(result['timing']['credited_player_seconds'], .1)
        self.assertIsNone(result['timing']['credited_team'])
        totals = result['totals']
        self.assertEqual(totals['player_with_unknown_team_duration_seconds'], .1)
        self.assertEqual(totals['unknown_player_duration_seconds'], .1)
        self.assertEqual(totals['unknown_team_duration_seconds'], .2)
        self.assertEqual(totals['team_possession_coverage_pct'], 0.)
        self.assertEqual([t['possession_pct'] for t in totals['teams']], [None, None])

    def test_team_label_changes_do_not_split_player_episode_or_credit_mismatch(self):
        for frame, team in enumerate(('A', 'A', 'A', None, 'B', 'B')):
            result = self.update(frame, [person(team=team), ball()])
        summary = self.analytics.summary()
        self.assertEqual(summary['episodes'], 1)
        self.assertIsNone(summary['tracks'][0]['team'])
        self.assertAlmostEqual(summary['confirmed_player_duration_seconds'], .4)
        self.assertAlmostEqual(summary['player_with_unknown_team_duration_seconds'], .2)
        self.assertAlmostEqual(summary['teams'][0]['duration_seconds'], .1)
        self.assertAlmostEqual(summary['teams'][1]['duration_seconds'], .1)
        self.assertEqual(result['timing']['credited_team'], 'B')

    def test_invalid_missing_estimated_and_ambiguous_ball_reset_owner(self):
        observed = [person(), ball()]
        invalid_ball = {**ball(), 'xyxy': [40., 68., math.nan, 72.]}
        for items, reason, flag in (
            ([person()], 'missing_ball', False),
            ([person(), ball(observed=False)], 'unobserved_ball', False),
            ([person(), invalid_ball], 'invalid_ball', False),
            ([person(), ball(), ball(60.)], 'ambiguous_ball', False),
            (observed, 'ambiguous_ball', True),
        ):
            with self.subTest(reason=reason, flag=flag):
                self.analytics = PossessionAnalytics(10, ('A', 'B'))
                for frame in range(3):
                    self.update(frame, observed)
                result = self.update(3, items, ambiguous_ball=flag)
                self.assertEqual(result['reason'], reason)
                self.assertIsNone(result['owner'])
                self.assertEqual(result['totals']['confirmed_player_duration_seconds'], .1)
                self.assertEqual(self.update(4, observed)['state'], 'pending')

    def test_invalid_stale_and_missing_player_observations_are_ineligible(self):
        tracks = [ball(), person(observed=False), person(track_id=None), person(track_id=0),
                  person(track_id=True), person(height=0.), person(height=-1.),
                  {**person(), 'xyxy': [math.inf, 30., 55., 70.]},
                  {**person(), 'xyxy': [-1., 30., 55., 70.]}]
        self.update(0, [person(), ball()])
        self.assertEqual(self.update(1, [person(), ball()])['state'], 'confirmed')
        result = self.update(2, tracks)
        self.assertEqual(result['reason'], 'no_nearby_player')
        self.assertEqual(result['totals']['confirmed_player_duration_seconds'], 0.)

    def test_short_transient_candidate_and_frame_rate_aware_confirmation(self):
        for fps in (10, 24, 60):
            with self.subTest(fps=fps):
                self.analytics = PossessionAnalytics(fps, ('A', 'B'))
                first_confirmed = math.ceil(.1 * fps)
                for index in range(first_confirmed):
                    self.assertEqual(self.update(index, [person(), ball()])['state'], 'pending')
                other = [person(track_id=2), ball()]
                self.assertEqual(self.update(first_confirmed, other)['state'], 'pending')
                self.assertEqual(self.analytics.summary()['episodes'], 0)
                self.analytics = PossessionAnalytics(fps, ('A', 'B'))
                for index in range(fps + 1):
                    result = self.update(index, [person(), ball()])
                    if index == first_confirmed:
                        self.assertEqual(result['state'], 'confirmed')
                        self.assertEqual(result['timing']['credited_player_seconds'], 0.)
                self.assertAlmostEqual(result['totals']['analyzed_duration_seconds'], 1.)
                self.assertAlmostEqual(result['totals']['known_team_duration_seconds'], 1 - first_confirmed / fps)
                self.assertEqual(self.analytics.summary()['episodes'], 1)

    def test_ten_second_reconciliation_and_ownership_transfer(self):
        for frame in range(101):
            if frame <= 61:
                tracks = [person(), ball()]
            elif frame <= 83:
                tracks = [person(track_id=2, team='B'), ball()]
            else:
                tracks = [person()]
            result = self.update(frame, tracks)
            if frame == 62:
                self.assertEqual(result['pending_candidate']['track_id'], 2)
                self.assertIsNone(result['owner'])
            if frame == 63:
                self.assertEqual(result['owner']['team'], 'B')
                self.assertEqual(result['timing']['credited_player_seconds'], 0.)
        summary = self.analytics.summary()
        self.assertAlmostEqual(summary['analyzed_duration_seconds'], 10.)
        self.assertAlmostEqual(summary['confirmed_player_duration_seconds'], 8.)
        self.assertAlmostEqual(summary['known_team_duration_seconds'], 8.)
        self.assertAlmostEqual(summary['unknown_player_duration_seconds'], 2.)
        self.assertAlmostEqual(summary['unknown_team_duration_seconds'], 2.)
        self.assertAlmostEqual(summary['team_possession_coverage_pct'], 80.)
        self.assertEqual(summary['episodes'], 2)
        for index, duration, share in ((0, 6., 75.), (1, 2., 25.)):
            self.assertAlmostEqual(summary['teams'][index]['possession_pct'], share)
            self.assertAlmostEqual(summary['teams'][index]['duration_seconds'], duration)
            self.assertAlmostEqual(summary['tracks'][index]['duration_seconds'], duration)
            self.assertEqual(summary['tracks'][index]['episodes'], 1)

    def test_cut_resets_pending_evidence_and_keeps_reused_ids_separate(self):
        tracks = [person(), ball()]
        for index in range(3):
            self.update(index, tracks)
        cut = self.update(3, tracks, segment=1, cut=True)
        self.assertEqual(cut['state'], 'pending')
        self.assertEqual(cut['pending_candidate']['segment'], 1)
        self.assertEqual(cut['timing']['credited_player_seconds'], 0.)
        self.assertEqual(cut['timing']['elapsed_seconds'], .1)
        self.update(4, tracks, segment=1)
        self.update(5, tracks, segment=1)
        result = self.analytics.summary()
        self.assertEqual([t['duration_seconds'] for t in result['tracks']], [.1, .1])
        self.assertEqual([t['status'] for t in result['tracks']], ['inactive', 'confirmed'])
        self.assertEqual([t['track_id'] for t in result['tracks']], [1, 1])

    def test_timestamp_clock_and_nonconsecutive_frames(self):
        tracks = [person(), ball()]
        for index, stamp in enumerate((0., .06, .11, .21, .21, .4, .5, .6)):
            result = self.update(index, tracks, timestamp=stamp)
            if index == 1:
                self.assertEqual(result['state'], 'pending')
            if index == 2:
                self.assertEqual(result['state'], 'confirmed')
                self.assertEqual(result['timing']['source'], 'video_timestamp')
            if index == 4:
                self.assertEqual(result['reason'], 'invalid_timing')
            if index in (4, 5):
                self.assertIsNone(result['timing']['elapsed_seconds'])
        self.assertAlmostEqual(result['totals']['confirmed_player_duration_seconds'], .2)
        total = result['totals']['analyzed_duration_seconds']
        duplicate = self.update(7, tracks, timestamp=.6)
        skipped = self.update(9, tracks, timestamp=.9)
        self.assertEqual(duplicate['reason'], 'invalid_timing')
        self.assertEqual(skipped['state'], 'pending')
        self.assertEqual(skipped['totals']['analyzed_duration_seconds'], total)

    def test_invalid_timing_and_config_and_json_nulls(self):
        for fps in (0, -1, math.inf, math.nan, True):
            with self.assertRaises(ValueError):
                PossessionAnalytics(fps, ('A', 'B'))
        for overrides in ({'max_normalized_distance': 0.}, {'min_distance_margin': -.1},
                          {'min_distance_margin': .6}, {'confirmation_seconds': math.nan},
                          {'hysteresis_seconds': -.1}, {'max_relative_speed': True},
                          {'max_observation_gap_seconds': 0.}):
            with self.assertRaises(ValueError):
                PossessionConfig(**overrides)
        for index, timestamp in enumerate((-1., math.nan, math.inf)):
            result = self.update(index, [person(), ball()], timestamp=timestamp)
            self.assertEqual(result['reason'], 'invalid_timing')
        decoded = json.loads(json.dumps(result, allow_nan=False))
        self.assertIsNone(decoded['owner'])
        self.assertIsNone(decoded['totals']['team_possession_coverage_pct'])
        self.assertEqual([t['possession_pct'] for t in decoded['totals']['teams']], [None, None])
