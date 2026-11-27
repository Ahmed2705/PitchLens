"""Synthetic behavior tests; these do not measure real-world accuracy."""
import unittest
from unittest.mock import patch
import numpy as np
from pitchlens.detector import Detection
from pitchlens.tracking import Tracker, TrackingConfig
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker


def det(x, cls=0, score=.9):
    return Detection((x, 20., x + (5 if cls == 3 else 20), 25. if cls == 3 else 60.), cls, score)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 200, 3), np.uint8)
        self.tracker = Tracker(10)

    def test_default_botsort_uses_source_frames_without_reid(self):
        self.assertEqual(self.tracker.config.people_tracker, 'botsort')
        self.assertEqual(len(self.tracker.people), 3)
        for person_tracker in self.tracker.people:
            self.assertIsInstance(person_tracker, BOTSORT)
            self.assertEqual(person_tracker.gmc.method, 'sparseOptFlow')
            self.assertIsNone(person_tracker.encoder)
            self.assertFalse(person_tracker.args.fuse_score)
            self.assertEqual(person_tracker.args.track_buffer, 10)
            self.assertEqual((person_tracker.args.track_low_thresh, person_tracker.args.track_high_thresh,
                              person_tracker.args.new_track_thresh, person_tracker.args.match_thresh),
                             (.10, .25, .25, .80))
        gmc = self.tracker.people[0].gmc
        with patch.object(gmc, 'apply', wraps=gmc.apply) as motion:
            first = self.tracker.update(self.frame, [det(20)])
            recovery = self.tracker.update(self.frame, [det(20, score=.101)])
            empty = self.tracker.update(self.frame, [])
        self.assertEqual(first.tracks[0].track_id, recovery.tracks[0].track_id)
        self.assertAlmostEqual(recovery.tracks[0].confidence, .101)
        self.assertEqual(empty.tracks, ())
        self.assertEqual(motion.call_count, 3)
        for call in motion.call_args_list:
            self.assertIs(call.args[0], self.frame)
        self.assertEqual(motion.call_args.args[1].shape, (0, 4))

    def test_explicit_bytetrack_fallback_preserves_recovery(self):
        fallback = Tracker(10, TrackingConfig(people_tracker='bytetrack'))
        for person_tracker in fallback.people:
            self.assertIs(type(person_tracker), BYTETracker)
        first = fallback.update(self.frame, [det(20)])
        second = fallback.update(self.frame, [det(21, score=.2)])
        self.assertEqual(first.tracks[0].track_id, second.tracks[0].track_id)

    def test_interleaved_video_instances_keep_separate_state(self):
        other = Tracker(10)
        first = self.tracker.update(self.frame, [det(20)])
        other_frame = np.full_like(self.frame, 255)
        other_first = other.update(other_frame, [det(100)])
        self.assertEqual(first.tracks[0].track_id, 1)
        self.assertEqual(other_first.tracks[0].track_id, 1)
        self.assertIsNot(self.tracker.people[0].gmc, other.people[0].gmc)
        other.reset()
        continued = self.tracker.update(self.frame, [det(21, score=.2)])
        self.assertEqual(continued.frame_index, 1)
        self.assertEqual(continued.tracks[0].track_id, 1)
        self.assertEqual(other.update(other_frame, []).frame_index, 0)

    def test_people_stable_low_confidence_and_occlusion(self):
        first = self.tracker.update(self.frame, [det(20)]).tracks[0].track_id
        second = self.tracker.update(self.frame, [det(21, score=.2)]).tracks[0].track_id
        self.assertEqual(first, second)
        self.assertEqual(self.tracker.update(self.frame, []).tracks, ())
        recovered = self.tracker.update(self.frame, [det(22)]).tracks[0].track_id
        self.assertEqual(first, recovered)

    def test_two_people_and_class_scoping(self):
        a = self.tracker.update(self.frame, [det(10), det(100), det(50, 1)])
        b = self.tracker.update(self.frame, [det(101), det(11), det(51, 1)])
        for cls in (0, 1):
            self.assertEqual({x.track_id for x in a.tracks if x.class_id == cls}, {x.track_id for x in b.tracks if x.class_id == cls})

    def test_ball_gating_prediction_expiry_and_reacquisition(self):
        self.tracker.update(self.frame, [det(20, 3)])
        b = self.tracker.update(self.frame, [det(22, 3, .6), det(150, 3, .99)])
        self.assertLess(b.tracks[0].xyxy[0], 30)
        self.assertFalse(b.ball_observation_ambiguous)
        p = self.tracker.update(self.frame, [])
        self.assertFalse(p.tracks[0].observed)
        self.assertIsNone(p.tracks[0].confidence)
        self.tracker.update(self.frame, [])
        self.assertTrue(self.tracker.update(self.frame, [det(150, 3)]).tracks[0].observed)

    def test_ball_competition_is_reported_without_changing_tracker_choice(self):
        first = self.tracker.update(self.frame, [det(20, 3), det(120, 3, .7)])
        self.assertTrue(first.ball_observation_ambiguous)
        self.assertTrue(first.tracks[0].observed)
        self.assertEqual(first.tracks[0].xyxy[0], 20.)
        second = self.tracker.update(self.frame, [det(21, 3), det(26, 3, .8)])
        self.assertTrue(second.ball_observation_ambiguous)
        self.assertEqual(second.tracks[0].xyxy[0], 21.)
        self.assertFalse(self.tracker.update(self.frame, [det(22, 3)]).ball_observation_ambiguous)

    def test_person_boxes_are_clipped_at_image_edge(self):
        for x in (10, 7, 4, 1, 0, 0):
            result = self.tracker.update(self.frame, [det(x)])
        self.assertGreaterEqual(result.tracks[0].xyxy[0], 0)

    def test_late_low_confidence_person_is_confirmed(self):
        self.tracker.update(self.frame, [])
        self.tracker.update(self.frame, [det(20, score=.28)])
        self.assertEqual(len(self.tracker.update(self.frame, [det(20, score=.28)]).tracks), 1)

    def test_ball_velocity_cleared_at_boundary(self):
        self.tracker.update(self.frame, [det(185, 3)])
        self.tracker.update(self.frame, [det(194, 3)])
        self.tracker.update(self.frame, [])
        self.tracker.update(self.frame, [det(40, 3)])
        p = self.tracker.update(self.frame, []).tracks[0]
        self.assertAlmostEqual(p.xyxy[0], 40)

    def test_cut_and_video_reset(self):
        first = self.tracker.update(self.frame, [det(10), det(20, 3)])
        previous_trackers = self.tracker.people
        cut = self.tracker.update(np.full_like(self.frame, 255), [det(10)])
        self.assertTrue(cut.camera_cut)
        self.assertEqual(cut.segment, 1)
        self.assertEqual(len(cut.tracks), 1)
        self.assertGreater(cut.tracks[0].track_id, first.tracks[0].track_id)
        self.assertEqual(cut.ball_trajectory, ())
        for previous, current in zip(previous_trackers, self.tracker.people, strict=True):
            self.assertIsNot(previous, current)
            self.assertIsNot(previous.gmc, current.gmc)
            self.assertEqual(current.frame_id, 1)
        self.tracker.reset()
        self.assertEqual(self.tracker.update(self.frame, []).frame_index, 0)

    def test_invalid_inputs(self):
        for fps in (0, float('nan'), -1, True):
            with self.assertRaises(ValueError): Tracker(fps)
        for invalid in ('strongsort', True, None, ['botsort']):
            with self.subTest(people_tracker=invalid), self.assertRaisesRegex(ValueError, 'people_tracker'):
                TrackingConfig(people_tracker=invalid)
        with self.assertRaises(ValueError): TrackingConfig(person_low=.5)
        with self.assertRaises(ValueError): self.tracker.update(self.frame, [det(199)])
        with self.assertRaises(ValueError): self.tracker.update(np.zeros((0, 2, 3), np.uint8), [])
        self.tracker.update(self.frame, [])
        with self.assertRaises(ValueError): self.tracker.update(np.zeros((10, 10, 3), np.uint8), [])


if __name__ == '__main__': unittest.main()
