"""Synthetic behavior tests; these do not measure real-world accuracy."""
import unittest
import numpy as np
from pitchlens.detector import Detection
from pitchlens.tracking import Tracker, TrackingConfig


def det(x, cls=0, score=.9):
    return Detection((x, 20., x + (5 if cls == 3 else 20), 25. if cls == 3 else 60.), cls, score)


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 200, 3), np.uint8)
        self.tracker = Tracker(10)

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
        p = self.tracker.update(self.frame, [])
        self.assertFalse(p.tracks[0].observed)
        self.assertIsNone(p.tracks[0].confidence)
        self.tracker.update(self.frame, [])
        self.assertTrue(self.tracker.update(self.frame, [det(150, 3)]).tracks[0].observed)

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
        self.tracker.update(self.frame, [det(10), det(20, 3)])
        cut = self.tracker.update(np.full_like(self.frame, 255), [])
        self.assertTrue(cut.camera_cut)
        self.assertEqual(cut.segment, 1)
        self.assertEqual(cut.tracks, ())
        self.assertEqual(cut.ball_trajectory, ())
        self.tracker.reset()
        self.assertEqual(self.tracker.update(self.frame, []).frame_index, 0)

    def test_invalid_inputs(self):
        for fps in (0, float('nan'), -1, True):
            with self.assertRaises(ValueError): Tracker(fps)
        with self.assertRaises(ValueError): TrackingConfig(person_low=.5)
        with self.assertRaises(ValueError): self.tracker.update(self.frame, [det(199)])
        with self.assertRaises(ValueError): self.tracker.update(np.zeros((0, 2, 3), np.uint8), [])
        self.tracker.update(self.frame, [])
        with self.assertRaises(ValueError): self.tracker.update(np.zeros((10, 10, 3), np.uint8), [])


if __name__ == '__main__': unittest.main()
