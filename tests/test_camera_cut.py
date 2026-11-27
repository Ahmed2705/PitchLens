"""Exercise shot detection and coordinated resets through the video runner."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from pitchlens.calibration import PitchConfig, project
from pitchlens.camera_cut import is_hard_cut, thumbnail
from pitchlens.detector import Detection
from pitchlens.pitch_lines import STRAIGHT_MARKINGS
from pitchlens.tracking import Tracker, TrackingConfig
from scripts import track_video


CUT_THRESHOLDS = {"appearance_threshold": .055, "low_detail_threshold": .32}


def scene(seed):
    rng = np.random.default_rng(seed)
    return cv2.resize(rng.integers(25, 205, (16, 28, 3), dtype=np.uint8), (320, 180))


class CameraCutTests(unittest.TestCase):
    def test_cuts_between_scenes_with_similar_brightness_and_a_shared_scorebug(self):
        before, after = scene(13), scene(27)
        after[:30, :100] = before[:30, :100]
        a, b = map(thumbnail, (before, after))
        legacy_difference = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                                        cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)).mean() / 255
        self.assertLess(legacy_difference, .32)
        self.assertTrue(is_hard_cut(a, b, **CUT_THRESHOLDS))

    def test_fast_pans_and_zoom_preserve_the_shot(self):
        before = scene(43)
        a = thumbnail(before)
        transforms = [
            np.array([[1., 0., 18.], [0., 1., -8.]]),
            cv2.getRotationMatrix2D((160, 90), 0., 1.2),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                after = cv2.warpAffine(before, transform, (320, 180), borderMode=cv2.BORDER_REFLECT)
                b = thumbnail(after)
                self.assertGreater(cv2.absdiff(a, b).mean() / 255, .055)
                self.assertFalse(is_hard_cut(a, b, **CUT_THRESHOLDS))

    def test_pan_with_straight_pitch_lines_and_no_corners_preserves_the_shot(self):
        before = np.full((90, 160, 3), (35, 145, 35), np.uint8)
        before[:26] = 20
        before[49:52] = 235
        shift = np.array([[1., 0., 0.], [0., 1., 10.]])
        after = cv2.warpAffine(before, shift, (160, 90), borderMode=cv2.BORDER_REPLICATE)
        self.assertGreater(cv2.absdiff(before, after).mean() / 255, .055)
        tracker = Tracker(30)
        first = tracker.update(before, [Detection((30, 30, 50, 65), 0, .9)])
        second = tracker.update(after, [Detection((30, 40, 50, 75), 0, .9)])
        self.assertFalse(second.camera_cut)
        self.assertEqual(first.tracks[0].track_id, second.tracks[0].track_id)

    def test_pan_with_detail_at_the_edge_preserves_the_shot_in_both_directions(self):
        before = np.full((90, 160, 3), (35, 145, 35), np.uint8)
        before[:16] = 235
        for dy in (14., 13.25, 13.5, 13.75):
            with self.subTest(dy=dy):
                after = cv2.warpAffine(before, np.array([[1., 0., 0.], [0., 1., dy]]),
                                       (160, 90), borderMode=cv2.BORDER_REPLICATE)
                self.assertGreater(cv2.absdiff(before, after).mean() / 255, .055)
                for a, b in ((before, after), (after, before)):
                    self.assertFalse(is_hard_cut(a, b, **CUT_THRESHOLDS))

    def test_lower_third_appearing_and_disappearing_preserves_the_shot(self):
        before = thumbnail(scene(43))
        for scorebug in (False, True):
            with self.subTest(scorebug=scorebug):
                after = before.copy()
                after[72:, 5:-5] = 20
                if scorebug:
                    after[:9, :16] = 230
                self.assertGreater(cv2.absdiff(before, after).mean() / 255, .055)
                for a, b in ((before, after), (after, before)):
                    self.assertFalse(is_hard_cut(a, b, **CUT_THRESHOLDS))

    def test_mowing_stripes_do_not_hide_a_cut_with_different_scene_content(self):
        before = np.full((90, 160, 3), (35, 145, 35), np.uint8)
        for x in range(0, 160, 16):
            before[:, x:x+8] = (38, 163, 38)
        textured = before.copy()
        for y in range(13, 90, 18):
            for x in range(8, 160, 16):
                cv2.circle(textured, (x, y), 2, (35, 180, 35), -1)
        for image in (before, textured):
            first, second = image.copy(), image.copy()
            first[:27, :64] = 220
            second[63:, 96:] = 220
            for a, b in ((first, second), (second, first)):
                self.assertTrue(is_hard_cut(a, b, **CUT_THRESHOLDS))

    def test_refinement_keeps_conflicting_features_at_integer_overlap_boundary(self):
        before = np.full((90, 160, 3), (35, 145, 35), np.uint8)
        before[:55] = 235
        after = cv2.warpAffine(before, np.array([[1., 0., 0.], [0., 1., 9.]]),
                               (160, 90), borderMode=cv2.BORDER_REPLICATE)
        after[9, 20:140] = 0
        for a, b in ((before, after), (after, before)):
            self.assertTrue(is_hard_cut(a, b, **CUT_THRESHOLDS))

    def test_matching_field_line_does_not_hide_conflicting_pitch_details(self):
        before = np.full((90, 160, 3), (35, 145, 35), np.uint8)
        after = np.full_like(before, (15, 140, 65))
        before[:7] = 235
        after[:17] = 235
        after[:9] = (80, 25, 40)
        before[:5, ::2] = 100
        after[10:15, 1::2] = 100
        before[50:58, 40:46] = (0, 0, 210)
        after[62:70, 105:111] = (210, 0, 0)
        for a, b in ((before, after), (after, before)):
            self.assertTrue(is_hard_cut(a, b, **CUT_THRESHOLDS))

    def test_brighter_exposure_with_same_structure_stays_in_shot(self):
        before = scene(13)
        after = cv2.add(before, (48, 48, 48, 0))
        self.assertFalse(is_hard_cut(thumbnail(before), thumbnail(after), **CUT_THRESHOLDS))

    def test_cut_configuration_changes_sensitivity_and_can_disable_resets(self):
        before, after = scene(13), scene(27)
        for config, expected in (
            (TrackingConfig(), True),
            (TrackingConfig(cut_appearance_threshold=.32), False),
            (TrackingConfig(cut_threshold=1.), False),
        ):
            with self.subTest(config=config):
                tracker = Tracker(30, config)
                tracker.update(before, [])
                self.assertEqual(tracker.update(after, []).camera_cut, expected)
        tracker = Tracker(30, TrackingConfig(cut_appearance_threshold=1., cut_threshold=.1))
        tracker.update(np.full_like(before, 40), [])
        self.assertTrue(tracker.update(np.full_like(before, 80), []).camera_cut)
        with self.assertRaises(ValueError):
            TrackingConfig(cut_appearance_threshold=0.)

    def test_runner_resets_ball_people_team_and_homography_at_each_cut(self):
        before, after = scene(13), scene(27)
        world = PitchConfig().landmarks()
        camera = np.array([[2., 0., 30.], [0., 2., 20.], [0., 0., 1.]])
        projected_before = project(camera, world)
        projected_after = projected_before + [1., 0.]
        # Keep the different seeded textures while giving the calibrator
        # corroborating paint and green shoulders in both shots.
        for frame, projected in ((before, projected_before), (after, projected_after)):
            frame[..., 0] //= 2
            frame[..., 2] //= 2
            frame[..., 1] = np.maximum(frame[..., 1], 115)
            for ids in STRAIGHT_MARKINGS.values():
                start, end = projected[np.array(ids) - 1].round().astype(int)
                cv2.line(frame, start, end, (230, 230, 230), 1)
        players = []
        for index in range(6):
            x = index * 45 + 5
            before[30:100, x:x+27] = (255, 0, 0) if index < 3 else (255, 255, 255)
            players.append(Detection((x, 30., x+27, 100.), 0, .9))
        after[30:100, 5:32] = (255, 0, 0)
        keeper = Detection((286., 30., 310., 95.), 1, .9)
        ball = Detection((120., 111., 127., 118.), 3, .9)
        next_ball = Detection((270., 130., 277., 137.), 3, .9)
        detections = [players + [keeper, ball], players + [keeper, ball],
                      [players[0], keeper], [players[0], next_ball], [players[0]]]
        detector = Mock(device="cpu", image_size=960)
        detector.detect.side_effect = detections
        capture = Mock()
        capture.get.side_effect = {cv2.CAP_PROP_FPS: 30., cv2.CAP_PROP_FRAME_WIDTH: 320.,
                                   cv2.CAP_PROP_FRAME_HEIGHT: 180., cv2.CAP_PROP_POS_MSEC: 0.}.__getitem__
        frames = [before, before, after, after, before]
        capture.read.side_effect = [(True, frame.copy()) for frame in frames] + [(False, None)]

        points_before = np.column_stack((projected_before, np.full(len(world), .95)))
        points_after = np.column_stack((projected_after, np.full(len(world), .95)))
        landmark_detector = Mock()
        landmark_detector.detect.side_effect = [points_before, points_before, points_after, points_after, None]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "comparison.mp4"
            arguments = ["track_video.py", "--input", "synthetic.mp4", "--output", str(output),
                         "--weights", "fixture.pt", "--pitch-weights", "pitch-fixture.pt", "--team-mode", "color"]
            with (patch.object(sys, "argv", arguments),
                  patch.object(track_video, "Detector", return_value=detector),
                  patch.object(cv2, "VideoCapture", return_value=capture),
                  patch.object(cv2, "VideoWriter", return_value=Mock()),
                  patch("pitchlens.pitch_landmarks.PitchLandmarkDetector", return_value=landmark_detector),
                  patch("builtins.print")):
                track_video.main()
            rows = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
            summary = json.loads(output.with_suffix(".json").read_text())

        self.assertEqual([row["camera_cut"] for row in rows], [False, False, True, False, True])
        self.assertEqual([row["segment"] for row in rows], [0, 0, 1, 1, 2])
        first, first_cut, after_cut, second_cut = (rows[index] for index in (1, 2, 3, 4))
        self.assertTrue(all(item["team"] in (1, 2) for item in first["tracks"] if item["class_id"] == 0))
        self.assertTrue(first["calibration"]["stabilized"])
        self.assertEqual([len(row["ball_trajectory"]) for row in rows], [1, 2, 0, 1, 0])
        self.assertEqual(after_cut["ball_trajectory"], [[273.5, 133.5, True]])
        old_ids = {item["track_id"] for item in first["tracks"] if item["class_id"] != 3}
        new_ids = {item["track_id"] for item in first_cut["tracks"]}
        last_ids = {item["track_id"] for item in second_cut["tracks"]}
        self.assertEqual([item["class_id"] for item in first_cut["tracks"]], [0, 1])
        self.assertTrue(old_ids.isdisjoint(new_ids))
        self.assertTrue((old_ids | new_ids).isdisjoint(last_ids))
        self.assertIsNone(first_cut["tracks"][0]["team"])
        self.assertTrue(first_cut["calibration"]["valid"])
        self.assertFalse(first_cut["calibration"]["stabilized"])
        self.assertTrue(after_cut["calibration"]["stabilized"])
        self.assertFalse(second_cut["calibration"]["valid"])
        self.assertEqual(second_cut["tracks"][0]["pitch"], {"status": "unavailable", "xy_m": None})
        self.assertEqual(first["movement_timing"]["source"], "frame_index_fps")
        self.assertTrue(any(item["movement"]["accepted_segments"] == 1
                            for item in first["tracks"] if item["class_id"] in (0, 1)))
        for cut in (first_cut, second_cut):
            self.assertIsNone(cut["possession"]["owner"])
            self.assertAlmostEqual(cut["possession"]["timing"]["elapsed_seconds"], 1 / 30)
            self.assertEqual(cut["possession"]["timing"]["credited_player_seconds"], 0.)
            self.assertIsNone(cut["movement_timing"]["elapsed_seconds"])
            for item in cut["tracks"]:
                self.assertIsNone(item["movement"]["speed_kmh"])
                self.assertIsNone(item["movement"]["observed_distance_m"])
        self.assertEqual({item["segment"] for item in summary["movement"]["tracks"]}, {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
