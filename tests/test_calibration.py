import unittest

import cv2
import numpy as np

from pitchlens.calibration import PitchCalibrator, PitchConfig, project
from pitchlens.pitch_lines import STRAIGHT_MARKINGS
from pitchlens.pitch_minimap import PitchMinimap


class PitchCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = PitchConfig()
        self.calibrator = PitchCalibrator(self.config)
        self.world = self.config.landmarks()
        self.camera = np.array([[8, 0.7, 100], [0.3, 6, 70], [0.001, 0.002, 1]])
        self.shape = (720, 1280, 3)

    def observe(self, camera=None):
        return np.column_stack((project(self.camera if camera is None else camera, self.world),
                                np.full(len(self.world), 0.95)))

    def painted_frame(self, camera=None, shape=None, faint_marking=None):
        frame = np.full(self.shape if shape is None else shape, (45, 110, 55), dtype=np.uint8)
        for name, ids in STRAIGHT_MARKINGS.items():
            points = project(self.camera if camera is None else camera, self.world[np.array(ids) - 1])
            start, end = np.rint(points).astype(int)
            color = (52, 115, 62) if name == faint_marking else (220, 225, 220)
            cv2.line(frame, start, end, color, 1 if faint_marking else 2)
        return frame

    def test_concentrated_landmarks_reject_an_exact_fit(self):
        camera = np.array([[3., 0., 400.], [0., 3., 200.], [0., 0., 1.]])
        points = self.observe(camera)
        points[self.world[:, 0] > 16.5, 2] = 0
        baseline = PitchCalibrator(self.config, legacy_validation=True).update(points, self.shape)
        self.assertTrue(baseline.metadata()['valid'])
        self.assertLess(baseline.reprojection_rmse_px, 0.01)
        result = self.calibrator.update(points, self.shape)
        self.assertEqual(result.reason, 'concentrated_support')
        self.assertEqual(result.map_contact((410, 230, 420, 250)), {'xy_m': None, 'status': 'unavailable'})

    def test_low_residual_fit_with_high_leverage_is_rejected(self):
        camera = self.camera.copy()
        camera[:2, :2] *= 1.4
        points = self.observe(camera)
        ids = [1, 3, 7, 21, 23, 26, 30]
        points[:, 2] = 0
        points[ids, 2] = 0.95
        points[ids, :2] += [[-3.64, 0.81], [-1.15, 0.8], [0.6, 0.36], [1.84, 1.11],
                            [0.94, -1.79], [3.45, 0.25], [2.21, 0.65]]
        baseline = PitchCalibrator(self.config, legacy_validation=True).update(points, self.shape)
        self.assertLess(baseline.reprojection_rmse_px, 1)
        self.assertEqual(self.calibrator.update(points, self.shape).reason, 'unstable_support')

    def test_paint_rejects_consistently_shifted_landmarks_and_clears_history(self):
        frame = self.painted_frame()
        points = self.observe()
        self.assertTrue(self.calibrator.update(points, self.shape, frame=frame).metadata()['valid'])
        self.assertTrue(self.calibrator.update(points, self.shape, frame=frame).stabilized)
        shifted = points.copy()
        shifted[:, :2] += [9, 10]
        baseline = PitchCalibrator(self.config, legacy_validation=True).update(shifted, self.shape)
        self.assertLess(baseline.reprojection_rmse_px, 0.01)
        result = self.calibrator.update(shifted, self.shape, frame=frame)
        self.assertEqual(result.reason, 'contradictory_paint')
        self.assertFalse(self.calibrator.update(points, self.shape, frame=frame).stabilized)
        self.assertFalse(self.calibrator.update(points, self.shape, camera_cut=True, frame=frame).stabilized)

    def test_frame_with_no_painted_evidence_is_unavailable(self):
        frame = np.full(self.shape, (40, 100, 50), dtype=np.uint8)
        result = self.calibrator.update(self.observe(), self.shape, frame=frame)
        self.assertEqual(result.reason, 'insufficient_paint')
        self.assertIsNone(result.image_to_pitch)

    def test_neighboring_paint_does_not_veto_exact_fit_with_faint_goal_area(self):
        camera = np.array([[7.5, 0., 100], [0., 0.75, 220], [0., 0., 1.]])
        frame = self.painted_frame(camera, (540, 960, 3), 'Small rect. left top')
        result = self.calibrator.update(self.observe(camera), frame.shape, frame=frame)
        self.assertTrue(result.metadata()['valid'], result.reason)

    def test_visible_continuation_explains_neighbor_even_when_its_paint_is_partial(self):
        camera = np.array([[10.5, 0., 100], [0., 0.75, 220], [0., 0., 1.]])
        frame = self.painted_frame(camera, (540, 960, 3), 'Small rect. left top')
        # Leave only the 74 px visible stretch of the neighboring penalty-box edge.
        frame[230, 101:273] = (45, 110, 55)
        cv2.line(frame, (118, 230), (192, 230), (220, 225, 220), 1)
        result = self.calibrator.update(self.observe(camera), frame.shape, frame=frame)
        self.assertTrue(result.metadata()['valid'], result.reason)

    def test_weak_neighbor_cannot_explain_coherently_shifted_paint(self):
        camera = np.array([[7.5, 0., 100], [0., 1., 220], [0., 0., 1.]])
        frame = np.full((540, 960, 3), (45, 110, 55), dtype=np.uint8)
        for name, ids in STRAIGHT_MARKINGS.items():
            points = project(camera, self.world[np.array(ids) - 1])
            if points[0, 0] == points[1, 0] or name in ('Small rect. left top', 'Small rect. right top'):
                cv2.line(frame, *np.rint(points).astype(int), (220, 225, 220), 1)
        self.assertTrue(self.calibrator.update(self.observe(camera), frame.shape, frame=frame).metadata()['valid'])
        shifted = self.observe(camera)
        shifted[:, 1] += 11
        self.assertEqual(self.calibrator.update(shifted, frame.shape, frame=frame).reason,
                         'contradictory_paint')
        # Detached paint raises both neighboring marks' support without
        # extending either shared ridge.
        cv2.line(frame, (166, 245), (223, 245), (220, 225, 220), 1)
        cv2.line(frame, (764, 245), (822, 245), (220, 225, 220), 1)
        self.assertEqual(self.calibrator.update(shifted, frame.shape, frame=frame).reason,
                         'contradictory_paint')

    def test_recovers_contact_despite_outlier(self):
        points = self.observe()
        points[8, :2] += 80
        points[5, 2] = 0.1
        calibration = self.calibrator.update(points, self.shape)
        self.assertTrue(calibration.metadata()['valid'])
        self.assertEqual(calibration.inliers, 30)
        self.assertLess(calibration.reprojection_rmse_px, 0.01)
        x, y = project(self.camera, np.array([[30., 22.]]))[0]
        actual = calibration.map_contact((x-5, y-30, x+5, y))
        self.assertEqual(actual['status'], 'valid')
        np.testing.assert_allclose(actual['xy_m'], [30, 22], atol=1e-4)

    def test_invalid_frame_clears_history_and_cut_disables_smoothing(self):
        points = self.observe()
        self.calibrator.update(points, self.shape)
        self.assertTrue(self.calibrator.update(points, self.shape).stabilized)
        self.assertFalse(self.calibrator.update(points, self.shape, camera_cut=True).stabilized)
        missing = self.calibrator.update(None, self.shape)
        self.assertFalse(missing.metadata()['valid'])
        self.assertEqual(missing.map_contact((100, 100, 110, 130))['status'], 'unavailable')
        self.assertFalse(self.calibrator.update(points, self.shape).stabilized)

    def test_follows_camera_pan_and_zoom(self):
        self.calibrator.update(self.observe(), self.shape)
        camera = self.camera.copy()
        camera[:2, :2] *= 1.15
        camera[0, 2] += 55
        result = self.calibrator.update(self.observe(camera), self.shape)
        self.assertFalse(result.stabilized)
        x, y = project(camera, np.array([[50, 30]]))[0]
        np.testing.assert_allclose(result.map_contact((x, y-20, x, y))['xy_m'], [50, 30], atol=1e-4)

    def test_support_limits_extrapolation_and_off_pitch_points(self):
        points = self.observe()
        points[self.world[:, 0] > 53, 2] = 0
        calibration = self.calibrator.update(points, self.shape)
        for world_x, expected in [(50, 'valid'), (56, 'estimated'), (100, 'unavailable'), (-1, 'unavailable')]:
            x, y = project(self.camera, np.array([[world_x, 34]]))[0]
            self.assertEqual(calibration.map_contact((x, y-20, x, y))['status'], expected)

    def test_collinear_and_nonfinite_landmarks_fail(self):
        points = self.observe()
        points[:, 1] = 200
        points[0, 0] = np.nan
        self.assertFalse(self.calibrator.update(points, self.shape).metadata()['valid'])
        points[1:, 2] = 0.1
        self.assertEqual(self.calibrator.update(points, self.shape).reason, 'too_few_landmarks')

    def test_dimensions_and_invalid_configuration(self):
        custom = PitchConfig(length_m=110, width_m=70).landmarks()
        np.testing.assert_array_equal(custom[29], [110, 70])
        np.testing.assert_array_equal(custom[9], [16.5, 14.84])
        for settings in ({'width_m': 30}, {'length_m': float('nan')}, {'min_inliers': 4}, {'smoothing': 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PitchConfig(**settings)

    def test_penalty_arc_ids_intersect_box_fronts_and_penalty_circles(self):
        for length, width in [(105, 68), (110, 70)]:
            with self.subTest(length=length, width=width):
                points = PitchConfig(length_m=length, width_m=width).landmarks()
                arcs = points[[10, 11, 18, 19]]
                np.testing.assert_allclose(arcs[:, 0], [16.5, 16.5, length - 16.5, length - 16.5])
                np.testing.assert_allclose(np.linalg.norm(arcs - points[[8, 8, 21, 21]], axis=1), 9.15)
                np.testing.assert_array_equal(arcs[:, 1] < width / 2, [True, False, True, False])
                np.testing.assert_allclose(arcs[[0, 2], 1] + arcs[[1, 3], 1], width)
                np.testing.assert_allclose(points[[6, 7, 22, 23], 1],
                                           [width / 2 - 9.16, width / 2 + 9.16] * 2)

    def test_landmark_order_matches_export_horizontal_flip(self):
        flip = [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15,
                16, 9, 10, 11, 12, 8, 6, 7, 0, 1, 2, 3, 4, 5, 31, 30]
        mirrored = self.world.copy()
        mirrored[:, 0] = self.config.length_m - mirrored[:, 0]
        np.testing.assert_allclose(mirrored, self.world[flip])

    def test_clipped_landmarks_and_ground_contact_are_unavailable(self):
        points = self.observe()
        points[:4, 0] = [0, 1280, 0, 1280]
        result = self.calibrator.update(points, self.shape)
        self.assertEqual(result.candidates, 28)
        camera = np.array([[8., 0., 200.], [0., 8., 300.], [0., 0., 1.]])
        result = self.calibrator.update(self.observe(camera), self.shape)
        self.assertIsNotNone(result.image_to_pitch)
        self.assertEqual(result.map_contact((490, 600, 510, 720))['status'], 'unavailable')

    def test_one_point_off_a_line_cannot_constrain_homography(self):
        points = self.observe()
        points[6:, 2] = 0
        points[13, 2] = 0.95
        result = self.calibrator.update(points, self.shape)
        self.assertEqual(result.reason, 'ambiguous_geometry')

    def test_exact_touchline_remains_within_pitch(self):
        result = self.calibrator.update(self.observe(), self.shape)
        x, y = project(self.camera, np.array([[60, 68]]))[0]
        mapped = result.map_contact((x-5, y-20, x+5, y))
        self.assertEqual(mapped['status'], 'valid')
        np.testing.assert_allclose(mapped['xy_m'], [60, 68], atol=1e-5)

    def test_minimap_suppresses_old_points_when_calibration_unavailable(self):
        minimap = PitchMinimap(self.config, self.shape)
        frame = np.zeros(self.shape, dtype=np.uint8)
        track = {'team': 1, 'pitch': {'xy_m': [30, 22], 'status': 'valid'}}
        minimap.draw(frame, [track], True)
        x, y = minimap._pixel(track['pitch']['xy_m'])
        h, w = minimap.background.shape[:2]
        pixel = (self.shape[0] - h + y, self.shape[1] - w + x)
        np.testing.assert_array_equal(frame[pixel], [255, 120, 0])
        minimap.draw(frame, [track], False)
        self.assertNotEqual(frame[pixel].tolist(), [255, 120, 0])
