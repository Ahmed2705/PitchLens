import unittest

import numpy as np

from pitchlens.calibration import PitchCalibrator, PitchConfig, project
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
