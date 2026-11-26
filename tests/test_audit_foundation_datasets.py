from collections import Counter, defaultdict
from io import BytesIO
import unittest
from zipfile import ZipFile

from scripts.audit_foundation_datasets import archive_members, soccer_annotations, source_keys, yolo_annotations


class ArchiveTests(unittest.TestCase):
    def archive(self, *entries):
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            for name, data in entries:
                archive.writestr(name, data)
        return ZipFile(buffer)

    def test_normal_nested_file(self):
        with self.archive(("valid/", ""), ("valid/00000.jpg", b"image")) as archive:
            self.assertEqual(list(archive_members(archive)), ["valid/00000.jpg"])

    def test_rejects_unsafe_or_ambiguous_names(self):
        paths = ("../data", "C:/data", "aux.txt", "valid/name. ")
        for path in paths:
            with self.subTest(path=path), self.archive((path, b"data")) as archive:
                with self.assertRaises(ValueError):
                    archive_members(archive)
        with self.archive(("train/image.jpg", b"data")) as archive:
            raw = archive.fp.getvalue().replace(b"train/image.jpg", b"train\\image.jpg")
        with ZipFile(BytesIO(raw)) as archive, self.assertRaises(ValueError):
            archive_members(archive)
        for entries in ((("foo", b"one"), ("FOO", b"two")), (("foo", b"one"), ("foo/bar", b"two"))):
            with self.subTest(entries=entries), self.archive(*entries) as archive:
                with self.assertRaises(ValueError):
                    archive_members(archive)


class AnnotationTests(unittest.TestCase):
    def test_source_keys_preserve_clip_and_group(self):
        self.assertEqual(source_keys("42ba34_3_5_png_jpg.rf.123"), ("42ba34_3_5", "42ba34_3", "42ba34"))
        self.assertEqual(source_keys("Match2_720p_frame0020_png"), ("Match2_720p_frame0020", "Match2_720p", "Match"))

    def test_box_and_pose_visibility_are_checked(self):
        stats = {"ball_sizes_at_640": [], "visible_per_row": [], "visibility_by_index": defaultdict(Counter)}
        objects, issues = yolo_annotations(
            "0 .5 .5 .006 .007\n0 .005 .5 .02 .02\n0 nan .5 .02 .02",
            ["soccer ball"], False, 640, 640, stats,
        )
        self.assertEqual([obj[:2] for obj in objects], [("ball", "box"), ("ball", "box")])
        self.assertEqual(issues, {"box_over_1px": 1, "invalid_class_or_nonfinite": 1})
        pose = "0 .5 .5 1 1 .25 .5 0 .5 .5 2 " + "0 0 0 " * 30
        _, issues = yolo_annotations(pose, ["pitch"], True, 960, 960, stats)
        self.assertEqual(issues, {"unlabeled_point_with_coordinates": 1})
        self.assertEqual(stats["visible_per_row"], [1])

    def test_polyline_problem_counts(self):
        _, issues = soccer_annotations('{"Middle line":[{"x":1.1,"y":0}],'
                                      '"Side line left":[{"x":0,"y":0},{"x":0,"y":0}]}')
        self.assertEqual(issues, {"fewer_than_two_points": 1, "repeated_polyline_point": 1,
                                  "polyline_point_outside_image": 1})


if __name__ == "__main__":
    unittest.main()
