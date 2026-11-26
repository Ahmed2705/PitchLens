"""Exercise the annotation review without requiring local dataset media."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts import review_candidate_annotations as tool


class AnnotationReviewTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.data, self.work = root / "candidate", root / "corrections"
        sources = [
            ("val", "Full-Match-PERSIJA_Jakarta_VS_PSS_Sleman", 10020, 20),
            ("train", "2015-02-21-18-00Chelsea1-1Burnley1_720p", 100, 58),
            ("train", "2015-02-21-18-00Chelsea1-1Burnley2_720p", 100, 55),
        ]
        rows = []
        self.original = b"3 0.5 0.5 0.01 0.02\n"
        for split, prefix, start, count in sources:
            for name in ("images", "labels"):
                (self.data / split / name).mkdir(parents=True, exist_ok=True)
            for number in range(start, start + count):
                key = f"{prefix}_frame{number:04d}"
                media = key.encode()  # HTTP/browser integration checks use actual JPEGs.
                digest = sha256(media).hexdigest()
                image = f"{split}/images/{digest}.jpg"
                (self.data / image).write_bytes(media)
                (self.data / f"{split}/labels/{digest}.txt").write_bytes(self.original)
                rows.append({"frame_id": key, "split": split, "output": image, "sha256": digest,
                             "label_sha256": sha256(self.original).hexdigest()})
        # A protected row without a readable path must never be opened.
        rows.append({"frame_id": rows[0]["frame_id"], "split": "test", "output": "test/protected.jpg"})
        manifest = json.dumps(list(reversed(rows))).encode()
        (self.data / "manifest.json").write_bytes(manifest)
        patcher = patch.object(tool, "MANIFEST_SHA256", sha256(manifest).hexdigest())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.review = tool.Review(self.data, self.work)

    def test_load_prioritizes_development_and_separates_flags(self):
        frames = list(self.review.frames.values())
        self.assertEqual([frame["group"] for frame in frames], [0] * 20 + [1] * 58 + [2] * 55)
        self.assertEqual(self.review.state["cursor"], frames[0]["id"])
        self.assertEqual(frames[0]["flags"][0][:2], (376, 250))
        self.assertEqual(frames[0]["boxes"], [[3, 316.8, 313.6, 323.2, 326.4]])
        self.assertFalse(self.work.exists())

    def test_only_reviewed_boxes_export_and_progress_survives_restart(self):
        first, second, third = list(self.review.frames)[:3]
        state = {**self.review.state, "cursor": second, "drafts": {
            first: {"status": "reviewed", "boxes": [[3, 0, 0, 4, 6]]},
            second: {"status": "uncertain", "boxes": []},
            third: {"status": "reviewed", "boxes": self.review.frames[third]["boxes"]},
        }}
        self.assertEqual(self.review.save(state), 1)
        resumed = tool.Review(self.data, self.work)
        self.assertEqual(resumed.state["cursor"], second)
        self.assertEqual(resumed.state["drafts"], state["drafts"])
        first_export = resumed.export()
        self.assertNotEqual(first_export, resumed.export())
        first_label = self.review.frames[first]["label_path"]
        self.assertEqual(tool.parse_yolo((first_export / first_label).read_bytes()), [[3, 0, 0, 4, 6]])
        self.assertEqual((self.data / first_label).read_bytes(), self.original)
        self.assertFalse((first_export / self.review.frames[second]["label_path"]).exists())
        self.assertEqual((first_export / self.review.frames[third]["label_path"]).read_bytes(), self.original)
        summary = json.loads((first_export / "review.json").read_bytes())
        self.assertEqual([record["status"] for record in summary["frames"][:4]],
                         ["reviewed", "uncertain", "reviewed", "unreviewed"])

    def test_rejects_invalid_or_stale_saves_and_dataset_output(self):
        with self.assertRaisesRegex(ValueError, "separate"):
            tool.Review(self.data, self.data / "corrections")
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.review.export()
        state = self.review.state
        cursor = state["cursor"]
        for box in ([4, 0, 0, 3, 4], [1, 3, 0, 2, 4], [1, 0, 0, 2, 641], [1, 0, 0, float("nan"), 4]):
            with self.subTest(box=box), self.assertRaisesRegex(ValueError, "Invalid frame"):
                self.review.save({**state, "drafts": {cursor: {"status": "reviewed", "boxes": [box]}}})
        self.assertFalse(self.review.progress.exists())
        self.review.save(state)
        with self.assertRaisesRegex(ValueError, "another tab"):
            self.review.save(state)
