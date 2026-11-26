"""Serve the offline Detection Candidate v1 correction tool on localhost."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from threading import Lock

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "739efbbad75d9021100e24d188e1895689004a84cfa7e7dd3ad3bdaaab495a5b"
SIZE = 640
CLASSES = ("player", "goalkeeper", "referee", "ball")
STATUSES = ("unreviewed", "reviewed", "uncertain")
GROUPS = ("Persija · validation", "Chelsea · first half", "Chelsea · second half")

# Approximate pixel locations from the documented manual review. These markers
# are guidance only and never enter the editable or exported bounding boxes.
FLAGS = {
    (0, 10020): [(376, 250, "Check the padded ball box.")],
    (0, 10920): [(322, 435, "Check the unboxed player on the ground.")],
    (0, 10980): [(336, 435, "Check the unboxed player on the ground.")],
    (0, 12720): [(437, 363, "Check the unboxed goalkeeper."),
                 (543, 401, "Check the unboxed white-shirted player.")],
    (0, 12780): [(339, 514, "Check the overlapping white-shirted player.")],
    (0, 13020): [(460, 365, "Check the overlapping white-shirted player.")],
    (0, 13080): [(170, 429, "Check the overlapping white-shirted player."),
                 (538, 385, "Check the unboxed white-shirted player.")],
    (1, 21360): [(19, 172, "Check the unboxed Burnley player."),
                 (361, 516, "Check the padded ball box.")],
    (1, 30960): [(485, 377, "Check the padded ball box.")],
    (1, 46920): [(126, 355, "Check the two overlapping players.")],
    (2, 720): [(105, 407, "Check the two players sharing one box.")],
    (2, 29760): [(443, 430, "Check the ball box extent and motion blur.")],
}
GOAL_LINE = {12720: (500, 322), 12780: (510, 316), 13020: (435, 326),
             13080: (189, 278), 13140: (16, 179), 14040: (253, 207)}


def valid_box(box: list) -> bool:
    if not isinstance(box, list) or len(box) != 5 or type(box[0]) is not int:
        return False
    category, x1, y1, x2, y2 = box
    return (0 <= category < len(CLASSES)
            and all(type(v) in (int, float) for v in box[1:])
            and 0 <= x1 < x2 <= SIZE and 0 <= y1 < y2 <= SIZE)


def parse_yolo(label: bytes) -> list[list]:
    boxes = []
    for row in label.decode("utf-8").splitlines():
        category, x, y, width, height = row.split()
        x, y, width, height = map(float, (x, y, width, height))
        xyxy = [round(v * SIZE, 6) for v in
                (x - width / 2, y - height / 2, x + width / 2, y + height / 2)]
        if not all(-0.00001 <= v <= SIZE + 0.00001 for v in xyxy):
            raise ValueError("Source box extends outside the frame.")
        box = [int(category), *(min(SIZE, max(0, v)) for v in xyxy)]
        if not valid_box(box):
            raise ValueError("Invalid source box.")
        boxes.append(box)
    return boxes


def format_yolo(boxes: list[list]) -> bytes:
    rows = []
    for category, x1, y1, x2, y2 in boxes:
        x, y = (x1 + x2) / (2 * SIZE), (y1 + y2) / (2 * SIZE)
        width, height = (x2 - x1) / SIZE, (y2 - y1) / SIZE
        rows.append(f"{category} {x:.8f} {y:.8f} {width:.8f} {height:.8f}\n")
    return "".join(rows).encode("ascii")


class Review:
    def __init__(self, data: Path, work: Path) -> None:
        data, work = data.resolve(), work.resolve()
        if work.is_relative_to(data) or data.is_relative_to(work):
            raise ValueError("Choose a work directory separate from the candidate dataset.")
        manifest = (data / "manifest.json").read_bytes()
        if sha256(manifest).hexdigest() != MANIFEST_SHA256:
            raise ValueError("The manifest does not match Detection Candidate v1.")
        selected = []
        for row in json.loads(manifest):
            frame = row["frame_id"]
            persija = re.fullmatch(r"Full-Match-PERSIJA_Jakarta_VS_PSS_Sleman_frame(\d+)", frame)
            chelsea = re.fullmatch(r"2015-02-21-18-00Chelsea1-1Burnley([12])_720p_frame(\d+)", frame)
            if persija and row["split"] == "val":
                selected.append((0, int(persija[1]), row))
            elif chelsea and row["split"] == "train":
                selected.append((int(chelsea[1]), int(chelsea[2]), row))
        if Counter(group for group, _, _ in selected) != {0: 20, 1: 58, 2: 55}:
            raise ValueError("Expected 20 Persija and 113 Chelsea frames.")

        self.frames, self.images, self.labels = {}, {}, {}
        for group, number, row in sorted(selected, key=lambda item: item[:2]):
            split, digest = row["split"], row["sha256"]
            image_path = f"{split}/images/{digest}.jpg"
            if not re.fullmatch("[0-9a-f]{64}", digest) or row["output"] != image_path:
                raise ValueError("Unexpected image path in candidate manifest.")
            key = row["frame_id"]
            label_path = f"{split}/labels/{digest}.txt"
            image, label = (data / image_path).read_bytes(), (data / label_path).read_bytes()
            if sha256(image).hexdigest() != digest or sha256(label).hexdigest() != row["label_sha256"]:
                raise ValueError(f"File changed since candidate preparation: {key}")
            flags = list(FLAGS.get((group, number), ()))
            if group == 0 and number in GOAL_LINE:
                flags.append((*GOAL_LINE[number], "Verify the black-clad person's role using adjacent frames."))
            self.frames[key] = {"id": key, "number": f"{number:04d}", "group": group,
                                "label_path": label_path, "boxes": parse_yolo(label), "flags": flags}
            self.images[key], self.labels[key] = image, label

        self.work = work
        self.progress = work / "progress.json"
        self.state = {"manifest_sha256": MANIFEST_SHA256, "revision": 0,
                      "cursor": next(iter(self.frames)), "drafts": {}}
        if self.progress.exists():
            state = json.loads(self.progress.read_bytes())
            self.validate(state)
            self.state = state
        self.lock = Lock()

    def validate(self, state: dict) -> None:
        if (not isinstance(state, dict) or state.get("manifest_sha256") != MANIFEST_SHA256
                or type(state.get("revision")) is not int or state["revision"] < 0
                or not isinstance(state.get("cursor"), str) or state["cursor"] not in self.frames
                or not isinstance(state.get("drafts"), dict)):
            raise ValueError("Invalid progress file or wrong candidate manifest.")
        for key, draft in state["drafts"].items():
            if (key not in self.frames or not isinstance(draft, dict) or draft.get("status") not in STATUSES
                    or not isinstance(draft.get("boxes"), list) or len(draft["boxes"]) > 300
                    or not all(valid_box(box) for box in draft["boxes"])):
                raise ValueError(f"Invalid frame corrections for {key}.")

    def save(self, state: dict) -> int:
        self.validate(state)
        if state["revision"] != self.state["revision"]:
            raise ValueError("Progress changed in another tab. Reload before editing again.")
        state = {**state, "revision": state["revision"] + 1}
        self.work.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.work) as staging:
            snapshot = Path(staging) / "progress.json"
            snapshot.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
            snapshot.replace(self.progress)
        self.state = state
        return state["revision"]

    def export(self) -> Path:
        drafts = self.state["drafts"]
        statuses = {key: drafts.get(key, {}).get("status", "unreviewed") for key in self.frames}
        reviewed = [key for key, status in statuses.items() if status == "reviewed"]
        if not reviewed:
            raise ValueError("Mark at least one frame reviewed before exporting.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = self.work / f"export-{stamp}"
        # Stage beside the final export so the files inherit the workspace's
        # permissions on Windows and can be read outside the editor.
        payload = self.work / f".{destination.name}.tmp"
        payload.mkdir()
        try:
            records = []
            for key, frame in self.frames.items():
                record = {"frame_id": key, "status": statuses[key]}
                if statuses[key] == "reviewed":
                    path = payload / frame["label_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    boxes = drafts[key]["boxes"]
                    changed = boxes != frame["boxes"]
                    path.write_bytes(format_yolo(boxes) if changed else self.labels[key])
                    record.update(label_path=frame["label_path"], changed=changed)
                records.append(record)
            (payload / "review.json").write_text(json.dumps({
                "manifest_sha256": MANIFEST_SHA256, "progress_revision": self.state["revision"],
                "classes": CLASSES, "frames": records,
            }, indent=2) + "\n", encoding="utf-8")
            payload.rename(destination)
        except BaseException:
            shutil.rmtree(payload, ignore_errors=True)
            raise
        return destination


def serve(review: Review, port: int) -> ThreadingHTTPServer:
    assets = {
        "/": ("text/html; charset=utf-8", Path(__file__).with_name("review_candidate_annotations.html").read_bytes()),
        "/review.js": ("text/javascript; charset=utf-8", Path(__file__).with_name("review_candidate_annotations.js").read_bytes()),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def reply(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'unsafe-inline'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def reply_json(self, value: dict, status: int = 200) -> None:
            self.reply(json.dumps(value).encode(), "application/json", status)

        def local_host(self) -> bool:
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self) -> None:
            if not self.local_host():
                self.send_error(403)
            elif self.path in assets:
                content_type, body = assets[self.path]
                self.reply(body, content_type)
            elif self.path == "/state":
                with review.lock:
                    state = review.state
                self.reply_json({"frames": list(review.frames.values()), "state": state,
                                 "groups": GROUPS, "classes": CLASSES, "size": SIZE})
            elif self.path.startswith("/image/") and (key := self.path[7:]) in review.images:
                self.reply(review.images[key], "image/jpeg")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            origin = f"http://127.0.0.1:{self.server.server_port}"
            if not self.local_host() or self.headers.get("Origin") != origin:
                self.send_error(403)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get_content_type() != "application/json" or not 0 < length <= 4_000_000:
                    raise ValueError("Expected a JSON review request.")
                payload = json.loads(self.rfile.read(length))
                with review.lock:
                    if self.path == "/save":
                        result = {"revision": review.save(payload)}
                    elif self.path == "/export":
                        if payload != {"revision": review.state["revision"]}:
                            raise ValueError("Save current progress before exporting.")
                        result = {"directory": str(review.export())}
                    else:
                        self.send_error(404)
                        return
                self.reply_json(result)
            except ValueError as exc:
                self.reply_json({"error": str(exc)}, 400)
            except OSError as exc:
                self.reply_json({"error": f"Local file operation failed: {exc}"}, 500)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/processed/detection-candidate-v1")
    parser.add_argument("--work", type=Path, default=PROJECT / "runs/detection-candidate-v1-corrections")
    args = parser.parse_args()
    try:
        review = Review(args.data, args.work)
        with serve(review, 8765) as server:
            print("Open http://127.0.0.1:8765/ in your browser. Press Ctrl+C to close.", flush=True)
            server.serve_forever()
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
