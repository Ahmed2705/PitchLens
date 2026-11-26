"""Inventory the seven supplied archives without changing their contents."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PureWindowsPath
import random
import re
import shutil
import stat
from zipfile import ZipFile

import cv2
import numpy as np
import yaml


ARCHIVES = {
    "football-players-detection-v10": ("football-players-detection.v10i.yolo26.zip", "raw", False),
    "football-object-detection-v1": ("Football Object Detection.v1-new-test-version-1.0-augm..yolo26.zip", "raw", False),
    "wrkplace-football-detection": ("Football detection dataset.v1i.yolo26.zip", "raw", True),
    "ifap-football-objects": ("football-objects-detection.v5-2025-01-24-10-58am.yolo26.zip", "raw", True),
    "football-player-referee-goalkeeper": ("Football Player Referee and goalkeeper detection.v1i.yolo26.zip", "raw", True),
    "roboflow-field-detection-v18": ("football-field-detection.v18i.yolov8.zip", "calibration", True),
    "soccernet-calibration-2023-valid": ("valid.zip", "calibration", True),
}
NAMES = list(ARCHIVES)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ALIASES = {"soccer ball": "ball", "soccer-player": "player"}
BUDGET = 2_500_000_000
SOCCERNET_SHA = "9b8846782e16297a64a9f4c8e3b5bf32cfa62037dc735be79e168ada3aac0aa4"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def archive_members(archive):
    """Validate Windows paths and allow only identical repeated files."""
    members = {}
    seen = {}
    for entry in archive.infolist():
        raw = entry.orig_filename
        parts = raw.rstrip("/").split("/")
        if ("\\" in raw or any(
                not part or part in {".", ".."} or part.endswith((" ", "."))
                or (os.path.isreserved(part) if os.name == "nt" else PureWindowsPath(part).is_reserved())
                or re.search(r'[<>:"|?*\x00-\x1f]', part) for part in parts
        ) or stat.S_ISLNK(entry.external_attr >> 16) or entry.flag_bits & 1):
            raise ValueError(f"Unsafe ZIP member {raw!r}")
        key = raw.rstrip("/").casefold()
        if key in seen:
            previous = seen[key]
            if (previous.orig_filename != raw or entry.is_dir()
                    or archive.read(previous) != archive.read(entry)):
                raise ValueError(f"Ambiguous ZIP member {raw!r}")
            continue
        seen[key] = entry
        if not entry.is_dir():
            members[raw] = entry
    for key in seen:
        ancestors = PureWindowsPath(key).parents
        if any(str(parent).replace("\\", "/") in seen
               and not seen[str(parent).replace("\\", "/")].is_dir()
               for parent in ancestors):
            raise ValueError(f"Member has a file as parent: {key}")
    return members


def extract_archives(downloads, storage):
    for name, (filename, folder, new) in ARCHIVES.items():
        source = downloads / filename
        target = storage / "datasets" / folder / name
        if any(path.is_symlink() or path.is_junction() for path in (target, *target.parents)):
            raise ValueError(f"Storage path redirects through a symlink or junction: {target}")
        if name == NAMES[-1] and digest(source) != SOCCERNET_SHA:
            raise ValueError("SoccerNet archive differs from the publisher SHA-256")
        with ZipFile(source) as archive:
            members = archive_members(archive)
            if target.exists():
                actual = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
                if actual != members.keys():
                    raise ValueError(f"Existing dataset differs from archive: {target}")
                staging = target
            else:
                if not new:
                    raise ValueError(f"Expected original dataset is missing: {target}")
                staging = target.with_name(f".{name}.extracting")
                staging.mkdir(parents=True, exist_ok=False)
                required = sum(m.file_size for m in members.values())
                if shutil.disk_usage(staging).free < required:
                    raise ValueError(f"Insufficient free space for {target}")
                for relative, entry in members.items():
                    path = staging / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as reader, path.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
            for relative, entry in members.items():
                path = staging / relative
                if any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
                    raise ValueError(f"Extracted file path redirects: {path}")
                if digest(path) != hashlib.sha256(archive.read(entry)).hexdigest():
                    raise ValueError(f"Extracted file differs from archive: {path}")
            if staging != target:
                staging.rename(target)
            print(f"Verified {len(members)} files in {target}", flush=True)


def source_keys(stem):
    stem = re.sub(r"(?:_png|_jpg|_jpeg)+$", "", stem.split(".rf.")[0], flags=re.I)
    numbered = re.fullmatch(r"(.+?)_frame_?(\d+)", stem, flags=re.I)
    if numbered:
        clip = numbered[1]
        match = re.sub(r"[12]_720p$", "", clip)
        match = re.sub(r"_[12]$", "", match) if re.fullmatch(r"[a-f0-9]{8}_[12]", match) else match
        return stem, clip, match
    if re.fullmatch(r"[a-f0-9]{6}_\d+_\d+", stem):
        return stem, stem.rsplit("_", 1)[0], stem.split("_", 1)[0]
    return stem, "", ""


def yolo_annotations(raw, names, pose, width, height, stats):
    features = []
    issues = Counter()
    seen = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        try:
            cls, *values = map(float, tokens)
            if not all(map(math.isfinite, (cls, *values))) or cls != int(cls) or not 0 <= cls < len(names):
                raise ValueError
        except ValueError:
            issues["invalid_class_or_nonfinite"] += 1
            continue
        category = ALIASES.get(names[int(cls)], names[int(cls)])
        if tuple([cls, *values]) in seen:
            issues["duplicate_annotation_rows"] += 1
        seen.add(tuple([cls, *values]))
        if pose and len(values) != 100:
            issues["bad_pose_row_length"] += 1
            continue
        if not pose and len(values) >= 6 and len(values) % 2 == 0:
            pts = np.array(values).reshape(-1, 2)
            if (pts < 0).any() or (pts > 1).any():
                issues["polygon_outside"] += 1
            if abs(cv2.contourArea(pts.astype(np.float32))) == 0:
                issues["degenerate_polygon"] += 1
            features.append((category, "polygon", values))
            continue
        if not pose and len(values) != 4:
            issues["bad_detection_row_length"] += 1
            continue
        x, y, w, h = values[:4]
        if w <= 0 or h <= 0:
            issues["nonpositive_box"] += 1
        overflow = max((w/2-x)*width, (x+w/2-1)*width, (h/2-y)*height, (y+h/2-1)*height)
        if overflow > 1e-5:
            issues["box_over_1px" if overflow > 1.00001 else "box_within_1px"] += 1
        if category == "ball":
            stats["ball_sizes_at_640"].append(max(w*width, h*height)*640/max(width, height))
        if pose:
            points = np.array(values[4:]).reshape(-1, 3)
            stats["visible_per_row"].append(int((points[:, 2] == 2).sum()))
            for index, (px, py, vis) in enumerate(points):
                stats["visibility_by_index"][str(index+1)][str(vis)] += 1
                if vis not in (0, 1, 2):
                    issues["bad_visibility"] += 1
                if not (0 <= px <= 1 and 0 <= py <= 1):
                    issues["pose_point_outside_image"] += 1
                if vis == 0 and (px != 0 or py != 0):
                    issues["unlabeled_point_with_coordinates"] += 1
                if vis > 0 and not (x-w/2-0.001 <= px <= x+w/2+0.001
                                    and y-h/2-0.001 <= py <= y+h/2+0.001):
                    issues["labeled_point_outside_pose_box"] += 1
            features.append((category, "pose", values))
        else:
            features.append((category, "box", values))
    return features, issues


def soccer_annotations(raw):
    features = []
    issues = Counter()
    for category, points in json.loads(raw).items():
        coordinates = [(point["x"], point["y"]) for point in points]
        if len(coordinates) < 2:
            issues["fewer_than_two_points"] += 1
        if len(set(coordinates)) != len(coordinates):
            issues["repeated_polyline_point"] += 1
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in coordinates):
            issues["polyline_point_outside_image"] += 1
        features.append((category, "polyline", coordinates))
    return features, issues


def image_features(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError("Image did not decode")
    gray = cv2.cvtColor(cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(gray.astype(np.float32))[:8, :8].ravel()[1:]
    phash = int.from_bytes(np.packbits(dct > np.median(dct)).tobytes(), "big")
    return image, phash, float(gray.mean())


def quantiles(values):
    if not values:
        return {}
    return dict(zip(("min", "p10", "median", "p90", "max"),
                    np.round(np.percentile(values, [0, 10, 50, 90, 100]), 3).tolist()))


def audit_archive(path, name, output):
    archive_sha = digest(path)
    cache = output / f"{name}.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved["archive"]["sha256"] != archive_sha:
            raise ValueError(f"Archive changed since checkpoint: {path}")
        return saved
    soccer = name == NAMES[-1]
    pose = name == NAMES[-2]
    if soccer and archive_sha != SOCCERNET_SHA:
        raise ValueError("SoccerNet archive differs from the publisher SHA-256")
    stats = {"ball_sizes_at_640": [], "visible_per_row": [], "visibility_by_index": defaultdict(Counter)}
    counts = defaultdict(Counter)
    class_images = defaultdict(Counter)
    issue_counts = Counter()
    issue_images = Counter()
    match_counts = Counter()
    rows = []
    used_labels = set()
    with ZipFile(path) as archive:
        members = archive_members(archive)
        if bad := archive.testzip():
            raise ValueError(f"CRC failure: {bad}")
        config = {} if soccer else yaml.safe_load(archive.read("data.yaml"))
        metadata = {n: archive.read(n).decode("utf-8") for n in members if n.startswith("README")}
        match_info = json.loads(archive.read("valid/match_info.json")) if soccer else {}
        per_match = json.loads(archive.read("valid/per_match_info.json")) if soccer else {}
        images = sorted(n for n in members if Path(n).suffix.lower() in IMAGE_SUFFIXES)
        for filename in images:
            file = Path(filename)
            split = file.parts[0]
            data = archive.read(filename)
            image, phash, brightness = image_features(data)
            height, width = image.shape[:2]
            root, clip, group = source_keys(file.stem)
            if soccer:
                info = match_info[file.name]
                group = " | ".join(info[k] for k in ("league", "season", "match", "date"))
                clip = f"{group} | {info['half']} | {info['event_id']}"
                match_counts[group] += 1
                label = file.with_suffix(".json").as_posix()
            else:
                label = (file.parent.parent / "labels" / file.with_suffix(".txt").name).as_posix()
            if label in members:
                used_labels.add(label)
                raw = archive.read(label).decode("utf-8")
                if soccer:
                    features, issues = soccer_annotations(raw)
                else:
                    features, issues = yolo_annotations(raw, config["names"], pose, width, height, stats)
                if not features:
                    issues["empty_annotation"] += 1
                if pose and len(features) != 1:
                    issues["pose_row_count_not_one"] += 1
            else:
                features = []
                issues = Counter({"missing_annotation": 1})
            classes = Counter(kind for kind, _, _ in features)
            counts[split].update(f"{kind} ({shape})" for kind, shape, _ in features)
            class_images[split].update(classes.keys())
            issue_counts.update(issues)
            issue_images.update(issues.keys())
            rows.append(dict(path=filename, split=split, stem=root, clip=clip, group=group,
                             resolution=f"{width}x{height}", brightness=round(brightness, 2),
                             sha256=hashlib.sha256(data).hexdigest(),
                             pixel_sha256=hashlib.sha256(
                                 f"{width}x{height}:".encode() + image.tobytes()).hexdigest(),
                             phash=phash, label=label, classes=classes, issues=issues))
        labels = {n for n in members if ("/labels/" in n and n.endswith(".txt"))
                  or (soccer and re.fullmatch(r"valid/\d+\.json", n))}
        ball = stats.pop("ball_sizes_at_640")
        stats.update(ball_size_at_640=quantiles(ball), ball_under_8_at_640=sum(v < 8 for v in ball),
                     visible_per_row=quantiles(stats["visible_per_row"]))
        record = dict(
            archive=dict(filename=path.name, bytes=path.stat().st_size, sha256=archive_sha,
                         crc="passed", entries=len(archive.infolist()), unique_files=len(members),
                         extracted_bytes=sum(m.file_size for m in members.values())),
            config=config, readmes=metadata, rows=rows, counts=counts, class_images=class_images,
            issue_rows=issue_counts, issue_images=issue_images, orphan_labels=sorted(labels-used_labels),
            resolutions=Counter(row["resolution"] for row in rows),
            images_by_split=Counter(row["split"] for row in rows),
            image_brightness=quantiles([row["brightness"] for row in rows]), **stats,
            matches=match_counts, per_match_metadata_groups=len(per_match),
            match_metadata_total=len(match_info),
        )
    write_json(cache, record)
    print(f"Audited {name}: {len(rows)} images", flush=True)
    return record


def pair_inventory(records, output):
    combined = [(name, row) for name in NAMES for row in records[name]["rows"]]
    hashes = np.array([row["phash"] for _, row in combined], dtype=np.uint64)
    sha = defaultdict(list)
    pixels = defaultdict(list)
    stems = defaultdict(list)
    for i, (name, row) in enumerate(combined):
        sha[row["sha256"]].append(i)
        pixels[row["pixel_sha256"]].append(i)
        if row["group"] and name != NAMES[-1]:
            stems[row["stem"]].append(i)
    candidates = set()
    for start in range(0, len(hashes), 128):
        distances = np.bitwise_count(hashes[start:start+128, None] ^ hashes)
        left, right = np.where(distances <= 6)
        candidates.update((int(i+start), int(j)) for i, j in zip(left, right) if i+start < j)
    for groups in (sha, pixels, stems):
        for indexes in groups.values():
            candidates.update(itertools.combinations(indexes, 2))
    comparison = defaultdict(Counter)
    samples = {}
    with (output / "duplicate-candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset_a", "path_a", "dataset_b", "path_b", "hamming", "exact_file",
                         "exact_pixels", "same_frame_key", "cross_split", "class_counts_differ"))
        for i, j in sorted(candidates):
            a, x = combined[i]
            b, y = combined[j]
            hamming = (x["phash"] ^ y["phash"]).bit_count()
            exact = x["sha256"] == y["sha256"]
            pixel = x["pixel_sha256"] == y["pixel_sha256"]
            stem = bool(x["group"] and y["group"] and x["stem"] == y["stem"]
                        and NAMES[-1] not in (a, b))
            cross = x["split"] != y["split"]
            different = x["classes"] != y["classes"]
            writer.writerow((a, x["path"], b, y["path"], hamming, exact, pixel, stem, cross, different))
            summary = comparison[f"{a} | {b}"]
            for condition, key in ((exact, "exact_file"), (pixel, "exact_pixels"), (stem, "same_frame_key"),
                                   (hamming <= 6 and not pixel, "near_candidate"),
                                   (pixel and different, "pixel_exact_class_count_disagreement")):
                if condition:
                    summary[key] += 1
                    if cross:
                        summary[f"{key}_cross_split"] += 1
            if not pixel and hamming <= 6:
                band = "0-2" if hamming <= 2 else "3-6"
                bucket = samples.setdefault((a, b, band), [])
                bucket.append((a, x["path"], b, y["path"], hamming))
    duplicate_samples = []
    rng = random.Random(42)
    for bucket in samples.values():
        duplicate_samples.extend(rng.sample(bucket, min(2, len(bucket))))
    groups = {}
    common_sources = {}
    for a, record in records.items():
        by_group = defaultdict(Counter)
        clips = defaultdict(set)
        by_stem = defaultdict(Counter)
        for row in record["rows"]:
            if row["group"]:
                by_group[row["group"]][row["split"]] += 1
                clips[row["group"]].add(row["clip"])
            by_stem[row["stem"]][row["split"]] += 1
        groups[a] = dict(
            prefix_groups=by_group, clips=sum(len(v) for v in clips.values()),
            ungrouped=sum(not row["group"] for row in record["rows"]),
            frame_keys=len(by_stem), repeated_frame_keys=sum(sum(v.values()) > 1 for v in by_stem.values()),
            frame_keys_cross_split=sum(len(v) > 1 for v in by_stem.values()),
            source_groups_cross_split=sum(len(v) > 1 for v in by_group.values()),
            max_frame_copies=max(sum(v.values()) for v in by_stem.values()),
        )
        for b in list(records)[:list(records).index(a)]:
            other = records[b]["rows"]
            common_sources[f"{b} | {a}"] = {
                "common_frame_keys": len(set(by_stem) & {row["stem"] for row in other}),
                "common_source_groups": len(set(by_group) & {row["group"] for row in other}),
            }
    write_json(output / "overlap.json", dict(pairs=comparison, groups=groups, between=common_sources,
                                            perceptual_review_samples=duplicate_samples))
    return duplicate_samples


def contact_sheet(path, tiles, columns=3):
    if not tiles:
        return
    width, height = 640, 382
    canvas = np.full((math.ceil(len(tiles)/columns)*height, columns*width, 3), 22, np.uint8)
    for i, (image, titles) in enumerate(tiles):
        y, x = (i//columns)*height, (i % columns)*width
        h, w = image.shape[:2]
        scale = min((width-6)/w, (height-44)/h)
        preview = cv2.resize(image, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        canvas[y+42:y+42+preview.shape[0], x+3:x+3+preview.shape[1]] = preview
        for j, title in enumerate(titles):
            cv2.putText(canvas, title[:92], (x+4, y+15+17*j), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, (245, 245, 245), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write {path}")


def sample_tiles(downloads, records, duplicates, output):
    by_path = {name: {r["path"]: r for r in record["rows"]} for name, record in records.items()}
    colors = {"player": (80, 230, 80), "goalkeeper": (40, 145, 250), "referee": (250, 170, 40),
              "ball": (30, 40, 255), "corner": (160, 50, 220), "goal": (210, 210, 30)}
    rng = random.Random(42)
    selected = {}
    for name, record in records.items():
        pool = record["rows"]
        regular = []
        for split in sorted(record["images_by_split"]):
            rows = [row for row in pool if row["split"] == split]
            rng.shuffle(rows)
            seen_groups = set()
            for row in rows:
                if row["group"] not in seen_groups:
                    regular.append(row)
                    seen_groups.add(row["group"])
                if len(seen_groups) == (6 if name == NAMES[-1] else 2):
                    break
        extra = []
        for tag in ("ball", "goalkeeper"):
            candidates = [r for r in pool if tag in r["classes"] and r not in regular+extra]
            if candidates:
                extra.append(rng.choice(candidates))
        anomalous = [r for r in pool if r["issues"] and r not in regular+extra]
        if anomalous:
            extra.append(rng.choice(anomalous))
        if name == NAMES[-1]:
            eligible = [r for r in pool if len(r["classes"]) > 8 and r not in regular+extra]
            extra.extend(rng.sample(eligible, min(2, len(eligible))))
        selected[name] = [
            {"path": r["path"], "selection": "group_stratified" if i < len(regular) else "targeted"}
            for i, r in enumerate(regular+extra)
        ]
        tiles = []
        with ZipFile(downloads / ARCHIVES[name][0]) as archive:
            for i, row in enumerate(regular+extra):
                image, _, _ = image_features(archive.read(row["path"]))
                h, w = image.shape[:2]
                if row["label"] not in archive.namelist():
                    features = []
                elif name == NAMES[-1]:
                    features, _ = soccer_annotations(archive.read(row["label"]))
                else:
                    stats = {"ball_sizes_at_640": [], "visible_per_row": [], "visibility_by_index": defaultdict(Counter)}
                    features, _ = yolo_annotations(archive.read(row["label"]).decode(), record["config"]["names"],
                                                   name == NAMES[-2], w, h, stats)
                for category, kind, values in features:
                    color = colors.get(category, (200, 180, 40))
                    if kind == "box":
                        x, y, bw, bh = values
                        p0 = round((x-bw/2)*w), round((y-bh/2)*h)
                        p1 = round((x+bw/2)*w), round((y+bh/2)*h)
                        cv2.rectangle(image, p0, p1, color, max(1, w//480))
                        if category == "ball":
                            cv2.circle(image, (round(x*w), round(y*h)), max(8, w//64), color, 2)
                    elif kind == "pose":
                        for j, (x, y, visible) in enumerate(np.array(values[4:]).reshape(-1, 3)):
                            if visible > 0:
                                p = round(x*w), round(y*h)
                                cv2.circle(image, p, 4, color, -1)
                                cv2.putText(image, str(j+1), p, cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                                            (40, 40, 250) if visible == 1 else (255, 255, 255), 2)
                    else:
                        pts = np.round(np.array(values).reshape(-1, 2)*[w, h]).astype(np.int32)
                        if kind == "polygon" or not category.startswith("Circle"):
                            cv2.polylines(image, [pts], kind == "polygon", color, 2)
                        for p in pts:
                            cv2.circle(image, p, 3, (255, 50, 230), -1)
                short = row["path"].split("/")[-1].split(".rf.")[0]
                tiles.append((image, (f"{i+1}. {row['split']}  {short}", str(row["classes"]))))
        contact_sheet(output / f"sample-{name}.jpg", tiles)
    write_json(output / "sample-manifest.json", selected)
    opened = {name: ZipFile(downloads / ARCHIVES[name][0]) for name in records}
    try:
        tiles = []
        for i, (a, p, b, q, dist) in enumerate(duplicates):
            left, _, _ = image_features(opened[a].read(p))
            right, _, _ = image_features(opened[b].read(q))
            pair = np.hstack((cv2.resize(left, (420, 236)), cv2.resize(right, (420, 236))))
            tiles.append((pair, (f"{i+1}. d={dist} {a} / {b}", f"{by_path[a][p]['stem']} / {by_path[b][q]['stem']}")))
        for start in range(0, len(tiles), 12):
            contact_sheet(output / f"near-pairs-{start//12+1}.jpg", tiles[start:start+12], columns=2)
    finally:
        for archive in opened.values():
            archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--storage", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (args.storage or args.output):
        parser.error("Provide --storage for verified extraction or --output for analysis")
    if args.storage:
        extract_archives(args.downloads, args.storage)
    if args.output:
        output = args.output
        output.mkdir(parents=True, exist_ok=True)
        records = {name: audit_archive(args.downloads / ARCHIVES[name][0], name, output) for name in NAMES}
        total = sum(record["archive"]["bytes"] for record in records.values())
        overview = {name: {key: value for key, value in record.items() if key != "rows"}
                    for name, record in records.items()}
        write_json(output / "inventory.json", dict(datasets=overview, budget_bytes=BUDGET,
                                                  total_zip_bytes=total, remaining_bytes=BUDGET-total))
        samples = pair_inventory(records, output)
        sample_tiles(args.downloads, records, samples, output)
        print(f"Analysis saved to {output}", flush=True)


if __name__ == "__main__":
    main()
