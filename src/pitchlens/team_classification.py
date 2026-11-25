"""Conservative segment-local jersey colour classification."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class TeamConfig:
    min_pixels: int = 30
    min_tracks: int = 6
    min_separation: float = 20.0
    max_distance: float = 45.0
    min_margin: float = 8.0
    smoothing: float = 0.2
    min_chroma_separation: float = 30.0
    max_inactive_frames: int = 30
    allow_neutral_split: bool = False

    def __post_init__(self):
        for value, minimum in ((self.min_pixels, 1), (self.min_tracks, 4), (self.max_inactive_frames, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError('Invalid team classification count')
        for value in (self.min_separation, self.max_distance, self.min_margin, self.smoothing, self.min_chroma_separation):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid team classification threshold')
        if not isinstance(self.allow_neutral_split, bool):
            raise ValueError('allow_neutral_split must be boolean')
        if self.min_tracks > 100:
            raise ValueError("min_tracks exceeds retained track limit")
        if self.smoothing > 1:
            raise ValueError('Invalid smoothing')


def jersey_colour(frame, box, min_pixels=30):
    """Median LAB colour of the central torso, excluding likely grass and skin."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    xa, xb = max(0, int(x1 + .25*(x2-x1))), min(w, int(x1 + .75*(x2-x1)))
    ya, yb = max(0, int(y1 + .20*(y2-y1))), min(h, int(y1 + .55*(y2-y1)))
    crop = frame[ya:yb, xa:xb]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    # Only mask green pixels matching the surrounding pitch-like background.
    margin = max(4, int((x2-x1)*.5))
    left, right = max(0, int(x1)-margin), min(w, int(x2)+margin)
    top, bottom = max(0, int(y1)-margin), min(h, int(y2)+margin)
    outer = frame[top:bottom, left:right]
    background = cv2.cvtColor(outer, cv2.COLOR_BGR2HSV) if outer.size else hsv
    yy, xx = np.indices(background.shape[:2])
    outside = (xx + left < x1) | (xx + left >= x2) | (yy + top < y1) | (yy + top >= y2)
    green = outside & (background[:,:,0] >= 35) & (background[:,:,0] <= 90) & (background[:,:,1] > 45)
    grass = np.zeros(hsv.shape[:2], dtype=bool)
    if outside.any() and green.sum() / outside.sum() > .2:
        pitch = np.median(background[green], axis=0)
        grass = (np.abs(hsv[:,:,0].astype(float)-pitch[0]) < 10) & (hsv[:,:,1] > 45) & (np.abs(hsv[:,:,1].astype(float)-pitch[1]) < 50) & (np.abs(hsv[:,:,2].astype(float)-pitch[2]) < 35)
    skin = (ycrcb[:,:,1] >= 135) & (ycrcb[:,:,1] <= 180) & (ycrcb[:,:,2] >= 85) & (ycrcb[:,:,2] <= 135) & (hsv[:,:,1] < 150)
    # Very dark pixels carry too little reliable colour information. A track's
    # last useful observation can bridge shadows, but a new dark track is unknown.
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[~(grass | skin | (hsv[:,:,2] < 40))]
    if len(pixels) < min_pixels:
        return None
    return np.median(pixels, axis=0).astype(np.float32)


class TeamClassifier:
    """Maintain per-track colour estimates and segment-local team centroids."""
    def __init__(self, config=None):
        self.config = config or TeamConfig()
        self.reset()

    def reset(self):
        self.segment = None
        self.colours = {}
        self.centres = None
        self.last_seen = {}
        self.previous_labels = {}
        self.label_frames = {}
        self.unmatched_frames = 0

    def update(self, frame, tracked):
        if self.segment != tracked.segment or tracked.camera_cut:
            self.reset()
            self.segment = tracked.segment
        c = self.config
        expired = [key for key, last in self.last_seen.items() if tracked.frame_index - last > c.max_inactive_frames]
        for key in expired:
            self.colours.pop(key, None)
            del self.last_seen[key]
        for track in tracked.tracks:
            if track.class_id != 0 or track.track_id is None:
                continue
            colour = jersey_colour(frame, track.xyxy, c.min_pixels)
            if colour is not None:
                self.last_seen[track.track_id] = tracked.frame_index
                old = self.colours.get(track.track_id)
                self.colours.pop(track.track_id, None)
                self.colours[track.track_id] = colour if old is None else (1-c.smoothing)*old + c.smoothing*colour
                if len(self.colours) > 100:
                    oldest = next(iter(self.colours))
                    del self.colours[oldest]
                    del self.last_seen[oldest]
        prior_centres = self.centres
        anchor_nearest = anchor_confident = None
        fit_valid = self.centres is not None
        if self.centres is not None and len(self.colours) >= c.min_tracks:
            current = np.array(list(self.colours.values()), dtype=np.float32)
            distance = np.linalg.norm(current[:,None,:]-self.centres[None,:,:], axis=2)
            nearest = distance.argmin(axis=1)
            confident = (distance.min(axis=1) <= c.max_distance) & (np.abs(distance[:,0]-distance[:,1]) >= c.min_margin)
            support = np.bincount(nearest[confident], minlength=2)
            anchor_nearest = np.array([self.previous_labels.get(key, 0)-1 for key in self.colours])
            active_ids = {t.track_id for t in tracked.tracks if t.class_id == 0}
            anchor_confident = np.array([key in active_ids and key in self.label_frames and tracked.frame_index - self.label_frames[key] <= c.max_inactive_frames + 30 for key in self.colours]) & (anchor_nearest >= 0)
            self.unmatched_frames = self.unmatched_frames + 1 if np.count_nonzero(~confident) >= 2 else 0
            if np.count_nonzero(~confident) >= 2 and (min(support) < 2 or self.unmatched_frames >= 30):
                self.centres = None
                fit_valid = False
        if self.centres is None and len(self.colours) >= c.min_tracks:
            samples = np.array(list(self.colours.values()), dtype=np.float32)
            # Deterministic initial labels based on the most distant pair.
            distances = np.linalg.norm(samples[:,None] - samples[None,:], axis=2)
            i, j = np.unravel_index(np.argmax(distances), distances.shape)
            labels = (np.linalg.norm(samples-samples[j], axis=1) < np.linalg.norm(samples-samples[i], axis=1)).astype(np.int32).reshape(-1, 1)
            if 0 < labels.sum() < len(labels):
                _, labels, centres = cv2.kmeans(samples, 2, labels, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, .1), 1, cv2.KMEANS_USE_INITIAL_LABELS)
                chroma = centres[:, 1:] - 128.0
                norms = np.linalg.norm(chroma, axis=1)
                same_hue = min(norms) > 15 and float(np.dot(chroma[0], chroma[1]) / (norms[0] * norms[1])) > .95
                neutral = c.allow_neutral_split and max(norms) < 15 and abs(float(centres[0,0]-centres[1,0])) >= 80
                separation = np.linalg.norm(centres[0]-centres[1]) >= c.min_separation
                chroma_ok = np.linalg.norm(centres[0,1:]-centres[1,1:]) >= c.min_chroma_separation
                sample_distances = np.linalg.norm(samples[:,None,:]-centres[None,:,:], axis=2)
                nearest = sample_distances.argmin(axis=1)
                margin = np.abs(sample_distances[:,0]-sample_distances[:,1])
                accepted = (sample_distances.min(axis=1) <= c.max_distance) & (margin >= c.min_margin)
                usable = np.bincount(nearest[accepted], minlength=2)
                if separation and (neutral or (not same_hue and chroma_ok)) and min(usable) >= 2:
                    fit_valid = True
                    if prior_centres is None:
                        self.centres = centres[np.lexsort((centres[:,2], centres[:,1], centres[:,0]))]
                    else:
                        same = np.count_nonzero(anchor_confident & accepted & (anchor_nearest == nearest))
                        swapped = np.count_nonzero(anchor_confident & accepted & (anchor_nearest != nearest))
                        if same == swapped:
                            fit_valid = False
                        else:
                            self.centres = centres if same > swapped else centres[::-1]
        if self.centres is None and prior_centres is not None:
            self.centres = prior_centres
            fit_valid = True

        result = []
        for track in tracked.tracks:
            team = None
            if track.class_id == 0 and fit_valid and track.track_id in self.colours and tracked.frame_index - self.last_seen[track.track_id] <= c.max_inactive_frames:
                d = np.linalg.norm(self.centres-self.colours[track.track_id], axis=1)
                order = np.argsort(d)
                if d[order[0]] <= c.max_distance and d[order[1]]-d[order[0]] >= c.min_margin:
                    team = int(order[0]) + 1
            result.append(team)
        for t, team in zip(tracked.tracks, result):
            if t.class_id == 0 and t.track_id is not None and team is not None:
                self.previous_labels[t.track_id] = team
                self.label_frames[t.track_id] = tracked.frame_index
        for key in list(self.previous_labels):
            if key not in self.colours or tracked.frame_index - self.label_frames[key] > c.max_inactive_frames + 30:
                del self.previous_labels[key]
                del self.label_frames[key]
        # Goalkeeper team assignment is deferred until Feature 04.
        return result
