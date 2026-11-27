"""Assisted SigLIP classification with explicit representative player crops."""
from dataclasses import dataclass
import math
import cv2
import numpy as np


@dataclass(frozen=True)
class AssistedConfig:
    sample_interval: int = 10
    max_age: int = 30
    min_observations: int = 2
    min_similarity: float = 0.75
    min_margin: float = 0.02
    smoothing: float = 0.25

    def __post_init__(self):
        for x in (self.sample_interval, self.max_age, self.min_observations):
            if isinstance(x, bool) or not isinstance(x, int) or x < 1:
                raise ValueError('intervals and observation count must be positive integers')
        if self.max_age < self.sample_interval:
            raise ValueError('max_age must be at least sample_interval')
        for x in (self.min_similarity, self.min_margin, self.smoothing):
            if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or not 0 < x <= 1:
                raise ValueError('invalid similarity, margin or smoothing')


def normalized(features):
    data = np.asarray(features, dtype=np.float32)
    if data.ndim != 2 or not np.isfinite(data).all():
        raise ValueError('embeddings must be a finite matrix')
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError('embeddings must be nonzero')
    return data / norms


class SiglipEncoder:
    """Load the user's local pretrained vision model, never accessing the network."""
    def __init__(self, snapshot, device='cpu', batch_size=16, *, mean_pool=False):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.mean_pool = mean_pool
        self.processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
        model, loading = SiglipVisionModel.from_pretrained(snapshot, local_files_only=True, output_loading_info=True)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('SigLIP snapshot has missing, mismatched or invalid vision weights')
        self.model = model.to(device).eval()

    def __call__(self, crops):
        from PIL import Image
        result = []
        for i in range(0, len(crops), self.batch_size):
            imgs = [Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in crops[i:i+self.batch_size]]
            inputs = self.processor(images=imgs, return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                output = self.model(**inputs)
                features = output.last_hidden_state.mean(dim=1) if self.mean_pool else output.pooler_output
            result.extend(features.float().cpu().numpy())
        return normalized(result)


class AssistedTeamClassifier:
    """Fixed user-labeled prototypes and bounded track histories, reset on cuts."""
    def __init__(self, encoder, references, config=None):
        self.encoder = encoder
        self.config = config or AssistedConfig()
        if set(references) != {1, 2} or any(not references[k] for k in (1, 2)):
            raise ValueError('representative crops are required for both teams')
        embeddings = [normalized(encoder(references[k])).mean(axis=0) for k in (1, 2)]
        self.prototypes = normalized(embeddings)
        cosine = float(np.clip(np.dot(*self.prototypes), -1.0, 1.0))
        optimum_similarity = math.sqrt(max(0.0, (1.0-cosine)/2.0))
        if optimum_similarity >= self.config.min_similarity:
            max_margin = math.sqrt(max(0.0, 2.0-2.0*cosine))
        else:
            minimum = self.config.min_similarity
            max_margin = (1.0-cosine)*minimum + math.sqrt(1.0-minimum**2)*math.sqrt(max(0.0, 1.0-cosine**2))
        if max_margin < self.config.min_margin:
            raise ValueError('representatives are too similar to distinguish')
        self.reset()

    def reset(self):
        self.segment = None
        self.history = {}

    def update(self, frame, tracked):
        if self.segment != tracked.segment or tracked.camera_cut:
            self.reset()
            self.segment = tracked.segment
        c = self.config
        self.history = {k:v for k,v in self.history.items() if tracked.frame_index-v[1] <= c.max_age}
        targets, crops = [], []
        for t in tracked.tracks:
            if t.class_id != 0 or t.track_id is None:
                continue
            old = self.history.get(t.track_id)
            if old is not None and tracked.frame_index-old[1] < c.sample_interval:
                continue
            x1,y1,x2,y2 = map(int,t.xyxy)
            crop = frame[max(0,y1):min(frame.shape[0],y2),max(0,x1):min(frame.shape[1],x2)]
            if crop.shape[0] < 25 or crop.shape[1] < 12:
                continue
            targets.append(t.track_id); crops.append(crop)
        if crops:
            for key, feature in zip(targets, normalized(self.encoder(crops))):
                old = self.history.get(key)
                similarity = self.prototypes @ feature
                order = np.argsort(similarity)
                candidate = None
                if similarity[order[-1]] >= c.min_similarity and similarity[order[-1]] - similarity[order[-2]] >= c.min_margin:
                    candidate = int(order[-1]) + 1
                if old is None:
                    average, count = feature, 1 if candidate else 0
                else:
                    combined = (1-c.smoothing)*old[0]+c.smoothing*feature
                    if np.linalg.norm(combined) < 1e-8:
                        average, candidate, count = feature, None, 0
                    else:
                        average = normalized([combined])[0]
                        count = old[2]+1 if candidate is not None and candidate == old[3] else (1 if candidate else 0)
                self.history[key] = (average, tracked.frame_index, count, candidate)
        result = []
        for t in tracked.tracks:
            state = self.history.get(t.track_id) if t.class_id == 0 else None
            team = None
            if state is not None and state[2] >= c.min_observations:
                similarity = self.prototypes @ state[0]
                order = np.argsort(similarity)
                if similarity[order[-1]] >= c.min_similarity and similarity[order[-1]] - similarity[order[-2]] >= c.min_margin and int(order[-1])+1 == state[3]:
                    team = state[3]
            result.append(team)
        return result
