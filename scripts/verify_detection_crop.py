"""Compare a provider crop's labels with its original image in crop coordinates."""

import cv2
import numpy as np

SIZE = 640


class CropVerifier:
    """Align independently of labels, then require one-to-one same-class boxes."""

    def __init__(self):
        self.detector = cv2.SIFT_create(nfeatures=1100)
        self.matcher = cv2.BFMatcher()
        self.original = None
        self.features = ()

    def describe(self, gray):
        points, descriptors = self.detector.detectAndCompute(gray, None)
        return np.float32([point.pt for point in points]), descriptors

    def check(self, original, crop, original_labels, crop_labels):
        if original != self.original:
            gray = self.gray(original)
            self.features = (self.describe(gray), self.describe(cv2.flip(gray, 1)))
            self.original = original
        target_points, target_descriptors = self.describe(self.gray(crop))
        if target_descriptors is None:
            return "unverified_crop_geometry"
        for mirrored, (source_points, source_descriptors) in enumerate(self.features):
            if source_descriptors is None:
                continue
            pairs = self.matcher.knnMatch(source_descriptors, target_descriptors, k=2)
            matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
            if len(matches) < 18:
                continue
            source = source_points[[match.queryIdx for match in matches]]
            target = target_points[[match.trainIdx for match in matches]]
            transform, inliers = cv2.estimateAffine2D(source, target, ransacReprojThreshold=2.5)
            if transform is None:
                continue
            inlier = inliers.ravel().astype(bool)
            scale = transform.diagonal()
            # Reject small/local feature matches, rotations and identity-style variants.
            if (inlier.sum() < 18 or inlier.mean() < .45
                    or np.any(np.ptp(target[inlier], axis=0) < SIZE * .25)
                    or max(abs(transform[0, 1]), abs(transform[1, 0])) > .015
                    or np.any(scale < .97) or np.any(scale > 2.5) or scale.prod() < 1.15):
                continue
            return "compatible" if boxes_agree(original_labels, crop_labels, scale, transform[:, 2], mirrored) else "annotation_disagreement"
        return "unverified_crop_geometry"

    @staticmethod
    def gray(path):
        image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unreadable image: {path}")
        return cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def corners(x, y, width, height):
    return np.array([x - width / 2, y - height / 2, x + width / 2, y + height / 2]) * SIZE


def area(box):
    return np.maximum(0, box[2:] - box[:2]).prod()


def boxes_agree(original, crop, scale, offset, mirrored):
    expected = []
    for category, x, y, width, height in original:
        projected = corners(1 - x if mirrored else x, y, width, height) * np.tile(scale, 2) + np.tile(offset, 2)
        clipped = np.clip(projected, 0, SIZE)
        if area(clipped):
            expected.append((category, clipped, area(clipped) / area(projected) >= .25))
    matched = set()
    for category, *box in crop:
        actual = corners(*box)
        candidates = []
        for index, (expected_class, projected, _) in enumerate(expected):
            if expected_class != category:
                continue
            overlap = area(np.r_[np.maximum(actual[:2], projected[:2]), np.minimum(actual[2:], projected[2:])])
            iou = overlap / (area(actual) + area(projected) - overlap)
            # A 2 px registration error is material for a 4 px ball.
            tiny_ball = category == 3 and max(projected[2:] - projected[:2]) < 12
            if iou >= .55 or (tiny_ball and np.max(np.abs(actual - projected)) <= 2.5):
                candidates.append((iou, index))
        if not candidates:
            return False
        _, index = max(candidates)
        if index in matched:
            return False
        matched.add(index)
    return all(index in matched or not required for index, (_, _, required) in enumerate(expected))
