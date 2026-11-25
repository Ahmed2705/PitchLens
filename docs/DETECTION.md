# Object detection data

The detection feature is under development. The prepared dataset and weights are local artifacts and are not committed to Git. The configured raw and processed paths can be changed per machine.

## Sources and attribution

- [Football Players Detection v10](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/10), Roboflow Universe, CC BY 4.0 as stated in the supplied export: 312 original 1920 × 1080 frames.
- [Football Object Detection v1](https://universe.roboflow.com/football-detect/football-object-detection/dataset/1), Roboflow Universe, CC BY 4.0 as stated in the supplied export: 1,150 640 × 640 images with provider-generated crop variants. All 242 distinct frame identifiers in this export also occur in v10.

Both export class mappings are verified against their metadata. The output mapping is `0: player`, `1: goalkeeper`, `2: referee`, `3: ball`. Goal polygons and corner boxes are removed. Images and boxes are validated, with sub-pixel edge overflows clipped only within a one-pixel tolerance. Conflicting labels on byte-identical images stop preparation.

## Split and limitations

SHA-256 identifies exact duplicates; a 64-bit DCT perceptual hash flags similar images. The filename prefix groups related sequences and crops across both exports. A visual stadium review prompted conservative grouping of the 15 prefixes into six stadium groups in [the source configuration](../configs/detection_sources.json); cross-prefix perceptual-hash candidates are grouped as well. Stadium grouping is a precaution, not proof of original match identity; the source does not provide authoritative match IDs, so residual leakage remains possible.

With seed 42, held-out groups are selected using original-frame counts and class representation. Stadium grouping takes priority over the 80/10/10 target: the revised original-frame split is 241/44/27 (77.2% / 14.1% / 8.7%). Training has 1,145 images after adding variants and removing two duplicates; validation and test use only originals. The third exact duplicate is among excluded held-out augmentations. Validation covers two stadium groups, while test covers only one and has seven goalkeeper boxes. Adjacent images may be highly correlated, so these sets provide only a limited estimate of performance on new matches.

An earlier prefix-only split placed the same stadium in multiple splits. Its baseline was stopped after 11 epochs and is not used for the final model. Stadium groups from that run's validation set are explicitly ineligible for the revised test split; the revised baseline starts again from the official pretrained weights.

The sample of annotated training images reviewed was generally plausible, but this is not a complete annotation-quality audit. At 640 input size, 270 of 272 original ball boxes are smaller than 8 × 8 pixels after aspect-preserving resize; this and class imbalance make ball detection difficult.

## Prepare and verify

In PowerShell, from the project root, change the paths to suit the local machine. The output directory must not already exist and must not be inside or contain the raw directory.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_detection_dataset.py `
  --raw E:\PitchLensStorage\datasets\raw `
  --output data\processed\detection-v2
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The output contains YOLO image and label folders, `data.yaml`, a per-image `manifest.json` with source IDs and SHA-256 hashes, and `report.json` with grouping, class counts and audit results. Neither the raw data nor the processed data should be committed. Use a new output directory for each preparation instead of overwriting an earlier one.

## Train the baseline

The approved baseline uses the official pretrained `yolo26s.pt`, 640-pixel input, CUDA, AMP, automatic batch size and optimizer, seed 42, a 100-epoch maximum and early stopping after 20 unimproved epochs. [The configuration](../configs/detection_train.json) uses modest color, translation, scale, horizontal-flip and mosaic augmentations. Vertical flips, rotation, mixup and perspective changes are disabled. A smoke run uses 5% of training images for one epoch and never uses the test split.

Download the official weights separately to an ignored local location and supply the paths explicitly. Runs must use a new directory; existing experiments are never overwritten.

```powershell
.\.venv\Scripts\python.exe scripts\train_detection.py `
  --data data\processed\detection-v2\data.yaml `
  --weights models\yolo26s.pt --run runs\detection-smoke --smoke
.\.venv\Scripts\python.exe scripts\train_detection.py `
  --data data\processed\detection-v2\data.yaml `
  --weights models\yolo26s.pt --run runs\detection-baseline-640
```

Each run records input and manifest hashes, requested settings, GPU details, Ultralytics' resolved arguments, losses and validation metrics, per-epoch peak PyTorch GPU memory, and checkpoints. `best.pt` is selected on validation, never on test data. The figures depend on hardware and library versions; deterministic settings do not guarantee bit-for-bit equivalence across platforms.

Ultralytics identifies its installed package license as AGPL-3.0. The PitchLens project license remains an open decision.

## Frame detector and evaluation

The detector loads its weights once, selects CUDA when available and otherwise falls back to CPU. It accepts a nonempty OpenCV BGR `uint8` frame and returns `Detection` objects with `(left, top, right, bottom)` pixel coordinates in the original frame, class IDs, and confidence. Thresholds, input resolution and device are configurable. Missing weights, incompatible class mappings and invalid frames are rejected.

```python
import cv2
from pitchlens.detector import Detector

detector = Detector("runs/detection-baseline-640/weights/best.pt", confidence=0.25)
frame = cv2.imread("path/to/authorized/frame.jpg")
for detection in detector.detect(frame):
    print(detection.class_id, detection.confidence, detection.xyxy)
```

After the baseline has finished, select the checkpoint and the operating threshold using validation only. The following evaluates a frozen checkpoint with the preselected 0.25 threshold; use a new output directory for each run. Only use `--split test` after model selection is final, and do not use test results to tune the model or threshold.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detection.py `
  --data data\processed\detection-v2\data.yaml `
  --weights runs\detection-baseline-640\weights\best.pt `
  --run runs\detection-final-test --split test --confidence 0.25
```

`evaluation.json` records mAP50 and mAP50–95, per-class AP, and precision, recall and confusion counts at the fixed threshold and IoU 0.5. Overall precision and recall are macro averages over classes. Ultralytics' own console output may show precision and recall at its F1-selected confidence; use the fixed-threshold values in `evaluation.json` for the untouched test report. The utility also benchmarks the frame API on a validation frame without file I/O or drawing, records PyTorch peak GPU memory, and saves annotated validation samples. The benchmark excludes video decoding and encoding and is not an end-to-end video throughput claim.
