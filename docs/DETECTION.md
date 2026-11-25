# Object detection data

The object detection baseline has been evaluated on held-out broadcast frames, with the limitations below. The prepared dataset and weights are local artifacts and are not committed to Git. The configured raw and processed paths can be changed per machine.

## Sources and attribution

- [Football Players Detection v10](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/10), Roboflow Universe, CC BY 4.0 as stated in the supplied export: 312 original 1920 × 1080 frames.
- [Football Object Detection v1](https://universe.roboflow.com/football-detect/football-object-detection/dataset/1), Roboflow Universe, CC BY 4.0 as stated in the supplied export: 1,150 640 × 640 images with provider-generated crop variants. All 242 distinct frame identifiers in this export also occur in v10.

The license statements above are from the dataset exports; rights to the underlying broadcast footage have not been independently verified.

Both export class mappings are verified against their metadata. The output mapping is `0: player`, `1: goalkeeper`, `2: referee`, `3: ball`. Goal polygons and corner boxes are removed. Images and boxes are validated, with sub-pixel edge overflows clipped only within a one-pixel tolerance. Conflicting retained-class labels on byte-identical images stop preparation.

## Split and limitations

SHA-256 identifies exact duplicates; a 64-bit DCT perceptual hash flags similar images. The filename prefix groups related sequences and crops across both exports. A visual stadium review prompted conservative grouping of the 15 prefixes into six stadium groups in [the source configuration](../configs/detection_sources.json); cross-prefix perceptual-hash candidates are grouped as well. Stadium grouping is a precaution, not proof of original match identity; the source does not provide authoritative match IDs, so residual leakage remains possible.

With seed 42, held-out groups are selected using original-frame counts and class representation. Stadium grouping takes priority over the 80/10/10 target: the original-frame split assigned 241/44/27. Two previously inspected test-group frames are excluded from final scoring, leaving 241/44/25. Training has 1,145 images after adding variants and removing two duplicates; validation and test use only originals. The third exact duplicate is among excluded held-out augmentations. Validation covers two stadium groups, while test covers only one and has five goalkeeper boxes. Adjacent images may be highly correlated, so these sets provide only a limited estimate of performance on new matches.

An earlier prefix-only split placed the same stadium in multiple splits. Its baseline was stopped after 11 epochs and is not used for the final model. Stadium groups from that run's validation set are explicitly ineligible for the revised test split; the revised baseline starts again from the official pretrained weights.

Two preselected frames from the revised test group were accidentally used for an early checkpoint's detector smoke check. They are explicitly excluded from the final test dataset. No model or threshold was selected from those predictions. The baseline uses `detection-v2`; `detection-v4` excludes those two frames, and all training and validation images and labels were verified byte-for-byte identical across the versions.

Earlier dataset manifests hashed LF-normalized label text, while Windows wrote CRLF files, so their label checksums do not match the saved bytes. Version 4 writes explicit CRLF labels and records the SHA-256 of those bytes. All retained raw images, processed images and label hashes were verified against the final manifest.

The sample of annotated training images reviewed was generally plausible, but this is not a complete annotation-quality audit. At 640 input size, 270 of 272 original ball boxes are smaller than 8 × 8 pixels after aspect-preserving resize; this and class imbalance make ball detection difficult.

## Prepare and verify

In PowerShell, from the project root, change the paths to suit the local machine. The output directory must not already exist and must not be inside or contain the raw directory.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_detection_dataset.py `
  --raw E:\PitchLensStorage\datasets\raw `
  --output data\processed\detection-v4
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The output contains YOLO image and label folders, `data.yaml`, a per-image `manifest.json` with source IDs and SHA-256 hashes, and `report.json` with grouping, class counts and audit results. Neither the raw data nor the processed data should be committed. Use a new output directory for each preparation instead of overwriting an earlier one.

## Train the baseline

The approved baseline uses the official pretrained `yolo26s.pt`, 640-pixel input, CUDA, AMP, automatic batch size and optimizer, seed 42, a 100-epoch maximum and early stopping after 20 unimproved epochs. [The configuration](../configs/detection_train.json) uses modest color, translation, scale, horizontal-flip and mosaic augmentations. Vertical flips, rotation, mixup and perspective changes are disabled. A smoke run uses 5% of training images for one epoch and never uses the test split.

Download the official weights separately to an ignored local location and supply the paths explicitly. Runs must use a new directory; existing experiments are never overwritten.

```powershell
.\.venv\Scripts\python.exe scripts\train_detection.py `
  --data data\processed\detection-v4\data.yaml `
  --weights models\yolo26s.pt --run runs\detection-smoke --smoke
.\.venv\Scripts\python.exe scripts\train_detection.py `
  --data data\processed\detection-v4\data.yaml `
  --weights models\yolo26s.pt --run runs\detection-baseline-640
```

Each run records input and manifest hashes, requested settings, GPU details, losses and validation metrics, per-epoch peak PyTorch GPU memory, and checkpoints. The selected batch and optimizer are stored in checkpoints; periodic training checkpoints retain optimizer state, while Ultralytics strips it from finalized `best.pt` and `last.pt`. `best.pt` is selected on validation, never on test data. The figures depend on hardware and library versions; deterministic settings do not guarantee bit-for-bit equivalence across platforms.

Ultralytics identifies its installed package license as AGPL-3.0. The PitchLens project license remains an open decision.

## Frame detector and evaluation

The detector loads its weights once, selects CUDA when available and otherwise falls back to CPU. It accepts a nonempty OpenCV BGR `uint8` frame and returns `Detection` objects with `(left, top, right, bottom)` pixel coordinates in the original frame, class IDs, and confidence. The selected default is 960 pixels and 0.25 confidence. The global confidence threshold, optional per-class confidence thresholds, input resolution and device are configurable. The IoU setting is passed through for models that use non-maximum suppression; the standard end-to-end YOLO26s head is NMS-free, so that setting does not affect its predictions. Missing weights, incompatible class mappings and invalid frames are rejected.

```python
import cv2
from pitchlens.detector import Detector

detector = Detector("runs/detection-baseline-640/weights/best.pt", confidence=0.25)
frame = cv2.imread("path/to/authorized/frame.jpg")
for detection in detector.detect(frame):
    print(detection.class_id, detection.confidence, detection.xyxy)
```

Select the checkpoint, input resolution and operating threshold using validation only. The following evaluates a frozen checkpoint with the preselected 960-pixel resolution and 0.25 threshold; use a new output directory for each run. Only use `--split test` after model selection is final, and do not use test results to tune the model or threshold.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detection.py `
  --data data\processed\detection-v4\data.yaml `
  --weights runs\detection-baseline-640\weights\best.pt `
  --run runs\detection-final-test --split test --image-size 960 --confidence 0.25
```

`evaluation.json` records mAP50 and mAP50–95, per-class AP, and precision, recall and confusion counts at the fixed threshold and IoU 0.5. Overall precision and recall are macro averages over classes. Ultralytics' own console output may show precision and recall at its F1-selected confidence; use the fixed-threshold values in `evaluation.json` for the untouched test report. The utility also benchmarks the frame API on a validation frame without file I/O or drawing, records PyTorch peak GPU memory, and saves annotated validation samples. The benchmark excludes video decoding and encoding and is not an end-to-end video throughput claim.

## Measured baseline

The 640-pixel training run stopped after 81 epochs with the best validation checkpoint at epoch 61. Its local `best.pt` SHA-256 is `29135b3b60a241f006f88f042db57b7c60ad3c7f161454683338eea01cb4b94b`. The local experiment and evaluation files are in `runs/detection-baseline-640-v2/` and `runs/detection-final-test-960-v1/`.

Before the final test, inference resolutions were compared on the 44 validation images using the frozen checkpoint and 0.25 confidence:

| Input | mAP50 | mAP50–95 | Ball recall | Ball false positives | GPU FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 640 | 82.23% | 54.00% | 48.78% | 6 | 51.94 |
| 960 | 84.29% | 58.30% | 56.10% | 4 | 55.61 |
| 1280 | 81.49% | 54.99% | 63.41% | 5 | 45.41 |

The 960-pixel setting had the best aggregate validation metrics and fewer false ball detections. The 1280-pixel setting had higher ball recall but lower aggregate performance. These comparisons use a small validation set; the FPS measurements repeat one frame and are not a video benchmark.

The final 25-image test at 960 pixels and 0.25 confidence produced macro precision 83.63%, recall 69.94%, mAP50 81.24% and mAP50–95 55.61%:

| Class | Instances | Precision | Recall | AP50 |
| --- | ---: | ---: | ---: | ---: |
| Player | 499 | 91.79% | 98.60% | 98.64% |
| Goalkeeper | 5 | 75.00% | 60.00% | 61.36% |
| Referee | 54 | 86.49% | 59.26% | 90.11% |
| Ball | 21 | 81.25% | 61.90% | 74.87% |

At this threshold the model missed eight balls and produced three false ball detections. Twenty-one referee boxes were classified as players. The frame benchmark measured 54.06 FPS, with 128,099,840 bytes peak allocated and 148,897,792 bytes peak reserved by PyTorch during inference. Several provisional aggregate targets and the goalkeeper AP50 target were not met. The very small, correlated single-stadium test set, particularly its five goalkeeper boxes, means these numbers should not be generalized to new broadcasts without further independent evaluation. No authorized video was available for video-level testing.
