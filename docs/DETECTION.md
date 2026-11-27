# Object detection data

## Default video detection

`scripts/track_video.py` and `scripts/annotate_video.py` default to the `final` profile in [detection.json](../configs/detection.json). The runners load each local checkpoint once and send each frame to both models. The people detector uses the PitchLens class IDs `0` player, `1` goalkeeper and `2` referee. Its ball class is excluded in dual mode, and the dedicated detector maps its own class `0` to PitchLens class `3`.

| Role | Verified checkpoint | Inference size | Detection confidence | Verified SHA-256 |
| --- | --- | ---: | ---: | --- |
| People | `runs/detection-candidate-v2-640/weights/best.pt` | 640 | 0.25 | `cd05e022e02e5d4f6f6192fea438e8ebf5644a0e8fa575295366af9dc1e658eb` |
| Ball | `E:\PitchLensStorage\models\pretrained\yolo-sn-ball-opt.pt` | 960 | 0.15 | `a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36` |

At 640, YOLO26s Candidate V2 had the best player results among the compared checkpoints on both external sets. Player AP50–95 rose from 7.8% to 41.7% on Crystal Palace–Arsenal and from 49.5% to 51.8% on IFAP against the original production checkpoint at 640. The approved SoccerNet-v3D model is supported by the prior [passing-clip evaluation](SOCCERNET_BALL.md#passing-clip-evaluation). The people comparison report, provenance and metrics are under `E:\PitchLensStorage\outputs\final-detector-comparison-v1`.

At confidence 0.25 on IFAP, Candidate V2 matched 58 of 308 goalkeepers and 197 of 683 referees. Goalkeeper precision on Crystal Palace–Arsenal was 23.2% despite 98.0% recall. Role confusion and missed detections remain material limitations.

From the project root, use a new output stem in an existing directory:

```powershell
$env:ULTRALYTICS_SAFE_LOAD = '1'
.\.venv\Scripts\python.exe scripts/track_video.py `
  --input 'E:\PitchLensStorage\footage\passes2.mp4' `
  --output 'E:\PitchLensStorage\outputs\passes2-tracked.mp4' --device auto
```

- `--detector-profile legacy` selects the original four-class checkpoint at 960 pixels and single-model inference. Existing commands with `--weights` and no profile retain legacy behavior. Add `--ball-weights` to an explicit legacy run to reproduce the earlier dual-model setup.
- `--detector-profile gsr` selects the preserved three-class GSR checkpoint at `runs/gsr-yolo26s-640-v1/weights/best.pt` under the external storage root and uses SoccerNet-v3D for the ball.
- Use `--storage-root` to relocate external checkpoints, `--detector-profile final --weights` to override the people checkpoint in the default dual setup, or `--ball-weights` to override the dedicated checkpoint. Missing or incompatible local weights stop the run.
- Use `--people-image-size` and `--people-confidence` for people settings. For ball settings, use `--ball-config` with a JSON object such as `{"image_size": 960, "confidence": 0.15}`. `--device auto` uses CUDA when available and otherwise uses CPU.

The original checkpoint at `runs/detection-baseline-640-v2/weights/best.pt` matched SHA-256 `29135b3b60a241f006f88f042db57b7c60ad3c7f161454683338eea01cb4b94b`. The GSR checkpoint matched SHA-256 `d70da7d2234479b9247cb274b1b65519531bb0ce0bc3c5b712123f866279d2b2`.

The annotation runner displays people detections at the operational confidence threshold. The tracking runner also passes people candidates down to the configured `person_low` (0.10 by default) to ByteTrack. Established tracks can use those candidates, so track confidence may be below 0.25. The tracker still admits ball candidates at 0.25 or higher, while raw dedicated ball diagnostics go down to 0.15. Tracking and downstream JSONL fields are unchanged, and the run summary records detector settings and candidate counts.

### Integration verification

All 200 automated tests passed with `python -B -m unittest discover -s tests` in the existing virtual environment. Coverage includes profile selection, separate inference settings, class mapping, empty and invalid inputs, legacy mode, ball deduplication, and ByteTrack recovery. Three-frame CPU checks also produced readable video for default annotation and for both `legacy` and `gsr` tracking.

The default CUDA run processed all 762 frames of `passes2.mp4` without processing errors. The 1920 × 1080 MP4 decoded fully at 59.99 source FPS and the JSONL contained consecutive frame indices 0–761. The run emitted all four internal track classes. ByteTrack emitted 335 people track records below 0.25, and all 609 observed ball tracks matched a dedicated ball detection at or above 0.25. There were 115 predicted ball frames and 38 frames without a ball track. The track, team, possession and pass fields were readable in every record. This clip had no detected camera cuts. The automated suite covers cut-related state resets.

Processing averaged 10.25 FPS over the frame loop, including decoding, tracking, overlays and writing, on the RTX 2060 SUPER. PyTorch peaked at 282.84 MiB allocated and 316 MiB reserved. The annotated video (`passes2-tracked.mp4`), JSONL, run summary and `verification.json` are under `E:\PitchLensStorage\outputs\final-detection-integration-v1`. These checks establish integration behavior on the clip. Tracking and analytics accuracy require ground truth.

The object detection baseline has been evaluated on held-out broadcast frames, with the limitations below. The prepared dataset and weights are local artifacts and are not committed to Git. The configured raw and processed paths can be changed per machine.

## Sources and attribution

- [Football Players Detection v10](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/10), Roboflow Universe, CC BY 4.0 as stated in the supplied export: 312 original 1920 × 1080 frames.
- [Football Object Detection v1](https://universe.roboflow.com/football-detect/football-object-detection/dataset/1), Roboflow Universe, CC BY 4.0 as stated in the supplied export: 1,150 640 × 640 images with provider-generated crop variants. All 242 distinct frame identifiers in this export also occur in v10.

The license statements above are from the dataset exports; rights to the underlying broadcast footage have not been independently verified.

Both export class mappings are verified against their metadata. The output mapping is `0: player`, `1: goalkeeper`, `2: referee`, `3: ball`. Goal polygons and corner boxes are removed. Images and boxes are validated, with sub-pixel edge overflows clipped only within a one-pixel tolerance. Conflicting retained-class labels on byte-identical images stop preparation.

The Foundation Reliability audit covers both baseline exports plus Wrkplace, IFAP and Football Player Referee and Goalkeeper Detection. Its inventory and split recommendations are in `E:\PitchLensStorage\outputs\dataset-audit-v1`. The current cumulative dataset download cap is 2.5 GB excluding pretrained weights. The baseline uses the partition described below.

[Detection candidate v1](DETECTION_CANDIDATE_V1.md) documents the revised split, reservation, and existing-checkpoint development evaluation for architectural review.

Detection Candidate v2 integrates the completed Wrkplace corrections and records the 960-pixel development comparison with production. [Candidate v2 training and evaluation](DETECTION_CANDIDATE_V2.md)

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

Supply the local official weights, a fresh run directory and the independently recorded digest of the approved training manifest. Set `$approvedManifestSha256` to that digest before running either command.

```powershell
.\.venv\Scripts\python.exe scripts\train_detection.py `
  --data data\processed\detection-v4\data.yaml `
  --manifest-sha256 $approvedManifestSha256 `
  --weights models\yolo26s.pt --run runs\detection-smoke --smoke
.\.venv\Scripts\python.exe scripts\train_detection.py `
  --data data\processed\detection-v4\data.yaml `
  --manifest-sha256 $approvedManifestSha256 `
  --weights models\yolo26s.pt --run runs\detection-baseline-640
```

Each run records input and manifest hashes, requested settings, GPU details, losses and validation metrics, per-epoch peak PyTorch GPU memory, and checkpoints. The selected batch and optimizer are stored in checkpoints. Periodic checkpoints and `best.pt` retain that state, and `--save-resumable` preserves the latest state before Ultralytics strips the finalized `last.pt`. Validation fitness selects `best.pt`. The figures depend on hardware and library versions, and deterministic settings do not guarantee bit-for-bit equivalence across platforms.

Ultralytics identifies its installed package license as AGPL-3.0. The PitchLens project license remains an open decision.

## Frame detector and evaluation

The low-level single-checkpoint `Detector` API loads weights once, selects CUDA when available and otherwise falls back to CPU. It accepts a nonempty OpenCV BGR `uint8` frame and returns `Detection` objects with `(left, top, right, bottom)` pixel coordinates in the original frame, class IDs, and confidence. Its constructor defaults to 960 pixels and 0.25 confidence. Confidence, per-class thresholds, input resolution and device are configurable. The IoU setting applies when the loaded model uses non-maximum suppression. Missing weights, incompatible class mappings and invalid frames are rejected.

```python
import cv2
from pitchlens.detector import Detector

detector = Detector("runs/detection-baseline-640/weights/best.pt", confidence=0.25)
frame = cv2.imread("path/to/authorized/frame.jpg")
for detection in detector.detect(frame):
    print(detection.class_id, detection.confidence, detection.xyxy)
```

Select the checkpoint, input resolution and operating threshold using validation only. Pass the independently recorded SHA-256 of the frozen manifest and use a fresh output directory outside the dataset. Scoring checks every selected image and label against the manifest and uses verified copies for AP and fixed-threshold metrics. The reserved test split requires separate approval after model selection.

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_detection.py `
  --data data\processed\detection-v4\data.yaml `
  --manifest-sha256 $approvedManifestSha256 `
  --weights runs\detection-baseline-640\weights\best.pt `
  --run runs\detection-development-check --split val --image-size 960 --confidence 0.25
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

At this threshold the model missed eight balls and produced three false ball detections. Twenty-one referee boxes were classified as players. The frame benchmark measured 54.06 FPS, with 128,099,840 bytes peak allocated and 148,897,792 bytes peak reserved by PyTorch during inference. Several provisional aggregate targets and the goalkeeper AP50 target were not met. The very small, correlated single-stadium test set, particularly its five goalkeeper boxes, means these numbers should not be generalized to new broadcasts without further independent evaluation. See [PROGRESS.md](PROGRESS.md#initial-video-check) for the initial unlabelled video check.

For a new local video, use `python scripts/annotate_video.py --input INPUT --output OUTPUT.mp4`. Use a fresh output stem. The script writes a silent MP4 with frame detections, a contact sheet and a JSON summary.
