# Detection Candidate v2

Candidate v2 applies the completed Wrkplace annotation review to the Candidate v1 development split. The existing production checkpoint is the comparison baseline. Dataset payloads, checkpoints and generated results are stored locally.

## Dataset verification

The builder verified the completion marker, export revision 423 and all 133 exported label files. It matched each reviewed frame by source frame ID, split, image SHA-256 and the original raw source entry recorded in the Candidate v1 provenance. An independent readback verified all 428 image and label hashes, all 15 replacement files, and 413 byte-identical labels.

| Split | Images | Reviewed | Changed | Player boxes | Goalkeeper boxes | Referee boxes | Ball boxes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 371 | 113 | 5 | 6,651 | 229 | 722 | 341 |
| Development validation | 57 | 20 | 10 | 977 | 37 | 120 | 54 |

The export contains 2,106 boxes. All Candidate v2 labels have valid class IDs and normalized coordinates. Checks found zero missing labels, exact or perceptual duplicate images, cross-split source-group links, or cross-split frame keys. Protected comparisons used archived provenance metadata for the frozen source group, the reserved Crystal Palace–Arsenal test and IFAP. The protected images and labels stayed sealed. Candidate v1 and the annotation draft were left in place.

The new dataset is at `data/processed/detection-candidate-v2`. `manifest.json` lists all image-label pairs, `corrections.json` links the reviewed frames to their original and corrected hashes, and `report.json` records the inputs, checks and counts. The development manifest SHA-256 is `eda48473212a3f4911b56ae9a061e1f938d8ccccc4961345c42bf62f6f666d1e`.

## Training

The RTX 2060 SUPER passed the one-epoch CUDA and AMP smoke test with 19 training images and all 57 development images. The full run used the official pretrained YOLO26s checkpoint, 640-pixel inputs, CUDA 12.8 with Torch 2.11.0, AMP, seed 42, deterministic training and the checked-in augmentations. Automatic batch sizing selected 12. Automatic optimizer selection chose AdamW with an initial learning rate of 0.00125. Peak allocated and reserved CUDA memory were 3.43 GiB and 4.71 GiB respectively.

The run completed 40 epochs in about 8 minutes 34 seconds. Early stopping triggered after 20 epochs without improved validation mAP50–95. The selected epoch was 20 with **78.71% mAP50** and **46.05% mAP50–95** at 640 pixels. The training curves are in `runs/detection-candidate-v2-640/results.png` and the epoch data are in `results.csv` in the same directory.

- `weights/best.pt` is the evaluated Candidate v2 checkpoint. SHA-256 `cd05e022e02e5d4f6f6192fea438e8ebf5644a0e8fa575295366af9dc1e658eb`.
- `weights/last-resumable.pt` retains optimizer, AMP scaler, EMA and early-stopping state through epoch 40. SHA-256 `8c6c5101cea85607584e2f28ef31108a64b7daf489d57837677510c935ad55a4`. The checkpoint was loaded to verify its state.
- The automatically exported `weights/last.pt`, periodic epoch checkpoints, training logs, effective arguments and GPU memory history are preserved.

## Development evaluation

Both checkpoints used all 57 corrected development frames, CUDA, 960-pixel inputs and evaluation batch size 1. AP sweeps confidence. Precision, recall and confusion counts use confidence 0.25 with IoU 0.50 matching. A class confusion contributes a false positive to the predicted class and a false negative to the labeled class. The production checkpoint SHA-256 is `29135b3b60a241f006f88f042db57b7c60ad3c7f161454683338eea01cb4b94b`.

| Overall | Production | Candidate v2 | Change |
| --- | ---: | ---: | ---: |
| mAP50 | 67.06% | 78.90% | +11.85 pp |
| mAP50–95 | 43.58% | 50.82% | +7.23 pp |
| Macro precision | 69.09% | 83.79% | +14.70 pp |
| Macro recall | 63.28% | 72.21% | +8.93 pp |
| False positives | 679 | 167 | −512 |
| False negatives | 210 | 127 | −83 |

| Class | Labels | Precision, production → v2 | Recall, production → v2 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| Player | 977 | 58.18% → 88.88% | 87.72% → 94.06% | 616 → 115 | 120 → 58 |
| Goalkeeper | 37 | 91.67% → 92.31% | 59.46% → 64.86% | 2 → 2 | 15 → 13 |
| Referee | 120 | 60.80% → 65.41% | 63.33% → 72.50% | 49 → 46 | 44 → 33 |
| Ball | 54 | 65.71% → 88.57% | 42.59% → 57.41% | 12 → 4 | 31 → 23 |

| Source group | Images | mAP50, production → v2 | mAP50–95, production → v2 | Matched balls, production → v2 |
| --- | ---: | ---: | ---: | ---: |
| Persija Jakarta–PSS Sleman | 20 | 14.63% → 58.13% | 3.57% → 24.04% | 2/20 → 14/20 |
| Legacy `121364` | 19 | 90.20% → 87.26% | 63.97% → 64.71% | 12/16 → 10/16 |
| Legacy `40cd38` | 18 | 81.15% → 74.96% | 53.88% → 50.83% | 9/18 → 7/18 |

Most of the improvement came from Persija, where player false positives fell from 564 to 40. Both checkpoints missed all five labeled Persija goalkeepers. Candidate v2 missed two additional balls in each legacy group and lost mAP50 in both. The development split has 57 potentially correlated frames across three groups. It was used for checkpoint selection, and production previously used the 37 legacy frames for model selection. The independent test and IFAP stayed sealed, so this result supports architectural review of the candidate and leaves generalization unmeasured.

The full confusion matrices, per-class AP and source metrics are in `evaluation.json` under `runs/detection-candidate-v2-production-eval` and `runs/detection-candidate-v2-eval`. The combined machine-readable training and evaluation report is `runs/detection-candidate-v2-640/training_report.json`. Evaluation inputs and class counts were checked against each other, and all 76 unit tests passed. Production remains on its original checkpoint.

## Reproduction

Run from the project directory. Choose fresh output paths for repeat experiments.

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python scripts/prepare_detection_candidate_v2.py `
  --source data/processed/detection-candidate-v1 `
  --export runs/detection-candidate-v1-corrections/export-20261126T123832095581Z-beb0af0f `
  --output data/processed/detection-candidate-v2

$data = 'data/processed/detection-candidate-v2/data.yaml'
& $python scripts/train_detection.py --data $data --weights models/yolo26s.pt `
  --run runs/detection-candidate-v2-smoke --smoke
& $python scripts/train_detection.py --data $data --weights models/yolo26s.pt `
  --run runs/detection-candidate-v2-640 --save-resumable

& $python scripts/evaluate_detection.py --data $data --split val --image-size 960 `
  --confidence 0.25 --by-source-group --weights runs/detection-baseline-640-v2/weights/best.pt `
  --run runs/detection-candidate-v2-production-eval
& $python scripts/evaluate_detection.py --data $data --split val --image-size 960 `
  --confidence 0.25 --by-source-group --weights runs/detection-candidate-v2-640/weights/best.pt `
  --run runs/detection-candidate-v2-eval
```
