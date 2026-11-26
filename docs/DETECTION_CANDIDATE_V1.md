# Detection candidate v1 checkpoint

Prepared for architectural review using `dataset-audit-v1`. The existing YOLO26s checkpoint is the baseline. The new test was reserved before the development run and has no model scores.

## Source and split decisions

- **Players v10** supplies the core training frames. Keep the legacy training and validation group assignments and exclude the complete frozen `08fd33` group, including both previously inspected frames and all Objects v1 variants. The original 25-image test remains a limited single-stadium reference.
- **Objects v1** adds training-only crops. The audit found that all 242 source frame families already occur in Players v10. Keep at most one visually distinct crop per retained training frame. Use image features to register the crop or mirrored crop against its v10 original. Require compatible retained-class boxes after projecting and clipping them to the crop. Remove corner and goal annotations.
- **Wrkplace** supplies Chelsea–Burnley for training, Persija Jakarta–PSS Sleman for development validation, and Crystal Palace–Arsenal for the new reserved test. Quarantine all remaining scouting clips. Review confirmed that the two `4dae79a9` clips link to different legacy venue groups. A Chelsea perceptual hit against SoccerNet was visually checked as PSG–Nantes footage from a different venue.
- **Wrkplace screening** keeps frames with at least eight annotated outfield players, samples Chelsea chronologically up to 60 frames per half before duplicate and camera screening, and removes manually reviewed alternate views, obstructive graphics and blurred transition frames. This is a wide-view candidate, with limited coverage of sparse-play situations and no labeled ball-absent Wrkplace frames.
- **IFAP** is a separate, unscored domain set covering seven source groups in its own manifest and YAML. **Football Player Referee and Goalkeeper** is excluded from training and evaluation because it largely repeats v10 and shares source frames with the original frozen test.

Use the supplied exports under their declared CC BY 4.0 terms with attribution. Underlying broadcast rights remain unverified. Training and development retain conservative legacy venue groups because authoritative match IDs are unavailable; adjacent frames and undetected transforms can still be correlated within a group. Crystal Palace–Arsenal is independent of the legacy checkpoint and candidate development data under the available provenance checks. It covers one match and one half, which limits any future generalization claim.

## Verified composition

Class columns count boxes. The main candidate has 537 images across eight source groups.

| Partition | Images | Groups | Player | Goalkeeper | Referee | Ball |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 371 | 4 | 6,648 | 229 | 722 | 341 |
| Development validation | 57 | 3 | 964 | 37 | 114 | 54 |
| Reserved independent test | 109 | 1 | 1,324 | 49 | 101 | 113 |
| IFAP domain evaluation | 462 | 7 | 6,621 | 308 | 683 | 465 |

Training contains 191 Players v10 originals, 67 Objects v1 crops and 113 Chelsea–Burnley frames. Development contains 37 legacy validation originals and 20 Persija–PSS Sleman frames. The test consists of 109 Crystal Palace–Arsenal frames. The separate IFAP set retains 462 of 739 exported images after perceptual pruning.

The builder kept all 141 images in the legacy frozen source group out of the candidate. There were zero cross-partition group, frame-key, file-hash, pixel-hash or audited near-image links among eligible sources and the protected group. It rejected 278 unverified crop alignments. No exact-image annotation conflicts were detected. These checks do not establish annotation completeness.

The reserved-test manifest SHA-256 is `22ecf793d02e8c4941100e069abde415181bbdd8a38484309a56a1512e0f1584`. The unchanged YOLO26s checkpoint SHA-256 is `29135b3b60a241f006f88f042db57b7c60ad3c7f161454683338eea01cb4b94b`.

## Development baseline

The existing checkpoint scored **67.99% mAP50** and **44.32% mAP50–95** on the 57 development images on the RTX 2060 SUPER, at 960 pixels. Macro precision was 68.99% and macro recall was 64.25% at confidence 0.25 and IoU 0.50. AP uses the evaluator's confidence sweep.

| Class | Boxes | AP50 | AP50–95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Player | 964 | 83.84% | 59.58% | 57.77% | 88.28% |
| Goalkeeper | 37 | 77.23% | 52.69% | 91.67% | 59.46% |
| Referee | 114 | 68.82% | 44.17% | 60.80% | 66.67% |
| Ball | 54 | 42.07% | 20.83% | 65.71% | 42.59% |

At the operating threshold, 31 of 54 balls were missed and 12 ball predictions were false positives. The run artifacts include confusion counts, AP metrics, evaluation inputs, and the single-frame inference benchmark in `runs/detection-candidate-v1-baseline-val/evaluation.json`. The development distribution differs from the original validation set, and 37 frames were reused from the original checkpoint-selection split. The new test and IFAP were left unscored.

The 20 Persija–PSS Sleman frames account for 565 of the 602 predicted players unmatched to any label at IoU 0.50. Player precision was 18.3% on this source, versus 96.9% and 89.8% on the two legacy validation groups. Four high-error frames showed multiple detections on individual athletes and predictions on graphics or off-pitch people. The run stores the group matrices in `validation-by-group.json` and the four annotated diagnostic frames in `validation-label-diagnostic.jpg`. This sample does not establish label completeness or isolate how much each factor contributed.

## Reproduce and verify

Run from the project root with the pinned project environment and new output paths. The audit and raw exports remain in external storage.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_detection_candidate.py `
  --raw E:\PitchLensStorage\datasets\raw `
  --audit E:\PitchLensStorage\outputs\dataset-audit-v1 `
  --frozen data\processed\detection-v4 `
  --weights runs\detection-baseline-640-v2\weights\best.pt `
  --output data\processed\detection-candidate-v1
.\.venv\Scripts\python.exe scripts\evaluate_detection.py `
  --data data\processed\detection-candidate-v1\data.yaml `
  --weights runs\detection-baseline-640-v2\weights\best.pt `
  --run runs\detection-candidate-v1-baseline-val --split val --image-size 960 --confidence 0.25
```

The local candidate directory contains `manifest.json` for train and validation, `reserved-test.json` for the locked holdout, `ifap-manifest.json` for domain evaluation, `provenance.json` for all inventory decisions, and `report.json` for input and manifest hashes, sources, counts and checks. The development YAML exposes train and validation only. The builder validates the existing checkpoint hash and all frozen test images and labels before and after preparation, and verifies the copied payload. Its duplicate checks use file SHA-256, decoded-pixel SHA-256 and the audit's perceptual distance of at most six. Near matches are conservatively pruned within each retained partition.

The full unit suite passed all 67 tests. An independent artifact check verified the 999 materialized image/label pairs against their manifests and raw sources, the 25 original frozen test pairs, split integrity, class counts, YAML isolation and the development run's input hashes.

## Recommended training configuration

After approval, start with the official pretrained YOLO26s weights and the existing [640-pixel configuration](../configs/detection_train.json). Run a CUDA and AMP smoke check on the RTX 2060 SUPER before the full run. Use seed 42, deterministic settings, automatic batch and optimizer, a 100-epoch ceiling, patience 20 and the checked-in augmentations. Start from the same pretrained weights as the legacy baseline to make the dataset comparison easier to interpret.

Use development validation for checkpoint and threshold selection. Compare per-class AP, ball recall by size, and goalkeeper and referee confusions by source. Consider a 960-pixel training comparison only after a memory smoke test and a specific gain target for small-ball recall. Keep the reserved test sealed until selection is final and report IFAP results separately.
