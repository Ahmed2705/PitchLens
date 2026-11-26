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

## Annotation review and training gate

**Blocked before CUDA training smoke test and full training.** The review found missing visible players in development and training, a missing visible goalkeeper in development, and ball boxes with substantial padding in both sources. These defects affect training targets or how detections are scored. Candidate v1 requires manual correction and verification before training.

The review covered all 20 retained Persija–PSS Sleman development frames and 16 of 113 retained Chelsea–Burnley training frames, selected at eight evenly spaced chronological positions in each half. Each prepared image and label matched the development manifest hashes. The corresponding raw images and labels also matched their recorded hashes. Visual inspection used the exported 640 × 640 images with and without label overlays. Baseline predictions at 960 pixels, confidence 0.25 and IoU 0.50 were used to diagnose discrepancies. The existing checkpoint hash matched the recorded baseline.

| Reviewed source | Frames | Labeled players | Matched players | Player predictions unmatched to any label | Ball labels | Matched balls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Persija–PSS Sleman development | 20 | 226 | 128 | 565 | 20 | 2 |
| Chelsea–Burnley training sample | 16 | 207 | 136 | 233 | 16 | 1 |

The Persija player count reproduced the saved baseline confusion matrix. Where the cause could be judged visually, the review found several distinct failure modes.

- **Detector errors.** In Persija frame `10020`, the detector put at least eight player boxes on the Vidio watermark and several partial boxes on the torso and legs of the same labeled foreground player near (313, 575). In frame `13020`, it put a player box on a steward behind the advertising boards. In Chelsea first-half frame `30960`, it predicted a player on the Premier League watermark and a ball at the penalty spot while the match ball was next to a Chelsea player.
- **Missing or merged targets.** Persija frame `13020` also has a separate unboxed white-shirted player. Chelsea first-half frame `21360` has an unboxed Burnley player near the left edge and a corresponding player prediction with confidence 0.87. In Chelsea second-half frame `0720`, a single player box overlaps two distinct players by the near touchline.
- **Ball box extent.** The detector localized the visible match ball in Persija frame `10020` and Chelsea first-half frame `30960`, but the annotation extended well beyond the visible ball. The overlaps were 0.295 and 0.428 respectively, so each prediction counted as a false positive and each label as a miss at IoU 0.50. Chelsea second-half frame `29760` showed a similar extent disagreement at IoU 0.468, with more motion blur. These examples do not establish how much of the aggregate ball error is due to annotation.

Manual verification should start with these examples. Coordinates are approximate pixel locations in the exported frame, measured from the top-left.

| Source and frame | What to verify |
| --- | --- |
| Persija `10920`, `10980` | The white-shirted player on the ground near (322, 435) and (336, 435) has no player box in either frame. |
| Persija `12720` | The yellow goalkeeper near (437, 363) and the white-shirted player near (543, 401) have no separate boxes. |
| Persija `12780`, `13020` | A white-shirted player overlaps a red-shirted player without a separate box near (339, 514) and (460, 365), respectively. |
| Persija `13080` | White-shirted players near (170, 429) and (538, 385) lack individual boxes. |
| Persija `12720`–`14040` goal-area views | Check the role of the black-clad person near the goal line against the frame sequence before assigning any class. |
| Chelsea first half `21360` | The Burnley player near (19, 172) has no box. Also check the vertically padded ball box near (361, 516). |
| Chelsea first half `46920` and second half `0720` | Separate the overlapping players near (126, 355) and (105, 407), respectively, if both visible instances are confirmed. |
| Persija `10020`, Chelsea first half `30960` and second half `29760` | Review match-ball extent near (376, 250), (485, 377) and (443, 430), respectively. |

Before reopening the training gate, manually review the 20 Persija frames and the full 113-frame Chelsea subset using a consistent rule for visible portions of overlapping, fallen and clipped players, match officials and the active ball. Use the source footage or adjacent frames for ambiguous roles and ball locations where available. Preserve the original exports and write verified corrections to a versioned candidate. Recompute the 960-pixel baseline on its corrected development labels before comparing a new checkpoint. The examples above are a focused finding list, and the unmatched totals have not been adjudicated box by box.

The review files are local under `runs/detection-candidate-v1-annotation-review`. `frames.json` maps review IDs to manifest paths, coordinates and per-frame confusion counts. `P01.jpg` through `P20.jpg` and `C01.jpg` through `C16.jpg` show the image, labels and disputed predictions side by side. On the overlays, player labels are green, goalkeeper labels orange, referee labels cyan and ball labels white. Disputed predictions are magenta or violet. No labels were changed by the review. No candidate checkpoint or candidate metrics were produced. The reserved test and IFAP sets were not opened.

The repository's full `unittest` suite passed all 67 tests.

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

After the annotation gate passes, start with the official pretrained YOLO26s weights and the existing [640-pixel configuration](../configs/detection_train.json). Run a CUDA and AMP smoke check on the RTX 2060 SUPER before the full run. Use seed 42, deterministic settings, automatic batch and optimizer, a 100-epoch ceiling, patience 20 and the checked-in augmentations. Start from the same pretrained weights as the legacy baseline to make the dataset comparison easier to interpret.

Use development validation for checkpoint and threshold selection. Compare per-class AP, ball recall by size, and goalkeeper and referee confusions by source. Consider a 960-pixel training comparison only after a memory smoke test and a specific gain target for small-ball recall. Keep the reserved test sealed until selection is final and report IFAP results separately.
