# Detection final evaluation v1

**Recommendation for architectural review. Retain production and keep Candidate v2 as a development checkpoint.** Candidate v2 improved all four classes on the Crystal Palace–Arsenal holdout. That gain comes from one match half. On the seven-group IFAP set, Candidate v2 lost 2.17 percentage points of mAP50–95, matched 18 fewer balls and added 194 goalkeeper false positives. The evidence supports further review of Candidate v2 for main-camera broadcasts before replacing production.

## Reserved evaluation

The developer authorized the 109-image Crystal Palace–Arsenal test and the 462-image IFAP domain set after the successful Sol review of `150d02e`. The original 25-frame frozen test stayed sealed. Both checkpoints and the source dataset files retained their verified bytes.

| Input | SHA-256 |
| --- | --- |
| Production `runs/detection-baseline-640-v2/weights/best.pt` | `29135b3b60a241f006f88f042db57b7c60ad3c7f161454683338eea01cb4b94b` |
| Candidate v2 `runs/detection-candidate-v2-640/weights/best.pt` | `cd05e022e02e5d4f6f6192fea438e8ebf5644a0e8fa575295366af9dc1e658eb` |
| Crystal Palace–Arsenal `reserved-test.json` | `22ecf793d02e8c4941100e069abde415181bbdd8a38484309a56a1512e0f1584` |
| IFAP `ifap-manifest.json` | `010925ac0e9777965de090518b92a11ae29c58524ae231ab8a6727c7d24f6b9b` |

Both source manifests under `data/processed/detection-candidate-v1` matched the recorded Candidate v1 digests. Before scoring, exact directory membership, per-file SHA-256, image decoding, class IDs, normalized box geometry and annotation counts were checked for all 571 pairs. No invalid or repeated label rows were found. Both checkpoints used the same manifest-derived, hash-ordered inputs. Temporary evaluator manifests exposed each reserved partition as `val` for grouped reporting and changed only the split and relative path. The reviewed evaluator verified the staged bytes, checked the paths and labels consumed by the AP loader, and isolated the aggregate and group caches. The original manifests, paired files and checkpoint hashes were verified again after all four runs.

Both checkpoints used 960-pixel inference on the RTX 2060 SUPER with PyTorch 2.11.0+cu128 and Ultralytics 8.4.218. AP used batch size 1, the standard 0.001 minimum confidence and an IoU sweep from 0.50 to 0.95 in 0.05 steps. Precision, recall and confusion used the deployed detector at confidence 0.25 and class-agnostic matching at IoU 0.50. The detector's inference IoU setting stayed at 0.70. The YOLO26s end-to-end heads are NMS-free. A matched wrong class contributes one FP for the predicted class and one FN for the labeled class.

### Crystal Palace–Arsenal

The single source group contains all 109 images and matches the aggregate metrics. Candidate v2 matched 73 of 113 ball labels and 39 of 49 goalkeeper labels, up from 3 and 14. Total false positives fell by 2,060 and false negatives by 766, largely through player detections. Ball and goalkeeper false positives rose by 11 and 14 respectively. Wrong-class matches also increased, including 21 players classified as balls and 27 players classified as goalkeepers.

Rates are percentages. Arrows show production → Candidate v2. Parentheses give the change in percentage points.

- mAP50 11.16 → 69.70 (+58.54) pp.
- mAP50–95 1.95 → 29.43 (+27.48) pp.
- FP 2532 → 472. FN 1076 → 310.
- Wrong-class matches 67 → 109. Unmatched predictions 2465 → 363. Unmatched labels 1009 → 201.

| Class | Labels | AP50 % | AP50–95 % | Precision % | Recall % | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| player | 1324 | 13.80 → 91.98 (+78.19) | 2.80 → 43.88 (+41.08) | 17.27 → 80.80 (+63.53) | 36.78 → 85.80 (+49.02) | 487 → 1136 | 2333 → 270 | 837 → 188 |
| **goalkeeper** | 49 | 22.93 → 73.94 (+51.01) | 3.34 → 36.01 (+32.67) | 24.14 → 40.21 (+16.07) | 28.57 → 79.59 (+51.02) | 14 → 39 | 44 → 58 | 35 → 10 |
| referee | 101 | 6.35 → 49.94 (+43.59) | 1.42 → 20.89 (+19.46) | 8.24 → 34.12 (+25.88) | 6.93 → 28.71 (+21.78) | 7 → 29 | 78 → 56 | 94 → 72 |
| **ball** | 113 | 1.57 → 62.96 (+61.39) | 0.23 → 16.95 (+16.72) | 3.75 → 45.34 (+41.59) | 2.65 → 64.60 (+61.95) | 3 → 73 | 77 → 88 | 110 → 40 |

Confusion counts are shown as production → Candidate v2. Rows are predictions and columns are labels.

| Predicted ↓ / Labeled → | Player | Goalkeeper | Referee | Ball | Background |
| --- | ---: | ---: | ---: | ---: | ---: |
| player | 487 → 1136 | 5 → 2 | 45 → 50 | 6 → 0 | 2277 → 218 |
| goalkeeper | 3 → 27 | 14 → 39 | 0 → 2 | 5 → 0 | 36 → 29 |
| referee | 3 → 7 | 0 → 0 | 7 → 29 | 0 → 0 | 75 → 49 |
| ball | 0 → 21 | 0 → 0 | 0 → 0 | 3 → 73 | 77 → 67 |
| background | 831 → 133 | 30 → 8 | 49 → 20 | 99 → 40 | 0 → 0 |

### IFAP

Candidate v2 reduced total false positives by 600 and false negatives by 334. Player AP50 improved by 8.45 points, while player AP50–95 fell by 2.48 points. Candidate v2 matched 82 of 465 ball labels, down from 100, with fewer ball false positives. Ball true positives fell in five of seven groups. Candidate v2 matched 72 of 308 goalkeeper labels, up from 42, but goalkeeper precision nearly halved and both goalkeeper AP measures fell. Neither checkpoint matched a goalkeeper on Sockers FC–Midwest United. Candidate v2's mAP50–95 fell in four of seven groups, with the largest drop on Manchester City–Genk.

- mAP50 33.89 → 34.55 (+0.66) pp.
- mAP50–95 20.26 → 18.09 (-2.17) pp.
- FP 3336 → 2736. FN 3094 → 2760.
- Wrong-class matches 472 → 621. Unmatched predictions 2864 → 2115. Unmatched labels 2622 → 2139.

| Class | Labels | AP50 % | AP50–95 % | Precision % | Recall % | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| player | 6621 | 70.69 → 79.14 (+8.45) | 45.39 → 42.91 (-2.48) | 62.17 → 71.28 (+9.11) | 70.34 → 75.03 (+4.70) | 4657 → 4968 | 2834 → 2002 | 1964 → 1653 |
| **goalkeeper** | 308 | 14.50 → 12.09 (-2.41) | 9.49 → 7.15 (-2.34) | 42.42 → 22.29 (-20.13) | 13.64 → 23.38 (+9.74) | 42 → 72 | 57 → 251 | 266 → 236 |
| referee | 683 | 27.71 → 31.23 (+3.52) | 17.11 → 17.39 (+0.28) | 37.17 → 34.03 (-3.14) | 26.94 → 28.55 (+1.61) | 184 → 195 | 311 → 378 | 499 → 488 |
| **ball** | 465 | 22.67 → 15.74 (-6.93) | 9.07 → 4.93 (-4.14) | 42.74 → 43.85 (+1.12) | 21.51 → 17.63 (-3.87) | 100 → 82 | 134 → 105 | 365 → 383 |

| IFAP source group | Images | mAP50 % | mAP50–95 % | Ball TP / labels | GK TP / labels | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Manchester_City_vs_Genk | 35 | 50.97 → 42.77 (-8.20) | 32.90 → 22.82 (-10.08) | 13 → 7 / 36 | 1 → 3 / 27 | 131 → 177 | 191 → 259 |
| FESAvsNASAfullgameveocamera1-24-2021 | 98 | 35.19 → 37.26 (+2.07) | 21.13 → 19.32 (-1.81) | 30 → 23 / 99 | 5 → 22 / 56 | 1201 → 1077 | 515 → 429 |
| AFC_04_vs_Legacy_Veo | 62 | 23.14 → 26.21 (+3.07) | 10.82 → 14.40 (+3.58) | 11 → 8 / 62 | 2 → 1 / 43 | 595 → 534 | 559 → 426 |
| CSAMonmouthvsBaltimoreArmour-MLSNext-Veo- | 62 | 51.63 → 44.95 (-6.67) | 30.15 → 24.75 (-5.40) | 14 → 7 / 62 | 28 → 28 / 46 | 142 → 224 | 518 → 477 |
| SockersFCvsMidwestUnited-NALU142010FullgameinVEO-2023- | 92 | 32.66 → 35.35 (+2.69) | 19.20 → 15.86 (-3.34) | 5 → 4 / 94 | 0 → 0 / 63 | 358 → 198 | 649 → 668 |
| PreDa_vs_2010_vienna | 46 | 21.00 → 36.31 (+15.31) | 12.17 → 19.97 (+7.80) | 4 → 7 / 47 | 0 → 3 / 36 | 268 → 132 | 436 → 274 |
| 2024-12-27VirginiaBeachCityFCvsRealColorado4kVeo | 67 | 46.62 → 53.88 (+7.25) | 30.67 → 31.49 (+0.82) | 23 → 26 / 65 | 6 → 15 / 37 | 641 → 394 | 226 → 227 |

Confusion counts are shown as production → Candidate v2. Rows are predictions and columns are labels.

| Predicted ↓ / Labeled → | Player | Goalkeeper | Referee | Ball | Background |
| --- | ---: | ---: | ---: | ---: | ---: |
| player | 4657 → 4968 | 85 → 47 | 187 → 218 | 6 → 4 | 2556 → 1733 |
| goalkeeper | 21 → 148 | 42 → 72 | 2 → 2 | 1 → 0 | 33 → 101 |
| referee | 133 → 145 | 35 → 53 | 184 → 195 | 0 → 0 | 143 → 180 |
| ball | 0 → 1 | 0 → 0 | 2 → 3 | 100 → 82 | 132 → 101 |
| background | 1810 → 1359 | 146 → 136 | 308 → 265 | 358 → 379 | 0 → 0 |

## Annotation quality and verification

- Crystal Palace–Arsenal uses 640 × 640 exports. Every image has a ball label, and four have two ball labels. Only 49 frames have goalkeeper labels and 88 have referee labels. Its single broadcast half and lack of ball-label-negative frames limit generalization and measurement of ball false alarms in ball-absent scenes.
- IFAP uses 1920 × 1080 exports from seven groups. Three frames have zero ball labels, five have multiple ball labels, and four ball boxes are at most eight pixels on their longest side in the source image. Goalkeeper labels occur on 307 frames. Differences in venues, camera, distance and annotation practices can affect the group scores. The seven sources and correlated frames limit the conclusions about new venues.
- A spot check of the first three hash-ordered frames per set showed unlabeled touchline figures and, on IFAP, people on adjacent pitches. Their eligibility depends on annotation scope. The spot check did not adjudicate all boxes or distinguish every semantic error from a labeling disagreement. A model prediction on an eligible unlabeled object would be counted as a false positive. Small shifts in box coordinates can change matches, especially for small balls and at high IoU. The reported metrics use the supplied frozen annotations.
- The first launch stopped before scoring because PyYAML emitted a list alias that Ultralytics rejects. Commit `5c440c2` serializes two equal lists and adds an actual Ultralytics parser check. The first Sol review flagged the test's import order. Commit `c6c4d34` corrected it. The read-only follow-up used `gpt-6.2-sol`, session `01a21865-9134-7b63-9bc0-6997f6a3b309`, and returned no blocking findings. All 95 project tests and all 12 targeted evaluation tests passed on the corrected revision.
- All four result files were reconciled against their matching input records, checkpoint and manifest digests, expected image and label counts, per-class AP means, and precision and recall derived from the matrices. Group matrices sum cell by cell to the aggregates. Each FP and FN total reconciles to unmatched counts plus wrong-class matches.

Machine-readable results are under `E:\PitchLensStorage\outputs\detection-final-evaluation-v1`. Each dataset contains the original and evaluator manifests, the evaluator YAML, and per-checkpoint `evaluation_input.json` and `evaluation.json`. The latter includes group-specific class metrics and confusion matrices. `comparison.json` includes candidate-minus-production deltas, count breakdowns and artifact SHA-256 values. `verification.json` records the review, tests and recommendation. The original `review-gate.json` records the earlier gate decision.

## Prior gate and remediation

The following sections document the historical review and development evidence that preceded the developer's authorization.

### Independent review

The established Codex CLI review completed on commit `b41e21331d6fb81d5db1e993e4c144f0f590449b` and the related detection implementation. The CLI identified the model as `gpt-6.2-sol`, session `01a21829-1e06-7c43-a780-97ce96d44573`. The reviewer judged the findings blocking for production promotion and allowed Candidate v2 to remain a development candidate. The evaluation gate stays closed because the current tooling cannot guarantee a frozen, identical frame and label set for both checkpoints.

| Severity | Affected component | Finding and consequence |
| --- | --- | --- |
| P2 | [Candidate v2 builder](../scripts/prepare_detection_candidate_v2.py#L44) | The export revision, frame identities, and totals do not pin the label bytes. A coordinate edit in a previously corrected file can pass the checks and silently create a different Candidate v2. Bind the full export payload or the expected final manifest to the approved version. |
| P2 | [Validation manifest check](../scripts/evaluate_detection.py#L84) | It verifies path membership without hashing the scored image and label files. A coordinate edit between model runs can leave the recorded manifest hash and class counts unchanged, while changing the comparison targets. Verify every scored file against the frozen manifest. |
| P2 | [Aggregate AP input](../scripts/evaluate_detection.py#L161) | Aggregate AP can recursively include a stray supported image. Group AP and confusion counts use manifest-checked top-level JPGs. This permits different image populations in one report. Use an explicit verified image list for aggregate AP, group AP, and fixed-threshold metrics. The reviewer found no evidence that this happened in the saved development runs. |
| P2 | [Resume documentation](DETECTION_CANDIDATE_V2.md#training) | The saved snapshot retains training state. The documented launcher starts a fresh run when given the snapshot as weights. Turning on resume in the launcher configuration can restore the original Ultralytics output directory and overwrite existing artifacts. An isolated continuation path and an actual state-restore check are still needed. |
| P3 | [Source-group evaluation](../scripts/evaluate_detection.py#L167) | Successive source-group runs can replace the full validation label cache with subset caches. This causes rescans and can write into the prepared dataset. Isolate cache writes for held-out evaluation. The reviewer treated this as an I/O issue and found no evidence that it affected the saved scores. |

The reviewer passed 28 targeted synthetic tests. A separate execution of the detection test subset passed 18 tests. The review corroborated the four-class mapping, the archived provenance checks, the approved training configuration, epoch 20 selection at 640 pixels, and the recorded comparison settings and confusion totals. Review of saved metadata leaves annotation completeness, residual leakage, byte identity between the prior evaluation passes, checkpoint continuation, and independent generalization unverified. The independent review ran no training or model evaluation.

### Existing development evidence

These are the previously reported development scores on 57 corrected frames from three source groups. This split was used for checkpoint selection. [Candidate v2 development report](DETECTION_CANDIDATE_V2.md)

| Metric | Production | Candidate v2 |
| --- | ---: | ---: |
| mAP50 | 67.06% | 78.90% |
| mAP50–95 | 43.58% | 50.82% |
| Ball recall | 42.59% (23/54) | 57.41% (31/54) |
| False positives | 679 | 167 |
| False negatives | 210 | 127 |

Persija accounted for most of the gain. Candidate v2 detected two fewer balls in each legacy group and missed 23 of 54 labeled balls overall. Both checkpoints missed every labeled Persija goalkeeper, five in total. These development results show uneven source performance and leave the candidate's generalization and suitability for production unresolved. The reserved Crystal Palace–Arsenal partition covers one half of one match, which will limit the reach of any eventual results. IFAP requires its own domain comparison.

The SHA-256 values below come from the prior development report. They were used as references for the review. No checkpoint or dataset payload was reopened for this gate.

| Artifact | Previously recorded SHA-256 |
| --- | --- |
| Production `runs/detection-baseline-640-v2/weights/best.pt` | `29135b3b60a241f006f88f042db57b7c60ad3c7f161454683338eea01cb4b94b` |
| Candidate v2 `runs/detection-candidate-v2-640/weights/best.pt` | `cd05e022e02e5d4f6f6192fea438e8ebf5644a0e8fa575295366af9dc1e658eb` |
| Candidate v2 development manifest | `eda48473212a3f4911b56ae9a061e1f938d8ccccc4961345c42bf62f6f666d1e` |

### Architectural recommendation

Keep production in place. Opening any reserved partition requires separate approval after infrastructure verification. The development scores above are historical.

The machine-readable gate decision and findings are in `E:\PitchLensStorage\outputs\detection-final-evaluation-v1\review-gate.json`. The raw local Codex review log is `outputs/detection-final-evaluation-v1-independent-review.log`.

### Infrastructure remediation

| Finding | Change and verification |
| --- | --- |
| Dataset builder integrity | The builder pins the original review manifest and the previously recorded byte-for-byte final Candidate v2 manifest. The final manifest binds every image and label SHA-256 to its source and frame. Identity, split, provenance and box-count checks run before publication. A fresh development rebuild matched all four existing metadata files and reproduced 371 training and 57 validation pairs. A regression test checks a coordinate edit that preserves the class and box count. |
| Evaluation input integrity | The evaluator requires the previously recorded manifest SHA-256. It rejects missing, changed, extra or linked files in the selected trees and copies verified bytes into the new run. Hash checks cover both image and annotation files. |
| Metric population | The aggregate AP loader receives a list of explicit files. Its accepted annotations and consumed frame identities are checked against that list. Source-group AP and fixed-threshold counts use exactly the manifest-derived subsets. Synthetic runs check aggregate and group coverage, including a skipped-frame failure. |
| Checkpoint continuation | The launcher reserves a fresh output directory and overrides the historical destination before Ultralytics can write artifacts. The setup-only CUDA smoke used a disposable copy of the epoch-40 checkpoint and verified model weights from EMA, EMA updates, 366 optimizer state entries, AMP scaler and early-stopping state. Ultralytics reconstructs the scheduler from the saved epoch and schedule settings. The next epoch was 41, with a verified learning rate of `0.000755` in each of the three parameter groups. The smoke ran zero training epochs. Before-and-after SHA-256 checks matched all 35 existing files in the Candidate v2 training and two development evaluation directories. |
| Cache isolation | The aggregate pass and each source group use independent temporary copies under the run directory. Two synthetic evaluations exercised the real Ultralytics dataset loader and checked matching populations and metric results, distinct caches, and preservation of a pre-existing dataset cache. |

The setup-only result is in `runs/detection-resume-smoke-repair-v1-verified/resume_state.json`. No YOLO26s performance metrics were generated during remediation. Repeated scoring was verified on synthetic fixtures.

All 88 tests passed with `python -m unittest discover -s tests` in the project environment.

#### Independent remediation review

**Blocking. Stop before using native continuation.** The read-only Codex CLI review used `gpt-6.2-sol` in session `01a21847-12d7-78b3-a0fc-3c0cc70e2557`. The reviewer cleared the builder pin, evaluation input verification, metric population consistency and cache isolation. The setup-only smoke established state restoration for the specific Candidate v2 checkpoint and development manifest used in that run. The reviewer identified the following gaps in native continuation.

| Severity | Finding |
| --- | --- |
| P2 | The launcher verifies the supplied manifest but does not compare its hash with the original checkpoint run’s recorded dataset hash. A different valid manifest can be paired with the saved optimizer and best fitness. [Resume input verification](../scripts/train_detection.py#L123) |
| P2 | The launcher accepts a checkpoint without saved early-stopping fields. Ultralytics initializes a new stopper in that case, resetting patience while the verification reports success. [Resume state verification](../scripts/train_detection.py#L82) |
| P3 | Ultralytics may designate a resumed checkpoint as the new run’s best on a tie with the historical best. That behavior conflicts with the documented strict-improvement rule. [Native continuation](../scripts/train_detection.py#L150) |

The first remediation stopped before commit or push. The local review log is `outputs/detection-repair-v1-independent-sol-review.log`.

### Final continuation remediation

| Finding | Repair |
| --- | --- |
| Checkpoint dataset binding | The training launcher requires an approved manifest SHA-256 and uses a verified temporary copy of the training and validation partitions. It embeds the manifest hash in each checkpoint. Continuation compares the supplied hash with the embedded value, checks the manifest bytes and verifies the staged image and label payload. A valid manifest from another dataset and a modified manifest are both rejected. |
| Complete early-stopping state | Checkpoints retain the best fitness and epoch, patience, epochs without improvement, and the imminent-stop flag. Continuation requires a complete, consistent record and rejects checkpoints whose patience or epoch budget is exhausted. |
| Strict best checkpoint selection | Selection compares finite fitness scores with a strict greater-than comparison. A continuation inherits the exact selected checkpoint, identified by the saved SHA-256 when it came from an earlier epoch. Equal or lower scores preserve the saved best fitness, epoch and bytes. Ultralytics final validation operates on a temporary copy of the selected checkpoint. |

The first review of this continuation repair cleared dataset binding and stopping-state restoration and found that a continuation needed to carry forward the historical best checkpoint. It also identified the pre-validation save path and the baseline command examples. The follow-up found that normal best-checkpoint replacement could remove the only copy needed by an earlier periodic checkpoint. The launcher now archives superseded best checkpoints by SHA-256, verifies and carries forward the required file, supports complete pre-validation state, and requires checkpoint saving. The loader checks accepted annotation rows against the manifest-verified labels, and both training examples document the required manifest argument.

The archived Candidate v2 snapshot was confirmed to fail the provenance check. Historical run metadata was preserved, and no checkpoint provenance was added retroactively. The positive CUDA smoke instead used a disposable checkpoint built from an untrained YOLO26s architecture and synthetic images, labels, and checkpoint state. It performed zero training epochs or performance evaluation. The reserved evaluation gate requires separate approval.

#### Final verification and independent review

- All 37 detection tests and all 95 project tests passed with `unittest discover`. The before-and-after inventory matched the SHA-256 of all 148 protected existing artifacts.
- The CUDA smoke resumed a synthetic periodic checkpoint after a later score superseded its historical best. It rejected a different valid dataset manifest and an incorrect sibling checkpoint, then restored the exact historical best from the archive. It verified the model from EMA, optimizer, scheduler, AMP scaler and early-stopping state. The next epoch was 4 and one epoch of patience remained. All checkpoint and score values in this smoke came from the fixture.
- The final read-only independent review was nonblocking. The Codex CLI identified the model as `gpt-6.2-sol` for session `01a2185d-0b04-7092-b484-159c3e592993`. It confirmed that the effective configuration guard rejects disabled checkpoint saving before the native trainer writes artifacts. It cleared all three continuation findings and confirmed the four earlier safeguards. The local review log is `outputs/detection-final-repair-gate-sol-review.log`.
