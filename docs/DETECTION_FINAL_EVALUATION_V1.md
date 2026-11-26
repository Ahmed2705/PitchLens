# Detection final evaluation v1

**The reserved evaluation gate remains closed.** Keep the current production checkpoint and retain Candidate v2 for architectural review. The 109-image Crystal Palace–Arsenal holdout, the 462-image IFAP domain set, and the original 25-frame frozen test stayed sealed.

## Independent review

The established Codex CLI review completed on commit `b41e21331d6fb81d5db1e993e4c144f0f590449b` and the related detection implementation. The CLI identified the model as `gpt-6.2-sol`, session `01a21829-1e06-7c43-a780-97ce96d44573`. The reviewer judged the findings blocking for production promotion and allowed Candidate v2 to remain a development candidate. The evaluation gate stays closed because the current tooling cannot guarantee a frozen, identical frame and label set for both checkpoints.

| Severity | Affected component | Finding and consequence |
| --- | --- | --- |
| P2 | [Candidate v2 builder](../scripts/prepare_detection_candidate_v2.py#L44) | The export revision, frame identities, and totals do not pin the label bytes. A coordinate edit in a previously corrected file can pass the checks and silently create a different Candidate v2. Bind the full export payload or the expected final manifest to the approved version. |
| P2 | [Validation manifest check](../scripts/evaluate_detection.py#L84) | It verifies path membership without hashing the scored image and label files. A coordinate edit between model runs can leave the recorded manifest hash and class counts unchanged, while changing the comparison targets. Verify every scored file against the frozen manifest. |
| P2 | [Aggregate AP input](../scripts/evaluate_detection.py#L161) | Aggregate AP can recursively include a stray supported image. Group AP and confusion counts use manifest-checked top-level JPGs. This permits different image populations in one report. Use an explicit verified image list for aggregate AP, group AP, and fixed-threshold metrics. The reviewer found no evidence that this happened in the saved development runs. |
| P2 | [Resume documentation](DETECTION_CANDIDATE_V2.md#training) | The saved snapshot retains training state. The documented launcher starts a fresh run when given the snapshot as weights. Turning on resume in the launcher configuration can restore the original Ultralytics output directory and overwrite existing artifacts. An isolated continuation path and an actual state-restore check are still needed. |
| P3 | [Source-group evaluation](../scripts/evaluate_detection.py#L167) | Successive source-group runs can replace the full validation label cache with subset caches. This causes rescans and can write into the prepared dataset. Isolate cache writes for held-out evaluation. The reviewer treated this as an I/O issue and found no evidence that it affected the saved scores. |

The reviewer passed 28 targeted synthetic tests. A separate execution of the detection test subset passed 18 tests. The review corroborated the four-class mapping, the archived provenance checks, the approved training configuration, epoch 20 selection at 640 pixels, and the recorded comparison settings and confusion totals. Review of saved metadata leaves annotation completeness, residual leakage, byte identity between the prior evaluation passes, checkpoint continuation, and independent generalization unverified. The independent review ran no training or model evaluation.

## Existing development evidence

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

## Architectural recommendation

Keep production in place. Opening any reserved partition requires separate approval after infrastructure verification. The development scores above are historical.

The machine-readable gate decision and findings are in `E:\PitchLensStorage\outputs\detection-final-evaluation-v1\review-gate.json`. The raw local Codex review log is `outputs/detection-final-evaluation-v1-independent-review.log`.

## Infrastructure remediation

| Finding | Change and verification |
| --- | --- |
| Dataset builder integrity | The builder pins the original review manifest and the previously recorded byte-for-byte final Candidate v2 manifest. The final manifest binds every image and label SHA-256 to its source and frame. Identity, split, provenance and box-count checks run before publication. A fresh development rebuild matched all four existing metadata files and reproduced 371 training and 57 validation pairs. A regression test checks a coordinate edit that preserves the class and box count. |
| Evaluation input integrity | The evaluator requires the previously recorded manifest SHA-256. It rejects missing, changed, extra or linked files in the selected trees and copies verified bytes into the new run. Hash checks cover both image and annotation files. |
| Metric population | The aggregate AP loader receives a list of explicit files. Its accepted annotations and consumed frame identities are checked against that list. Source-group AP and fixed-threshold counts use exactly the manifest-derived subsets. Synthetic runs check aggregate and group coverage, including a skipped-frame failure. |
| Checkpoint continuation | The launcher reserves a fresh output directory and overrides the historical destination before Ultralytics can write artifacts. The setup-only CUDA smoke used a disposable copy of the epoch-40 checkpoint and verified model weights from EMA, EMA updates, 366 optimizer state entries, AMP scaler and early-stopping state. Ultralytics reconstructs the scheduler from the saved epoch and schedule settings. The next epoch was 41, with a verified learning rate of `0.000755` in each of the three parameter groups. The smoke ran zero training epochs. Before-and-after SHA-256 checks matched all 35 existing files in the Candidate v2 training and two development evaluation directories. |
| Cache isolation | The aggregate pass and each source group use independent temporary copies under the run directory. Two synthetic evaluations exercised the real Ultralytics dataset loader and checked matching populations and metric results, distinct caches, and preservation of a pre-existing dataset cache. |

The setup-only result is in `runs/detection-resume-smoke-repair-v1-verified/resume_state.json`. No YOLO26s performance metrics were generated during remediation. Repeated scoring was verified on synthetic fixtures.

All 88 tests passed with `python -m unittest discover -s tests` in the project environment.

### Independent remediation review

**Blocking. Stop before using native continuation.** The read-only Codex CLI review used `gpt-6.2-sol` in session `01a21847-12d7-78b3-a0fc-3c0cc70e2557`. The reviewer cleared the builder pin, evaluation input verification, metric population consistency and cache isolation. The setup-only smoke established state restoration for the specific Candidate v2 checkpoint and development manifest used in that run. The reviewer identified the following gaps in native continuation.

| Severity | Finding |
| --- | --- |
| P2 | The launcher verifies the supplied manifest but does not compare its hash with the original checkpoint run’s recorded dataset hash. A different valid manifest can be paired with the saved optimizer and best fitness. [Resume input verification](../scripts/train_detection.py#L123) |
| P2 | The launcher accepts a checkpoint without saved early-stopping fields. Ultralytics initializes a new stopper in that case, resetting patience while the verification reports success. [Resume state verification](../scripts/train_detection.py#L82) |
| P3 | Ultralytics may designate a resumed checkpoint as the new run’s best on a tie with the historical best. That behavior conflicts with the documented strict-improvement rule. [Native continuation](../scripts/train_detection.py#L150) |

The first remediation stopped before commit or push. The local review log is `outputs/detection-repair-v1-independent-sol-review.log`.

## Final continuation remediation

| Finding | Repair |
| --- | --- |
| Checkpoint dataset binding | The training launcher requires an approved manifest SHA-256 and uses a verified temporary copy of the training and validation partitions. It embeds the manifest hash in each checkpoint. Continuation compares the supplied hash with the embedded value, checks the manifest bytes and verifies the staged image and label payload. A valid manifest from another dataset and a modified manifest are both rejected. |
| Complete early-stopping state | Checkpoints retain the best fitness and epoch, patience, epochs without improvement, and the imminent-stop flag. Continuation requires a complete, consistent record and rejects checkpoints whose patience or epoch budget is exhausted. |
| Strict best checkpoint selection | Selection compares finite fitness scores with a strict greater-than comparison. A continuation inherits the exact selected checkpoint, identified by the saved SHA-256 when it came from an earlier epoch. Equal or lower scores preserve the saved best fitness, epoch and bytes. Ultralytics final validation operates on a temporary copy of the selected checkpoint. |

The first review of this continuation repair cleared dataset binding and stopping-state restoration and found that a continuation needed to carry forward the historical best checkpoint. It also identified the pre-validation save path and the baseline command examples. The follow-up found that normal best-checkpoint replacement could remove the only copy needed by an earlier periodic checkpoint. The launcher now archives superseded best checkpoints by SHA-256, verifies and carries forward the required file, supports complete pre-validation state, and requires checkpoint saving. The loader checks accepted annotation rows against the manifest-verified labels, and both training examples document the required manifest argument.

The archived Candidate v2 snapshot was confirmed to fail the provenance check. Historical run metadata was preserved, and no checkpoint provenance was added retroactively. The positive CUDA smoke instead used a disposable checkpoint built from an untrained YOLO26s architecture and synthetic images, labels, and checkpoint state. It performed zero training epochs or performance evaluation. The reserved evaluation gate requires separate approval.

### Final verification and independent review

- All 37 detection tests and all 95 project tests passed with `unittest discover`. The before-and-after inventory matched the SHA-256 of all 148 protected existing artifacts.
- The CUDA smoke resumed a synthetic periodic checkpoint after a later score superseded its historical best. It rejected a different valid dataset manifest and an incorrect sibling checkpoint, then restored the exact historical best from the archive. It verified the model from EMA, optimizer, scheduler, AMP scaler and early-stopping state. The next epoch was 4 and one epoch of patience remained. All checkpoint and score values in this smoke came from the fixture.
- The final read-only independent review was nonblocking. The Codex CLI identified the model as `gpt-6.2-sol` for session `01a2185d-0b04-7092-b484-159c3e592993`. It confirmed that the effective configuration guard rejects disabled checkpoint saving before the native trainer writes artifacts. It cleared all three continuation findings and confirmed the four earlier safeguards. The local review log is `outputs/detection-final-repair-gate-sol-review.log`.
