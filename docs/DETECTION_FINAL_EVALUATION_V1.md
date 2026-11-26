# Detection final evaluation v1

**Stopped at the independent review gate. Keep the current production checkpoint and retain Candidate v2 for architectural review.** The 109-image Crystal Palace–Arsenal holdout, the 462-image IFAP domain set, and the original 25-frame frozen test stayed sealed. Evaluation input freezing and reserved scoring did not start.

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

Keep production in place. Address the annotation and evaluation input verification findings and approve an isolated evaluation path before opening either held-out set. The implementation, datasets, settings, and checkpoints remain as they were at the start of this review. No new model metrics or confusion matrices were generated.

The machine-readable gate decision and findings are in `E:\PitchLensStorage\outputs\detection-final-evaluation-v1\review-gate.json`. The raw local Codex review log is `outputs/detection-final-evaluation-v1-independent-review.log`.
