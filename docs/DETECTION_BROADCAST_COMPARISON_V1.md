# Detection comparison on main-camera footage

**Retain production as the main-camera default for architectural review.** Candidate v2 reduced several visible duplicate and graphic errors and recovered the ball in some production tracking failures. It also missed a visible ball that production caught on EGYPT, placed false balls on people and produced 55 fewer observed-ball outputs on that full clip. Only EGYPT could be compared end to end. The [reserved evaluation](DETECTION_FINAL_EVALUATION_V1.md) also showed a strong Crystal Palace–Arsenal gain alongside weaker IFAP ball recall and goalkeeper precision. The available evidence supports keeping Candidate v2 for further comparison.

## Inputs and settings

The relocated EGYPT original matched its Foundation manifest SHA-256. The MAN, FCB and LIV MP4s available in Foundation audit storage were annotated production outputs. Their original videos were unavailable in project storage. The surviving source JPEGs supported four same-frame detection checks per clip and six contiguous review strips. Each strip starts both trackers fresh. Re-encoded JPEG inputs and restarted track history limit comparability with the original full audit runs. LIV processing and inspection used only frame indices 0–544. All indices in this report are zero-based.

Both YOLO26s checkpoints matched the SHA-256 values in the reserved evaluation before and after the comparison. Both used CUDA, 960-pixel inference, the existing 0.10 person and 0.25 ball detection floors, detector IoU setting 0.70, and unchanged `configs/tracking.json`. The full EGYPT timing runs used the existing video runner with automatic team classification and no pitch mapping. All 928 frames of the production rerun exactly reproduced the archived tracks, team assignments, cut flags and ball trajectories. Production metrics and tracking evidence reuse the archived records. Timings come from the matched rerun and Candidate v2 run on the RTX 2060 SUPER.

The manual review kit contained a blank label export. The Foundation audit's four manually confirmed cut frames were used as reference. The still sheets support visual spot checks of visible people and balls. These examples provide neither exhaustive object annotations nor ground-truth identity trajectories.

## Visible detection differences

| Clip | Frame-level evidence |
| --- | --- |
| EGYPT | At 48, production boxed part of the score graphic as a player. Candidate v2 avoided it but put extra ball boxes on two red players. At 795, production found the ball near the goal area while Candidate v2 put its ball box on the red No. 10. Both missed the airborne red No. 11 and the diving goalkeeper. |
| MAN | At 133, both located the visible ball. Candidate v2 had fewer overlapping player boxes and also put two false ball boxes on the orange official. At 715, neither found the grounded goalkeeper and Candidate v2 missed the grounded player that production partially boxed. At 805, a 0.12 production goalkeeper candidate lay on the goal structure. Candidate v2 avoided that box, while both classified the visible goalkeeper as a player. |
| FCB | At 24, Candidate v2 labeled the far-side official as a goalkeeper and boxed the touchline coach as a player. At 237, Candidate v2 put a ball box on the yellow official. Both models labeled the purple goalkeeper as a player. At 625 and 673 both located the visible ball in the box sequence, with more overlapping and off-pitch player boxes in production. |
| LIV | Production boxed the score digit at 130 and 492 and the recording icon at 130, 247 and 358. Candidate v2 omitted those boxes. The verified production track for the recording icon persisted across the cut at 209, demonstrating that a stable ID can be wrong. At 247, both missed the visible ball and the two players contesting it in the close view. At 358, Candidate v2 added a 0.11 goalkeeper candidate on the cropped outfield player at the right edge. |

The stills show the detector candidates before tracking. Using a strict 0.70 IoU screen for same-class box pairs, the 16 stills yielded one overlapping pair per model, at EGYPT 48 for production and MAN 715 for Candidate v2. This screen misses loose, nested and cross-class duplicates. The visible clusters in production at MAN 133 and LIV 492 illustrate why pair counts alone understate the clutter.

## Tracking and processing performance

| EGYPT metric, all 928 frames | Production | Candidate v2 |
| --- | ---: | ---: |
| Ball output frames, observed / extrapolated / unavailable | 569 / 272 / 87 | 514 / 347 / 67 |
| Longest run without an observed ball | 46 | 28 |
| Longest run without any ball output | 31 | 16 |
| Distinct emitted outfield IDs | 179 | 91 |
| Outfield IDs emitted on one frame | 42 | 11 |
| Gaps followed by re-emission of the same outfield ID | 170 | 192 |
| Emitted outfield boxes | 13,175 | 12,071 |
| End-to-end processing FPS | 16.69 | 16.46 |
| Processing seconds | 55.60 | 56.39 |

Candidate v2 emitted fewer short-lived IDs, with more observed gaps within retained IDs. Fewer emitted boxes can reflect missed players as well as less clutter. These counts measure emitted track behavior. Identity fragmentation and switch rates require labeled identities. Candidate v2 had shorter runs without any ball output and relied more on extrapolation. The EGYPT sheet shows why this matters. At 366 and 924, Candidate v2 tracked the visible ball while production extrapolated in the wrong location, even though both detectors found the ball in the corresponding stills. At the manually confirmed cut at 925, both trackers extrapolated onto the celebration shot. One sequential timing run per checkpoint measures decode, inference, tracking, annotation and encode together. The observed 0.23 FPS difference does not establish a repeatable speed advantage.

| Fresh-start JPEG strip | Frames | Outfield IDs, production → v2 | Ball observed / extrapolated / unavailable, production → v2 |
| --- | --- | ---: | --- |
| MAN R1 | 791–808 | 9 → 5 | 5 / 13 / 0 → 5 / 13 / 0 |
| MAN R2 | 400–430 | 40 → 21 | 19 / 12 / 0 → 23 / 8 / 0 |
| FCB R1 | 216–246 | 32 → 24 | 9 / 17 / 5 → 10 / 18 / 3 |
| LIV R1 | 194–224 | 31 → 18 | 11 / 15 / 5 → 14 / 9 / 8 |
| LIV R2 | 343–373 | 41 → 19 | 11 / 5 / 15 → 16 / 0 / 15 |
| LIV R3 | 211–241 | 5 → 1 | 0 / 0 / 31 → 7 / 15 / 9 |

Neither model flagged the manually confirmed MAN 806 or LIV 209 and 358 cuts. Both carried a ball prediction over MAN 806 and LIV 209. At LIV 358, production chose the spurious ball near a runner and Candidate v2 chose the ball at the dribble. Three visually checked Candidate v2 observations at LIV 214, 231 and 238 fell on or near players while the ball was elsewhere. The overlapping LIV strips are reported separately. Their coverage, and the EGYPT coverage above, cannot establish ball accuracy. Comparable end-to-end performance on MAN, FCB and LIV remains unmeasured.

## Evidence files and verification

Local artifacts are under `outputs/detection-broadcast-comparison-v1`. `EGYPT-candidate-v2.mp4` is the full Candidate v2 video. The two MAN, one FCB and three LIV `*-candidate-v2-sample.mp4` files contain only the named strips. The four `*-side-by-side.jpg` sheets align source, production and Candidate v2 detections. `EGYPT-tracking-side-by-side.jpg` adds emitted track IDs and distinguishes observed and extrapolated balls. `LIV-ball-check.jpg` marks three misplaced Candidate v2 outputs and production's misplaced output at 358. `EGYPT-comparison.json` and `sample-evidence.json` retain the measured counts and per-frame sample evidence.

The full EGYPT comparison videos and all six sample videos decoded to their expected frame counts at 1920 × 1080. Sidecar indices and settings reconciled with the source manifest or declared strip. The checked artifacts exclude the reserved LIV interval. All 5 detector tests and 8 tracking tests passed. No checkpoints or configuration files were modified.
