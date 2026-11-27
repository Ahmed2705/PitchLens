# Camera calibration and pitch mapping

Feature #04 is an opt-in local evaluation mode in the existing tracking runner. It uses the Roboflow Soccer example's YOLOv8 pose checkpoint, independently of the existing YOLO26s detector. The pose model predicts 32 ordered pitch landmarks at 640-pixel input. These include pitch corners, box geometry, penalty spots, halfway-line points and four points on the center circle. The ordered template follows the [reference configuration](https://github.com/roboflow/sports/blob/main/sports/configs/soccer.py). Virtual landmarks on lines are treated as model predictions like the visible intersections.

The Foundation Reliability audit reviewed the Roboflow field v18 keypoint export and the full SoccerNet Calibration 2023 validation archive. The reports are in `E:\PitchLensStorage\outputs\dataset-audit-v1`. SoccerNet validation is evaluation-only. Roboflow v18 shares the checkpoint's dataset project lineage. The [reliability comparison](#template-reliability-comparison) records the four corrected coordinates and the remaining calibration errors.

## Checkpoint and environment

Use only the existing `football-pitch-detection-v9.pt` obtained from the Google Drive file linked by the [official Roboflow setup file](https://github.com/roboflow/sports/blob/main/examples/soccer/setup.sh). The download is 140,212,306 bytes. Archive CRCs passed, and the local SHA-256 is `28f68f7c4056d6d9b137efd2e7ab5f3c494039380c63831649126ced25628b36`. The runner checks this fingerprint. Roboflow did not publish a checksum in the setup file. The reference [licensing notes](https://github.com/roboflow/sports/blob/main/examples/soccer/README.md#-license) identify YOLOv8 as AGPL-3.0 and the reference repository is MIT. The exact v9 checkpoint's licensing terms remain unresolved. Use the weights only for local evaluation pending clarification. Keep them outside Git and do not redistribute them.

The existing Python environment is sufficient. Local evaluation uses Ultralytics 8.4.218, PyTorch 2.11.0+cu128, OpenCV 5.1.0, and CUDA on the RTX 2060 SUPER. Set the security flags before Python starts. Ultralytics uses its supported weights-only loader and PyTorch is configured to reject unrestricted pickle loads. The pitch runner requires the Ultralytics flag. It verifies and loads a private temporary copy, then removes that copy. A missing checkpoint or fingerprint mismatch fails locally.

```powershell
$env:ULTRALYTICS_SAFE_LOAD = '1'
$env:TORCH_FORCE_WEIGHTS_ONLY_LOAD = '1'
$env:YOLO_AUTOINSTALL = 'false'
.\.venv\Scripts\python.exe scripts/track_video.py `
  --input 'E:\PitchLensStorage\footage\your-video.mp4' `
  --output 'E:\PitchLensStorage\outputs\your-new-pitch-video.mp4' `
  --weights runs/detection-baseline-640-v2/weights/best.pt `
  --pitch-weights 'E:\PitchLensStorage\models\football-pitch-detection-v9.pt' `
  --device cuda
```

For a landmark-only benchmark, use `scripts/benchmark_pitch_landmarks.py --input INPUT --weights CHECKPOINT --output NEW_DIRECTORY --device cuda` with the same environment flags. This writes frame samples with IDs and a JSON timing and landmark report. The first sample includes model warm-up.

## Geometry and outputs

By default the field is 105 by 68 meters. Use `--pitch-config` with a JSON object such as `{"length_m": 110, "width_m": 70}` when actual dimensions are known. Standard box depths and widths, penalty spots and center-circle radius stay fixed as the field dimensions change. Coordinates originate at model corner 1, increase along the length toward corner 25 and across the width toward corner 6. The default dimensions are assumptions for the two supplied clips. The model's landmark orientation can be ambiguous on a symmetric pitch, especially across camera cuts. Coordinates do not establish a team's attacking direction.

The estimator fits a pitch-to-image homography every frame with RANSAC and inverts it for mapping. It excludes low-confidence and image-edge landmarks, then checks inlier count, spatial support, whether the correspondences constrain a unique projective fit, pixel residual and local conditioning. The video runner also requires dispersed support, stability under landmark removal and alignment with visible straight paint. When successive predictions agree, it blends their projected inlier landmarks and checks the blend against the current landmarks and paint. Invalid frames clear the transform history. The [tracking cut signal](TRACKING.md#hard-camera-cuts) discards the previous transform before fitting the first frame of a new shot. A frame with sufficient current evidence can emit a fresh, unsmoothed fit. Otherwise its coordinates are unavailable. A planar homography assumes a flat field and does not model lens distortion.

The silent annotated MP4 includes a top-down pitch inset. Field players and goalkeepers are mapped using the bottom center of their boxes as an estimated ground contact. The mapper rejects boxes clipped at the sides or bottom. Filled markers fall within the current landmark support. Hollow markers use limited extrapolation. Coordinates outside the field or beyond the support margin are unavailable. Ball and referee tracks retain the existing output. Goalkeeper team remains unknown. Position by a goal may provide evidence for which end a goalkeeper defends, but linking that end to a team requires independent evidence. Nearby outfield players can be on either team, and the current team classifier does not establish team-to-end identity.

Each JSONL frame adds `calibration` with validity, rejection reason, landmark and inlier counts, inlier ratio, mean inlier landmark score, fit RMSE in pixels and whether smoothing was accepted. Landmark confidence is a model score, and reprojection RMSE is measured on the accepted fit's inliers. Neither measures absolute position accuracy. The frame also records `provenance` (`direct`, `temporal` or `unavailable`). Mapped tracks add `pitch` with `xy_m` and `status` (`valid`, `estimated` or `unavailable`). `unavailable` has `xy_m: null`. The summary records the pitch settings, fingerprint, frame counts by provenance and mapping counts. The original tracker output format is preserved when `--pitch-weights` is omitted.

## Local evaluation

Before integration, the checkpoint was benchmarked on uniformly spaced frames from the two supplied clips. At confidence 0.50, the desktop sample had 2–16 candidate landmarks on 16 frames. U19 had 0–16 on 15 decoded sample frames, and six samples had zero. The U19 frame count in the source metadata included one frame OpenCV could not decode. Median warm landmark inference was 40.0 ms for desktop and 35.4 ms for U19. Overlay inspection showed generally plausible landmarks on wide desktop views, false or displaced landmarks in some partial views, and repeated failures with U19 shadows, close framing and spectator occlusion. No annotated landmark ground truth was available.

| Full CUDA run | Desktop | Inter–Ajax U19 |
| --- | ---: | ---: |
| Frames with an accepted fit | 591 / 928 (63.7%) | 49 / 390 (12.6%) |
| Player or goalkeeper observations with valid coordinates | 2,849 | 196 |
| Observations with estimated coordinates | 2,046 | 263 |
| Observations with unavailable coordinates | 8,685 | 2,910 |
| Accepted fits with smoothing | 424 | 4 |
| Inlier RMSE, median / 95th percentile (px) | 3.10 / 4.60 | 2.31 / 3.15 |
| Mean inlier landmark score, median over accepted frames | 0.959 | 0.942 |
| End-to-end processing FPS | 11.67 | 14.31 |

The desktop fit followed wide camera pans from one end through midfield to the other. Availability dropped on close or partial views. On adjacent frames with valid positions for the same outfield track in the same segment, median and 95th-percentile displacement were 0.09 m and 0.53 m across 2,465 desktop pairs, and 0.58 m and 1.75 m across 79 U19 pairs. These values mix player motion, detector and tracker error, and calibration jitter. The U19 result is sparse and unstable. Actual position accuracy is unmeasured.

Both annotated videos decoded to the same number of frames as their JSONL. All emitted positions were within pitch bounds, and invalid frames emitted no positions. The original grayscale heuristic flagged zero cuts across these two historical runs. A synthetic cut through the tracker and calibrator incremented the segment, prevented smoothing on the cut frame, and allowed it again on the next frame with fresh support. A 32-frame desktop run with pitch mapping disabled exactly matched the original tracking and team fields in the mapped run.

On desktop, 266 of 269 mapped goalkeeper observations fell in a modeled penalty area, and none of the seven U19 goalkeeper observations could be mapped. This supports checking a goalkeeper's likely defending end in wide views. It does not link that end to a team, so all goalkeeper team labels remained unknown.

The verified MP4, JSONL and summary JSON use the stems `pitch-desktop-v2` and `pitch-u19-v2` under `E:\PitchLensStorage\outputs`. The initial landmark overlays and sample reports are in `pitch-benchmark-desktop-v1` and `pitch-benchmark-u19-v1` in the same directory.

## Template reliability comparison

The reference's ordered `vertices` array and v18's 32-element `flip_idx` agree on ordering, including the half-to-half pairs 11↔19 and 12↔20. The reference's display labels are a separate numbering scheme. Its numerical example puts the four vertices at the goal-area half-width, 9.16 m from the pitch centerline. The audited numbered annotation overlays put them where the arcs meet the penalty-area fronts. At a front 16.5 m from the goal line, the offset from the 11 m penalty spot is 5.5 m. The 9.15 m arc radius gives a transverse offset of `sqrt(9.15² − 5.5²) = 7.31249… m`. IDs 11 and 19 are on the lower-coordinate side, and IDs 12 and 20 are on the higher-coordinate side. The four y coordinates changed. The v9 training manifest is unavailable, so the reference, v18 semantics and frozen v9 comparisons provide complementary evidence. [Roboflow vertex order](https://github.com/roboflow/sports/blob/main/sports/configs/soccer.py) · [IFAB Law 1](https://theifab.com/laws/latest/the-field-of-play/)

On the 317 audited v18 images, an annotation-only check held out all four arc landmarks from one common robust control fit. It scored visibly annotated points on the 960 × 960 stretched exports. Only controls accepted by the existing estimator contribute.

| 1-based landmark | Eligible checks | Original median (px) | Corrected median (px) |
| --- | ---: | ---: | ---: |
| 11 | 168 | 10.73 | 2.52 |
| 12 | 146 | 17.73 | 2.40 |
| 19 | 167 | 12.12 | 1.97 |
| 20 | 135 | 19.76 | 2.14 |

One v9 inference per image or frame supplied identical observations to both variants. Both used the default 105 × 68 m field, the same estimator settings and independent smoothing histories. Still images start fresh. Video uses the current tracker cut signal, including the cut before zero-based frame 925.

| Measurement | Original | Corrected |
| --- | ---: | ---: |
| Roboflow fits from visible annotation coordinates | 296 / 317 | 313 / 317 |
| Roboflow fits from v9 predictions | 125 / 317 (39.4%) | 175 / 317 (55.2%) |
| Roboflow projected landmark error, p50 / p90, common accepted images (px) | 56.02 / 182.73 | 55.78 / 180.67 |
| Roboflow projected error for visible IDs missing from candidate input, p50 / p90, common accepted images (px) | 153.83 / 289.34 | 151.24 / 284.07 |
| Desktop/Egypt fits from v9 predictions | 591 / 928 (63.7%) | 829 / 928 (89.3%) |
| Desktop/Egypt inlier RMSE, p50 / p90 (px) | 3.10 / 4.21 | 3.23 / 4.40 |
| Inter–Ajax U19 fits from v9 predictions | 49 / 390 (12.6%) | 162 / 390 (41.5%) |
| Inter–Ajax U19 inlier RMSE, p50 / p90 (px) | 2.31 / 2.96 | 2.25 / 2.97 |

Roboflow's common-acceptance comparison uses 115 images and 1,824 visible labels, including 470 without a candidate prediction. Both transforms project the same physical check coordinates. Direct v9-to-label distance was 46.45 px at p50 and 100.29 px at p90 for 3,352 candidate predictions matched by ID among 4,802 visible labels. Labels with visibility 0 or 1 are excluded. The large direct errors identify a landmark-prediction problem before fitting. The resized v18 export and checkpoint share lineage, so the result is diagnostic and the effect of square stretching is unmeasured.

- In v18 frame `121364_4_3`, the predicted penalty-box and goal-area points are visibly displaced. The corrected transform follows those predictions and misses the painted lines despite a 1.86 px inlier RMSE. In `42ba34_2_10`, both templates accept the same seven inliers out of nine candidates with 1.74 px RMSE. The fit roughly follows midfield while projecting the near touchline into the stands. Its support is concentrated around midfield and the far side, and the low residual conceals the extrapolation error.
- Desktop frame 505 accepts both fits with eight inliers. The corrected projection drifts off the painted halfway line more visibly than the original. Frame 540 is newly accepted with an inlier RMSE of 3.76 px; the projected halfway line and circle miss the markings while most support lies to the right. Frame 623 passes with six of eight candidates and visibly misplaces the penalty box. These whole-pitch overlays diagnose the transform. The production mapper separately restricts player contacts to the inlier hull and its extrapolation margin.
- Replaying frames 505, 540 and 623 without smoothing reproduced the visible errors. At frame 623 the corrected support covered only 2.66% of the image, with six inliers from one end. In v18 frame `121364_4_3`, the median direct prediction error was 90.04 px across 13 visible matches, while feeding the visible annotations themselves yielded a 1.21 px inlier residual. Together these tests identify two failure modes: displaced input landmarks and a fit that follows local support but misprojects other markings.
- The corrected desktop fit rejected 92 frames for too few inliers and seven for too few candidates. At frame 914 both variants had seven candidates, but only four or five inliers. At the cut to a close-up on frame 925, there were just two candidates and both rejected. This distinguishes inconsistent correspondences from frames without enough candidate landmarks. Of the changes in acceptance, 245 frames gained a fit and seven lost one. The footage has no independent player or landmark labels, so coverage and visual examples cannot establish a change in metric accuracy.
- U19 gained a fit on 116 frames and lost one on three. In frame 106 the corrected fit accepts seven of ten candidates, but the projected box edge diverges from the painted line near the shadow. In frame 176 a spectator obscures the field and neither variant receives a pitch instance. Across the clip, the detector did not return exactly one pitch on 200 frames. One further frame had too few candidates, and 27 corrected fits had too few inliers. The shared tracker signaled six resets amid occlusions and view changes. Both variants processed the same 390 decoded frames. OpenCV reported 391, so the summary flags the mismatch.
- The desktop file is byte-identical to the original EGYPT audit input (SHA-256 `879b7996768f595afb7d3e17197409256fba2d1ada84d2e71b5c8637c80c6bc6`). The original U19 WebM survives in the retained Downloads files (SHA-256 `e446b7d911638e6d8a2c69ad4a6a64c154fe291ba1d8fb3053d5fa00aac4a58b`). The other audit views were available as previously annotated outputs or reports. The reserved LIV interval was left unopened.

Reproduce the development comparisons from the project root with the checkpoint environment flags above. Each command requires a new output directory and writes per-frame predictions, paired fit metadata, and a summary. Video records both the reported and decoded frame counts, flags a mismatch or unknown count, and writes selected paired overlays with predictions in yellow and projected ground markings in blue.

```powershell
$storage = 'E:\PitchLensStorage'
$downloads = 'C:\Users\Ahmad\Downloads'
$audit = "$storage\outputs\dataset-audit-v1"
$weights = "$storage\models\football-pitch-detection-v9.pt"
.\.venv\Scripts\python.exe -m scripts.evaluate_calibration roboflow `
  --input "$storage\datasets\calibration\roboflow-field-detection-v18" `
  --audit "$audit\roboflow-field-detection-v18.json" --weights $weights `
  --output outputs/calibration-roboflow-new
.\.venv\Scripts\python.exe -m scripts.evaluate_calibration video `
  --input "$storage\footage\Desktop 2026.11.25 - 15.59.30.02.mp4" `
  --weights $weights --output outputs/calibration-desktop-new
.\.venv\Scripts\python.exe -m scripts.evaluate_calibration video `
  --input "$downloads\Next_Generation._U19._Inter_vs_Ajax._Ajax_goal..webm" `
  --weights $weights --output outputs/calibration-u19-new
```

The measured summaries are in `outputs/calibration-reliability-v1-roboflow-canonical`, `outputs/calibration-reliability-v1-desktop-verified`, and `outputs/calibration-reliability-v1-u19`. The Roboflow inspection overlays are in `outputs/calibration-reliability-v1-roboflow` and the desktop smoothing replay is in `outputs/calibration-reliability-v1-desktop/static-checks.jpg`.

## Landmark support and paint validation

This development comparison keeps the corrected penalty-arc coordinates, v9 pose checkpoint and RANSAC settings from commit `4710164`. It adds two conservative acceptance checks.

- Require the inlier hull to cover at least 3% of both the image area and modeled pitch area. Refit once per image-hull vertex, withholding that vertex. Reject if its projection is more than three times the RANSAC pixel tolerance from the observation, or if the refit cannot predict the vertex. A few observations can otherwise determine the shape of the whole projection despite a low inlier residual.
- Sample perpendicular to the projected straight markings at 48 interior locations per marking. Look for a locally brighter stripe with green shoulders. Distances scale with image diagonal. Require support from at least two markings, each with at least four aligned samples covering at least 25% of its grass-supported samples and spanning at least 24 px at a 960 px image diagonal. A coherent displaced ridge spanning the same distance can veto a fit. A neighboring projected marking can explain the ridge when it has aligned paint that continues beyond the shared part by the same minimum span. Missed samples break that continuation when the gap between aligned samples exceeds 8 px at that image scale. The check uses the original video pixels and also checks any candidate smoothed fit. If current evidence fails, the runner returns unavailable coordinates and discards calibration history.

The comparison replayed stored v9 predictions from the corrected-template study at their native float32 precision, with identical pixels and cut flags for both variants. The baseline acceptance and accepted matrices reproduced the retained reports. Roboflow images and visible annotations were checked against retained hashes. Errors are pooled native pixels at p50 and p90, using the same physical template for both variants.

| Measurement | Corrected baseline | With validation |
| --- | ---: | ---: |
| Roboflow v18 accepted | 175 / 317 (55.2%) | 43 / 317 (13.6%) |
| Desktop accepted | 829 / 928 (89.3%) | 143 / 928 (15.4%) |
| Inter–Ajax U19 accepted | 162 / 390 (41.5%) | 47 / 390 (12.1%) |
| Roboflow visible landmark error on each variant's accepted images (px) | 55.99 / 166.33 | 48.06 / 123.91 |
| Roboflow visible landmark error for IDs absent from the candidate input (px) | 141.95 / 278.56 | 111.87 / 203.98 |
| Roboflow missing-candidate median error above 50 px, among accepted eligible images | 108 / 118 | 26 / 27 |
| Roboflow failures above 50 px or rejected fits, among all eligible images | 230 / 240 | 239 / 240 |

The 43 common accepted Roboflow fits have identical projections. The aggregate error improvement comes from rejecting fits. Eligibility requires at least three visible labels whose IDs were absent from the candidate input. The new gates rejected 82 of the 108 eligible baseline fits above the descriptive 50 px threshold and 9 of the 10 at or below it. Large-error fits made up 96.3% of eligible accepted fits compared with 91.5% for the baseline. Baseline fits subsequently dropped had missing-candidate errors of 152.31 / 287.73 px across 544 labels. The threshold is used for reporting only.

| Documented failure case | Baseline inlier RMSE | Outcome with validation |
| --- | ---: | --- |
| Desktop 505, halfway line drift | 4.16 px | Rejected for unstable support |
| Desktop 540, displaced halfway line and circle | 3.76 px | Rejected for unstable support |
| Desktop 623, displaced penalty box | 4.09 px | Rejected for concentrated support |
| v18 `42ba34_2_10`, near touchline projects into the stands | 1.74 px | Rejected for insufficient paint |
| v18 `121364_4_3`, displaced box predictions | 1.86 px | Still accepted |

Paint rejected 482 desktop fits after the geometry checks, including frames 0, 144 and 421 where selected lines visibly diverge despite close fits to the landmarks. Some plausible fits can also be rejected when paint is shadowed, clipped, occluded or weak. The check can miss isolated or severely displaced lines when other markings provide enough nearby evidence. It samples straight lines and cannot resolve landmark identity, lens distortion or orientation ambiguity. The footage has no independent landmark annotations, so its acceptance and visual checks leave position accuracy unmeasured. The retained v18 errors and reduced coverage keep this an opt-in local evaluation.

Use the development commands above with `--reliability` and fresh output directories to reproduce the comparison with new inference. To render the target desktop examples add `--overlay-frame 505 --overlay-frame 540 --overlay-frame 623`. The replay summaries, per-image and per-frame records, and desktop and U19 overlays are under `outputs/calibration-validation-milestone-*`. The desktop reset remains at frame 925, with too few landmarks for a fresh fit. U19 has the same six resets and the same 390 decoded frames of 391 reported.

The CUDA desktop runner output decoded to 928 frames and matched the final replay's fit validity, rejection reasons, smoothing and cut flags throughout. The review fixes produced identical desktop replay matrices. The runner emitted 404 valid, 405 estimated and 12,771 unavailable player or goalkeeper observations. Every rejected frame returned null pitch coordinates and every emitted position was within the modeled field. The runner artifacts use the stem `outputs/calibration-validation-reviewed-runner-desktop`. The full local suite passed 127 tests.

Independent GPT-6.2 Sol review of `e8cca69` identified the need for a shared error denominator, explicit handling of undecoded requested overlays, and protection against confusing neighboring markings. Follow-up reviews verified the fixes and exercised shifted, faint, detached and sparsely sampled paint. The final review of `9df7ad2` passed 19 targeted tests (session `01a21960-8919-7f00-87ef-5ee6a144cdf8`). SoccerNet remained sealed throughout this cycle.

## Reserved SoccerNet evaluation

The protocol and scorer are ready for architectural review. SoccerNet predictions and scores were left uncomputed. The retained local `valid.zip` matched publisher SHA-256 `9b8846782e16297a64a9f4c8e3b5bf32cfa62037dc735be79e168ada3aac0aa4`. Its index and member paths agree on 3,212 image–label pairs from 55 matches, including both official rows `01463.jpg` and `01464.jpg`. The scorer reads pixels, labels and match metadata directly from the verified archive. Keep the partition outside training, threshold selection, model selection and calibration changes. The full v9 training inventory is unavailable. [SoccerNet annotations and challenge evaluation](https://github.com/SoccerNet/sn-calibration#evaluation)

- Score one image at a time with independent state, the same predictions per variant and the same physical 105 × 68 m geometry for scoring. Use ground-plane markings by semantic name. Do not derive landmark coordinates from clipped line endpoints. Exclude elevated and unknown classes. Drop nonfinite points and overflow beyond 1 native pixel, clamp smaller overflow, and deduplicate samples within each marking. A single usable point can score an existing fit. Record exclusions and retained support per image.
- Project visible segments in native 960 × 540 coordinates. Sample the center circle every degree and each penalty arc at 111 angles including both endpoints. Clip to the camera-facing viewport. Measure each annotation point's Euclidean distance to the nearest point on the projected marking of the same name. Test both global pitch orientations (0° and 180°), choose the one with more points within 5 px, then lower summed distance. A marking without visible projected segments counts as a miss. Report null for a quantile that falls among failed projections.
- Report accepted coverage and p50/p90 pixels, including the common accepted subset. Report the fraction within 5 px both among accepted fits and across all eligible points, counting fit rejection as a miss in the latter. Keep per-match summaries and event identifiers. For uncertainty in a later report, resample complete matches with replacement, using the same draws for both templates, to avoid treating repeated event frames as independent. This measures planar image alignment. The official SoccerNet 3D camera format and completeness × JaC@5 score require a camera model with elevated goal geometry.

```powershell
$soccerNetArchive = "$downloads\valid.zip"
.\.venv\Scripts\python.exe -m scripts.evaluate_calibration soccernet `
  --input $soccerNetArchive --weights $weights `
  --output outputs/calibration-soccernet-new
```

The full local suite passed 117 tests. Independent GPT-6.2 Sol review of `622165c` found two issues in the evaluator, concerning the provenance of SoccerNet labels and metadata and the handling of short video decodes. Follow-up reviews verified those fixes in `6078f1c` and the explicit frame-limit failure behavior in `5ab56c7`. The final review passed all 21 targeted tests (session `01a21944-9c04-72d3-8ca7-f1e5cc96b5eb`).

## Temporal recovery review

Strict direct calibration remains the runner default. Add `--pitch-temporal` to the existing pitch-enabled tracking command for local evaluation. `--pitch-config` supports `recovery_seconds` (default 0.5, zero disables recovery), `recovery_min_tracks` (12), `recovery_min_inliers` (10) and `recovery_min_inlier_ratio` (0.70). The evaluator's `video --temporal` mode compares it with strict direct calibration using one v9 prediction per frame for both.

When direct fitting fails, Lucas–Kanade tracks corners from grass-rich patches inside the last direct fit's support hull. Forward–backward consistency, RANSAC consensus, image coverage and motion bounds reject weak fits. RANSAC fits a similarity for translation, zoom and roll. The current pitch-to-image projection is the anchor projection composed with anchor-to-current motion. Its inverse maps image contacts to the pitch. Recovery rechecks local projection geometry and current straight paint, and limits support to the overlap between the transported anchor hull and current motion inliers. Existing contact bounds and extrapolation checks still apply.

Each attempt estimates motion from the last direct frame, which avoids chaining transforms or refreshing the age with a recovered fit. Age counts elapsed frames at source FPS. A failed recovery, expiration or cut discards the anchor, and a new direct fit immediately takes priority. On recovered frames the JSONL reports `recovery_age_seconds`, `motion_inliers` and `motion_rmse_px`. The landmark counts describe the failed direct attempt and landmark reprojection RMSE is null.

The comparison re-decoded the hash-verified source clips using the archived v9 predictions at the same strict settings. It reproduced the archived cut flags, direct decisions and homographies, and the enabled and strict-only branches returned identical direct transforms. Local summaries, overlays and the replay script are under `outputs/calibration-temporal-replay-*`.

| Clip | Decoded frames | Direct | Recovered | Unavailable with recovery | Strict coverage | Coverage with recovery | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Desktop | 928 | 143 | 101 | 684 | 15.4% | 26.3% | 10.9 pp |
| Inter–Ajax U19 | 390 | 47 | 40 | 303 | 12.1% | 22.3% | 10.3 pp |

- U19 metadata reported 391 frames, as in the archived run. Recovery was unavailable on and immediately after the desktop cut at frame 925 and the six U19 cuts at frames 275, 312, 314, 353, 358 and 361. The longest accepted recoveries were 0.20 s and 0.50 s respectively. Desktop frames 505, 540 and 623 stayed unavailable.
- Direct landmark fit RMSE (p50/p90) stayed 2.98/4.12 px on desktop and 2.59/3.00 px on U19. Recovery motion-inlier RMSE was 1.25/1.66 px and 0.56/0.89 px respectively. These residuals measure agreement with the fit inputs. Independent ground-truth projection error in pixels or meters is unavailable for both clips. A synthetic known pan/zoom test verified three projected positions to within 0.10 m per axis.
- On neighboring recovered frames, median projected movement of common in-support pitch probes had p50/p90 of 0.04/9.53 px across 85 desktop pairs and 7.09/17.13 px across 26 U19 pairs. This includes actual camera motion. One of the three desktop returns to direct calibration with overlapping support had a 64.1 px median step. The ten comparable U19 returns had a p90 of 19.4 px.
- Overlay spot checks showed the halfway line generally followed the paint at desktop frame 277, with the near touchline offset. The penalty area was visibly misaligned at frames 607 and 878 and at U19 frames 27, 80 and 166. The limited paint test can accept partial alignment, and a recovered projection can inherit errors from its direct anchor. A similarity also approximates perspective changes.
- Mean calibration-only time rose from 3.27 to 4.49 ms/frame on desktop and from 1.34 to 3.74 ms/frame on U19 in this paired replay. These timings exclude decoding, inference, tracking and rendering. The CUDA tracker smoke run completed 32 frames (2 direct, 12 recovered and 18 unavailable) and the CUDA evaluator smoke run completed 16 frames. All 133 automated tests passed.

Feature #04 was previously approved with limitations for the MVP. Camera calibration and dependent features remain under scope review. Strict direct calibration remains the default, with 15.4% frame coverage on the desktop clip and 12.1% on Inter–Ajax U19. Temporal recovery remains experimental and opt-in. Temporally recovered coordinates are excluded from player speed, distance and heatmaps. Missing coordinates remain unavailable. Independent projection ground truth is unavailable.
