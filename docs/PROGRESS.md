# Development progress

PitchLens has an approved dual-model detection default and a verified local tracking integration. Ground-truth tracking evaluation remains outstanding.

## Automatic team classification upgrade

- Automatic team classification now fits SigLIP embeddings with UMAP and two-cluster K-means. Track assignments use fixed clip-wide Team A and Team B identities and allow unknowns. The LAB classifier remains available through `--team-mode color`. [Model and usage](TRACKING.md#automatic-team-classification)
- All 208 tests passed. The CUDA smoke test exported all 762 `passes2.mp4` frames at 8.18 processing FPS after 20.98 seconds of team setup. Player and ball tracking fields matched the prior verified output. Manual review of 18 outfield crops found 16 correct, zero incorrect and two unknown. [Smoke-test results and limitations](TRACKING.md#integration-verification)

## BoT-SORT people tracking integration

- BoT-SORT is the default people tracker, with sparse optical-flow camera compensation and ReID disabled. ByteTrack remains selectable with `--people-tracker bytetrack`. Existing tracking thresholds, the ball tracker, and camera-cut resets are preserved.
- All 203 automated tests passed. The default `passes2.mp4` run processed and exported all 762 frames at 7.28 processing FPS. It emitted IDs for all person classes and observed the ball on 609 frames (79.92%). Ball output matched the prior integration on every frame. The smoke test verifies pipeline behavior. Identity switches remain possible and identity accuracy is unmeasured. [Configuration and smoke-test evidence](TRACKING.md#bot-sort-integration-verification)

## Final detection pipeline integration

- Candidate V2 is the default people detector at 640 pixels and 0.25 output confidence. SoccerNet-v3D is the default ball detector at 960 pixels and 0.15 confidence. Both local checkpoint hashes matched the recorded values. ByteTrack retains the 0.10 candidate floor, and ball tracker admission remains 0.25. The legacy and GSR profiles remain available. [Configuration, paths and hashes](DETECTION.md#default-video-detection)
- All 200 automated tests passed. On `passes2.mp4`, the default run processed all 762 frames at 10.25 processing FPS with PyTorch peaks of 282.84 MiB allocated and 316 MiB reserved. The annotated MP4 decoded fully and all four class IDs appeared in tracking output. All 609 observed ball tracks matched dedicated ball detections, and 335 tracked people records used sub-0.25 detections. [Smoke-test evidence](DETECTION.md#integration-verification)
- Candidate V2 was selected for its player results on Crystal Palace–Arsenal and IFAP. Goalkeeper and referee recall on IFAP remain 18.8% and 28.8%, respectively, at confidence 0.25. The integration check verifies downstream compatibility on this clip. Analytics accuracy remains outside the evidence.

## Foundation Reliability dataset audit

- Audited five detection exports and two calibration exports. All seven archive CRC checks passed. The five newly supplied archives were extracted to the approved external storage folders and their files were verified against the ZIP entries.
- The seven ZIPs use 926,862,224 bytes of the 2,500,000,000-byte cumulative dataset limit, leaving 1,573,137,776 bytes. The cap excludes pretrained weights.
- The published object detection splits share source groups. The Football Player Referee and Goalkeeper export also shares 261 source frame IDs with Football Players Detection v10. Roboflow field v18 was used for calibration diagnostics. SoccerNet Calibration 2023 validation is reserved for evaluation.
- Detailed audit reports, per-image evidence and reproducible checks are in `E:\PitchLensStorage\outputs\dataset-audit-v1`. Detection Candidate v2 passed the annotation gate, trained through epoch 40 and reached 78.90% mAP50 and 50.82% mAP50–95 on corrected development validation at 960 pixels. Production and the reserved evaluations remain unchanged pending architectural review. [Candidate v1](DETECTION_CANDIDATE_V1.md) · [Candidate v2](DETECTION_CANDIDATE_V2.md)

## Feature #04: camera calibration and 2D pitch mapping

- Implemented and verified an opt-in local evaluation baseline for dynamic metric pitch mapping with a top-down minimap and frame-level diagnostics. The official pose checkpoint was benchmarked on both clips before integration. Its exact license terms remain unresolved, so its use is restricted to local evaluation.
- End-to-end CUDA runs accepted 591 of 928 desktop frames and 49 of 390 U19 frames. Absolute mapping accuracy is unmeasured and U19 coverage is low. Goalkeeper affiliation remains unknown. See the [calibration guide](CALIBRATION.md#local-evaluation) for metrics, assumptions, and artifact locations.
- The complete suite passed 59 tests after resolving six independent GPT-6.2 Sol review findings covering degeneracy, checkpoint snapshot verification, clipped landmarks and contacts, numerical bounds, and annotation layering.
- Corrected four penalty-arc intersections. With shared frozen v9 predictions, accepted fits rose from 125 to 175 of 317 in Roboflow v18, from 591 to 829 of 928 on desktop, and from 49 to 162 of 390 decoded U19 frames. Visually incorrect fits still passed, including newly accepted frames. The SoccerNet planar evaluation is ready against the verified local archive and awaits architectural review. [Reliability findings and protocol](CALIBRATION.md#template-reliability-comparison)
- Added stricter inlier-support checks and straight paint validation. All three documented desktop failures now return unavailable. The corrected-template comparison found lower independent landmark projection error among retained Roboflow fits, with sharply reduced coverage and substantial remaining error. This closes the focused development cycle. SoccerNet remains sealed pending separate approval. [Validation results](CALIBRATION.md#landmark-support-and-paint-validation)
- Implemented opt-in temporal recovery from the last directly validated same-shot calibration. Frozen-prediction replay preserved all direct fits and recovered 101 additional desktop frames (143/928 to 244/928) and 40 U19 frames (47/390 to 87/390). No recovery crossed a detected cut. Desktop frames 505, 540 and 623 stayed unavailable.
- Sampled overlays still showed visibly misplaced pitch markings. The full suite passed 133 tests, and CUDA smoke runs verified the tracking and evaluator entry points. [Temporal recovery method, measurements and limitations](CALIBRATION.md#temporal-recovery-review)
- Previously approved with limitations for the MVP. Camera calibration and calibration-dependent features remain under scope review. Strict direct calibration remains the default, with 15.4% desktop and 12.1% U19 frame coverage. Temporal recovery remains experimental and opt-in. Recovered coordinates are excluded from speed, distance and heatmaps, and missing coordinates remain unavailable. Independent projection ground truth is unavailable.

## Feature #05: player speed and observed distance

- Implemented movement estimates for observed player and goalkeeper tracks using valid metre coordinates from strict direct calibration and source video timing. Track totals are scoped by camera shot, class and ID. Gaps and outliers reset the speed window and add no distance. Player-level coverage measures accepted duration over consecutive observed tracking duration. [Methods, configuration and outputs](MOVEMENT.md)
- All 148 tests passed. The full 928-frame desktop CUDA run yielded accepted movement for 21 of 189 tracks, including one goalkeeper. The audit matched all 185 accepted and 43 rejected segments, with 3.08 of 220.08 observed tracking seconds measured (1.40% aggregate coverage). Experimental temporal frames and the detected cut added no distance. A 96-frame strict smoke run agreed with the full run on all shared movement and tracker outputs.
- Previously approved with limitations for the MVP. Movement is under scope review with camera calibration. Observed distance covers eligible intervals only and cannot represent full-match distance. Real-world speed and distance accuracy remain unverified. [Verification results and limitations](MOVEMENT.md#verification)

## Feature #06: ball possession

- Implemented provisional player and team possession estimates in the video runner using observed ball boxes, nearby player or goalkeeper tracks, and 0.1 s of consistent evidence. The runner reports unknown ownership, time coverage and per-track episodes without requiring pitch calibration. [Method and output](POSSESSION.md)
- All 163 automated tests passed. The 10-second synthetic case produced 75% and 25% known-team shares at 80% coverage. CUDA runs analyzed 928 desktop and 390 U19 frames with pitch mapping disabled. Desktop covered 2.05% of analyzed time and credited 0.317 s across five episodes and three confirmed tracks. U19 confirmed no possession and exported null team shares.
- In the later passing evaluation, the best SoccerNet short clip covered 12.36% and the long clip covered 14.65% for players and 14.44% for known teams. Possession remains unreliable. Player tracking fragmentation and incorrect possession attribution remain major problems, and frame-level accuracy is unmeasured. [Passing-clip review](POSSESSION.md#passing-clip-review)

## Feature #07: automatic pass detection

- Integrated a calibration-independent transfer detector. It uses supported contact, observed ball separation and travel, and supported reception. The runner reports likely same-team passes, possible opponent turnovers and unclassified transfers, with shot-scoped identities and team totals. [Method and output](PASSES.md)
- All 181 tests passed at the initial implementation checkpoint. The later manual review verified 12 short-clip passes. YOLO26s alone emitted none and the SoccerNet configuration emitted one with the wrong passer. On the long clip, SoccerNet correctly identified one of six manually verified player pairs and emitted a false possible turnover. Pass detection remains unreliable. [Passing-clip review](PASSES.md#passing-clip-review)

## SoccerNet-v3D dedicated ball detector

- The original evaluation added opt-in dual-model CUDA inference using SoccerNet-v3D YOLO11l for the ball and the original PitchLens YOLO26s for people. The official checkpoint loaded using Ultralytics safe loading, and all 186 automated tests passed at that stage.
- The earlier four-clip evaluation showed mixed ball coverage and zero transfers. On four new paired short passing clips, SoccerNet improved observed ball coverage and continuity in every clip. Pooled coverage rose from 40.7% to 83.7%. SoccerNet reached 76.2% observed coverage on the long validation clip. The developer's video review also found better ball tracking. [Passing-clip evaluation](SOCCERNET_BALL.md#passing-clip-evaluation)

## Initial video check

- Processed the supplied Inter–Ajax U19 WebM with the existing checkpoint, CUDA, 960-pixel input and confidence 0.25. The 1280 × 720 source and generated MP4 each decoded to 390 frames at 24 FPS; source metadata reported 391 frames. Processing took 13.89 seconds (28.08 FPS including decoding, inference, drawing and encoding). The MP4 has no audio track.
- The detector produced ball detections in 91 frames. This is not ball recall: the video has no ground-truth annotations. Representative frames showed generally plausible player boxes, some overlapping or missed detections, a small ball, and substantial spectator occlusion near the end. No tracking or temporal identity checks were performed.
- The 248 images excluded from the final processed dataset comprise 244 held-out augmented images, two exact-duplicate images, and two previously inspected test frames. These categories are disjoint in the final preparation report.

## Delegated development workflow

- The project owner authorized autonomous execution of the complete detection feature, including compatible free dependencies, dataset preparation, local training, testing, documentation, and verified milestone commits and pushes.
- Work stays within the approved scope and $0 budget. The current cumulative dataset download limit is 2.5 GB excluding pretrained model weights. External data paths remain configurable. Raw datasets, model weights and generated experiments remain outside Git.
- The detection baseline is implemented and evaluated on frames, with material accuracy and dataset limitations documented below. Tracking and team classification were subsequently authorized; Feature #04 camera calibration and 2D pitch mapping is now authorized.

## Completed setup

- VS Code workspace created.
- Codex CLI installed and configured.
- `AGENTS.md` created, reviewed, and recognized.
- Local Git repository initialized.
- `.gitignore` created, corrected, and verified.
- Initial Git commit and GitHub push completed.
- Project folder structure designed and approved.
- Initial project structure and foundational documentation created, reviewed, and verified.

- Python environment setup completed and verified.
  - The `.venv/` interpreter reported Python 3.13.7 and confirmed it was running inside the virtual environment.
  - `pip --version` reported pip 25.2 from `.venv/`.
  - `git check-ignore` matched the `.venv/` rule, and `git ls-files` found no tracked files under `.venv/`.

## Completed and verified hardware and dependency setup

- Hardware assessment completed. PyTorch recognized the NVIDIA RTX 2060 SUPER and reported CUDA availability.
- External storage preparation and installation cache and temporary storage configuration completed and verified.
- PyTorch `2.11.0+cu128` and TorchVision `0.26.0+cu128` installed and verified in `.venv/`.
- PyTorch matrix multiplication was verified on both the CPU and the NVIDIA GPU using CUDA.

## Completed and verified package setup

- Package configuration reviewed and approved.
- PitchLens `0.1.0` installed successfully in editable mode.
- Python successfully imported `pitchlens` from `src/pitchlens/`.
- `python -m pip check` reported no broken requirements in `.venv/`.

## Completed and verified YOLO26 dependency setup

- Ultralytics `8.4.218` and its required dependencies were installed in `.venv/`. The `YOLO` API imported successfully and loaded the bundled YOLO26s detection configuration from YAML.
- PitchLens `0.1.0` still imported from `src/pitchlens/`, and `python -m pip check` reported no broken requirements after installation.
- PyTorch `2.11.0+cu128` and TorchVision `0.26.0+cu128` were preserved. PyTorch detected CUDA 12.8 on the NVIDIA GeForce RTX 2060 SUPER, and matrix multiplication returned the expected result on both CPU and GPU.

## Detection dataset acquisition and inspection

The two manually downloaded YOLO26 exports in Windows Downloads total 136,835,956 bytes, below the original 1 GB limit at acquisition. Both archives passed CRC validation, every image decoded successfully, and every image had a nonempty label file. No orphaned label files were found.

| Export | ZIP bytes | Extracted bytes | Extracted files | Train / validation / test | Image resolution |
| --- | ---: | ---: | ---: | --- | --- |
| Football Players Detection v10 | 71,896,874 | 72,158,240 | 627 | 250 / 43 / 19 | 312 at 1920 × 1080 |
| Football Object Detection v1 | 64,939,082 | 65,312,489 | 2,303 | 1,000 / 100 / 50 | 1,150 at 640 × 640 |

- Football Players Detection v10 contains 7,454 bounding boxes. Source IDs and counts are `0: ball` (272), `1: goalkeeper` (235), `2: player` (6,229), and `3: referee` (718). One referee box extends less than one pixel past the image edge. The export reports no preprocessing or augmentation. [Roboflow version 10](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/10).
- Football Object Detection v1 contains 28,275 bounding boxes and 506 polygons. Source IDs and counts are `0: ball` (977 boxes), `1: corner` (961 boxes), `2: goal` (506 polygons), `3: goalkeeper` (760 boxes), `4: player` (22,964 boxes), and `5: referee` (2,613 boxes). The ZIP retains all three byte-identical `data.yaml` entries. One copy was extracted to their shared path without overwriting. Roboflow applied auto-orientation, resized to 640 × 640, and added crop augmentation. [Roboflow version 1](https://universe.roboflow.com/football-detect/football-object-detection/dataset/1).
- Image SHA-256 checks found three exact duplicate groups in v1. All were within individual splits. No exact image matches were found between splits or between exports.
- Extraction is complete in `football-players-detection-v10` and `football-object-detection-v1` under `E:\PitchLensStorage\datasets\raw`. SHA-256 comparisons matched all 2,930 extracted files to their ZIP entries and reconfirmed both original ZIPs in Downloads. Recounts matched the original split, image resolution, and annotation totals. The external folders are outside any Git worktree, and Git ignored all corresponding paths under the repository's `data/raw/`.

## Detection dataset preparation and splitting — revised and verified

- Prepared both exports without modifying raw data. The v1 export contains variants of 242 v10 frames, rather than independent frames. A closer visual review found shared stadiums across the first split. The revised grouping conservatively joins the 15 source prefixes into six visually identified stadium groups; the exact original matches are not identified in the exports.
- Audited 1,462 input images and annotations. Removed 961 corner boxes and 506 goal polygons, clipped one sub-pixel edge overflow, found three exact duplicate groups and no duplicate annotation lines. Two duplicate images were removed from training and the remaining duplicate was among excluded held-out augmentations.
- The revised split assigned 241 training, 44 validation and 27 test original frames. Two test-group frames were inadvertently used in an earlier detector smoke check and are excluded from final scoring, leaving 25 test images. Training also contains augmented copies, for 1,145 total training images; validation and test contain original images only. Excluded 244 held-out augmentations. Each split contains all four classes, and no known stadium group, frame ID or exact image hash crosses splits.
- Revised training class counts are 22,871 player, 829 goalkeeper, 2,606 referee and 965 ball boxes. Of 272 original ball boxes, 270 are under 8 pixels wide and high when scaled to a 640-pixel input. The final test set contains 25 correlated frames from one stadium, including five goalkeeper and 21 ball boxes; this sharply limits generalization claims.
- Two independent preparations of the final version produced identical manifests and reports. SHA-256 checks verified every retained source image, processed image and saved label. Earlier manifests recorded LF-normalized label hashes rather than the Windows file bytes; version 4 corrects that checksum issue. Dataset details and commands are documented in [DETECTION.md](DETECTION.md).

## Detection training and evaluation — baseline verified on frames

- Official pretrained `yolo26s.pt` downloaded locally; SHA-256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`. It is excluded from Git.
- A fresh one-epoch CUDA smoke test on 57 training images and 44 validation images from the revised split completed with finite losses and metrics, AMP enabled, automatic batch size 12, and peak PyTorch allocated GPU memory of 3,520,646,656 bytes. The smoke results are not model performance claims.
- The first 640-pixel baseline was stopped after 11 epochs when the split leakage risk was discovered. Its artifacts are preserved but ineligible for final evaluation. The revised baseline started from the original pretrained weights, trained for 81 epochs and stopped with patience 20; epoch 61 was the best validation checkpoint. Auto-batch selected 12, and the auto optimizer selected AdamW. Peak PyTorch training allocation was 3,703,108,096 bytes and peak reserved memory was 4,227,858,432 bytes.
- Validation comparisons selected 960-pixel inference before the final test. On the 25-image held-out test, fixed-threshold macro precision was 83.63%, recall 69.94%, mAP50 81.24% and mAP50–95 55.61%. Per-class AP50 was player 98.64%, goalkeeper 61.36%, referee 90.11% and ball 74.87%. Ball recall was 13/21 at 0.25 confidence, with three false ball detections. Twenty-one of 54 referee boxes were classified as players.
- The 960-pixel GPU benchmark measured 54.06 FPS over 50 repetitions of a validation frame, excluding file I/O and video decoding. CPU and CUDA inference on a 1920 × 1080 validation frame both returned 26 detections with valid original-frame coordinates; empty results and invalid input handling were verified. All 16 automated tests passed. These checks do not establish video performance.
- The baseline used `detection-v2`; the final scoring dataset was `detection-v4`, which excludes only the two previously inspected test frames and corrects the label hashes. All training and validation images and labels were compared byte-for-byte across versions and matched. No final model or threshold was selected based on the two inspected predictions.
- Two repeated one-epoch smoke runs on the same earlier dataset produced identical logged metrics apart from elapsed time; this verifies the local smoke workflow, not bitwise equivalence across hardware or library versions.
- Aggregate precision, recall, mAP50 and mAP50–95 and goalkeeper AP50 did not meet their provisional targets. The final test covers one stadium and five goalkeeper boxes; underlying broadcast rights were not independently verified. No video footage was available for video-level verification.

## Pending setup and release work

- Additional dependency selection.
- Fresh-clone setup testing.
- Final integration and testing.
- Installation and usage documentation.
- Final portability and reproducibility testing.

## Pending core features

1. Player, goalkeeper, referee and ball detection — baseline implemented and frame-tested; architectural review and video verification outstanding
2. Player and ball tracking ? implemented and behavior-tested; annotated ground-truth evaluation outstanding
3. Automatic team classification
4. Camera calibration and 2D pitch mapping
5. Player speed and distance covered
6. Ball possession analysis
7. Automatic pass detection
8. Automatic shot detection
9. Player and team heatmaps
10. Interactive analytics dashboard

## Optional future features

- Pitch control analysis
- Expected goals (xG)
- Defensive weakness detection
- Automatic formation recognition
- Intelligent pass recommendation
- Jersey number recognition

Optional features require completion, integration, and testing of the core scope, followed by explicit feature approval and separate implementation authorization.

## Feature #02: tracking verification

- Implemented ByteTrack for person classes, a separate single-ball predictor, cut resets, structured JSONL output and annotated video generation. The tracker uses the accepted checkpoint at 960 pixels and no retraining. Details and limitations are in [TRACKING.md](TRACKING.md).
- The two supplied unlabelled videos processed on CUDA: U19, 390 frames at 27.32 processing FPS; desktop capture, 928 frames at 21.41 processing FPS. The videos and JSONL records each decoded or contained the same number of frames. A three-frame CPU smoke run completed. Timings include decoding, inference, tracking, drawing and encoding, including startup effects; they are not controlled benchmarks. The outputs contain no audio.
- Synthetic automated checks exercise ID continuity, a missing detection, lower confidence, competing ball candidates, bounded predictions, camera cuts and invalid inputs. No ground truth exists for these videos, so tracking accuracy and real-world ID switch counts are not measured. Visual and annotated evaluation across more matches remains necessary.

- The requested Codex independent review reported GPT-6.2 Sol and identified seven valid issues: output reservation, low-confidence confirmation, ball reacquisition and velocity reset, initialization order, odd-sized output and box clipping. These were addressed; the video workflow rejects odd dimensions. Spot checks of the unlabelled U19 output showed apparent ID churn in crowded play, which remains a quality limitation.

- The Feature #02 and #04 hard-cut reliability update uses appearance change and motion alignment to detect shot boundaries. It resets tracker, per-track team and calibration history at the incoming frame, with disjoint numeric track ID ranges across shots. Foundation replay found all four reviewed cuts and flagged none of 197 reviewed non-cut boundaries. The full EGYPT replay matched the archived output through frame 924 and removed three stale ball extrapolations after the cut. The reserved LIV interval remained sealed. [Method and verification](TRACKING.md#hard-camera-cuts)

## Feature #03: experimental team classification baseline

- Added experimental LAB torso classification, configurable clustering and JSONL/video integration without training. Referees are excluded; goalkeeper affiliation remains unknown; it was deferred until Feature #04, which is now authorized.
- Synthetic tests cover separation, reset, exclusions, grass, missing pixels, invalid configuration, and a correction when a second team enters after an early split. Two unlabelled videos were processed and representative frames inspected; no accuracy is measured. The focused revision passed 33 tests. U19 coverage was 0/3,362 player observations; desktop coverage was 11,986/13,175 with 2 label changes in 11,706 consecutive assigned same-ID pairs. Coverage is not accuracy. U19 remains unusable; this feature is not verified.

## Preliminary hybrid initialization exploration

A preliminary manual sample used frame 0 as initialization and frames 48 and 96 for evaluation. On selected crops from U19, SigLIP nearest prototypes correctly matched 14/14 while LAB matched 10/14; automatic LAB assigned 0/14. On desktop footage, both assisted methods matched 25/25 and automatic LAB matched 24/24 assigned (24/25 coverage). The assisted comparison forced assignments and reused track identities across frames; this tiny selected sample is not a generalization or uncertainty evaluation. Sample labels, code and crop montages are in ignored local outputs. The subsequent assisted baseline adds rejection tests; more independent examples are needed before claiming general reliability.

## Assisted SigLIP implementation and limited verification

The video runner supports selected crop references and offline SigLIP embeddings. Goalkeepers remain unknown. The final assisted runs processed 390 U19 and 928 desktop frames. U19 assigned 1,193 of 3,362 player observations; desktop assigned 8,244 of 13,175. There were zero label changes across 1,000 and 8,097 consecutive same-ID pairs respectively where both labels were non-null. These counts are not accuracy or identity-switch measurements. On a small manually reviewed selection from frames 48 and 96, U19 assigned 8 of 14 and classified 7 assigned crops correctly; desktop assigned and correctly classified 22 of 25. Initialization crops were from frame 0, but many evaluation crops show the same tracked players, so independence and generalization are limited. Reference crops, outputs and model weights are excluded from Git. No model was trained.

Automatic LAB reruns processed both complete videos after review fixes. U19 assigned 0/3,362 player observations, with no assigned pairs to assess. Desktop assigned 12,200/13,175; 2 of 11,914 consecutive assigned same-ID pairs changed labels. On the small selected evaluation, the U19 automatic mode assigned 0/14, and desktop assigned and correctly classified 24/25 after mapping cluster numbers to team labels from frame 0. Coverage and label changes are not accuracy; a stable wrong label or tracker ID switch is not captured. The U19 result requires assisted mode rather than implying automatic reliability.
