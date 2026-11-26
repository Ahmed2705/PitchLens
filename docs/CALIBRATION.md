# Camera calibration and pitch mapping

Feature #04 is an opt-in local evaluation mode in the existing tracking runner. It uses the Roboflow Soccer example's YOLOv8 pose checkpoint, independently of the existing YOLO26s detector. The pose model predicts 32 ordered pitch landmarks at 640-pixel input. These include pitch corners, box geometry, penalty spots, halfway-line points and four points on the center circle. The ordered template follows the [reference configuration](https://github.com/roboflow/sports/blob/main/sports/configs/soccer.py). Virtual landmarks on lines are treated as model predictions like the visible intersections.

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

The estimator fits a pitch-to-image homography every frame with RANSAC and inverts it for mapping. It excludes low-confidence and image-edge landmarks, then checks inlier count, spatial support, whether the correspondences constrain a unique projective fit, pixel residual and local conditioning. When successive predictions agree, it blends their projected inlier landmarks and accepts the blend only if it still fits current evidence. Invalid frames clear the transform history. The existing camera-cut heuristic prevents blending across detected cuts. A planar homography assumes a flat field and does not model lens distortion.

The silent annotated MP4 includes a top-down pitch inset. Field players and goalkeepers are mapped using the bottom center of their boxes as an estimated ground contact. The mapper rejects boxes clipped at the sides or bottom. Filled markers fall within the current landmark support. Hollow markers use limited extrapolation. Coordinates outside the field or beyond the support margin are unavailable. Ball and referee tracks retain the existing output. Goalkeeper team remains unknown. Position by a goal may provide evidence for which end a goalkeeper defends, but linking that end to a team requires independent evidence. Nearby outfield players can be on either team, and the current team classifier does not establish team-to-end identity.

Each JSONL frame adds `calibration` with validity, rejection reason, landmark and inlier counts, inlier ratio, mean inlier landmark score, fit RMSE in pixels and whether smoothing was accepted. Landmark confidence is a model score, and reprojection RMSE is measured on the accepted fit's inliers. Neither measures absolute position accuracy. Mapped tracks add `pitch` with `xy_m` and `status` (`valid`, `estimated` or `unavailable`). `unavailable` has `xy_m: null`. The summary records the pitch settings, fingerprint and mapping counts. The original tracker output format is preserved when `--pitch-weights` is omitted.

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

Both annotated videos decoded to the same number of frames as their JSONL. All emitted positions were within pitch bounds, and invalid frames emitted no positions. The existing heuristic flagged zero cuts across the two clips. A synthetic cut through the tracker and calibrator incremented the segment, prevented smoothing on the cut frame, and allowed it again on the next frame with fresh support. A 32-frame desktop run with pitch mapping disabled exactly matched the original tracking and team fields in the mapped run.

On desktop, 266 of 269 mapped goalkeeper observations fell in a modeled penalty area, and none of the seven U19 goalkeeper observations could be mapped. This supports checking a goalkeeper's likely defending end in wide views. It does not link that end to a team, so all goalkeeper team labels remained unknown.

The verified MP4, JSONL and summary JSON use the stems `pitch-desktop-v2` and `pitch-u19-v2` under `E:\PitchLensStorage\outputs`. The initial landmark overlays and sample reports are in `pitch-benchmark-desktop-v1` and `pitch-benchmark-u19-v1` in the same directory.
