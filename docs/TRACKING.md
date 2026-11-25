# Player and ball tracking

The tracking pipeline uses the existing YOLO26s checkpoint at 960 pixels. It calls the detector with a 0.10 person threshold so ByteTrack can associate lower-confidence detections; the ball retains the accepted 0.25 threshold. This differs from the detection evaluation's 0.25 global operating point. No model was retrained. `lap` supplies the assignment solver required by the pinned Ultralytics ByteTrack implementation.

```powershell
.\.venv\Scripts\python.exe scripts\track_video.py `
  --input 'E:\PitchLensStorage\footage\your-video.mp4' `
  --output 'E:\PitchLensStorage\outputs\new-tracking.mp4' `
  --weights runs\detection-baseline-640-v2\weights\best.pt `
  --config configs\tracking.json --device auto
```

The output directory must exist. Use a fresh name: the MP4, JSONL and summary JSON are never intentionally overwritten. MP4 output requires even frame dimensions and has no audio. Incomplete artifacts can remain if processing fails; choose a new name after examining them. `--device cpu` and `--device cuda` explicitly select the backend; `--max-frames` supports smoke checks. Processing is sequential, at the source frame rate. Variable-frame-rate timing is not represented.

Each JSONL line represents one zero-indexed frame, with `segment`, `camera_cut`, `tracks` and `ball_trajectory`. Tracks have `xyxy` coordinates in original-frame pixels, `class_id`, `track_id`, `confidence`, and `observed`. Identify people by the tuple `(segment, class_id, track_id)` within a video; IDs are restarted on cuts and new videos. Person tracks are only emitted when matched to a detection; a temporarily lost person may recover its ID but is not emitted during the gap. Class changes can produce a new ID. A ball has no persistent ID: predicted ball positions have `observed: false` and `confidence: null`; trajectory entries are `[x, y, observed]`. Green denotes observed ball positions, orange extrapolation. The tracker is initialized per video; API users must call `reset()` between videos and submit every frame in order at the declared FPS.

The ball starts at the highest-confidence candidate, then uses a constant-velocity prediction and selects candidates using confidence and distance within a frame-diagonal gate. It extrapolates for a bounded time and clears the trajectory when lost. ByteTrack score fusion is disabled so a consistent low-confidence new person can be confirmed using overlap. False positives can initialize or hijack a track, and fast motion, camera motion, occlusion or small detections can break it. The simple grayscale frame-difference cut heuristic resets all state on a sufficiently large change; it can miss cuts or mistake flashes and camera movement for cuts. Thresholds, time windows and gate size are configurable in `configs/tracking.json`. They are starting values, not validated optimal settings. No camera-motion compensation or re-identification is provided; ID switches remain possible, especially during occlusions. Quantitative tracking accuracy requires independently annotated footage.
