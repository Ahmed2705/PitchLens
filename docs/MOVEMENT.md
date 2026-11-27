# Player speed and observed distance

Feature #05 runs automatically in `scripts.track_video` when pitch mapping is enabled with `--pitch-weights`. Add `--movement-config PATH` to override the default settings with a JSON object. The tracking-only invocation keeps its existing output. The pitch-enabled runner adds movement fields to its usual JSONL and summary JSON, and displays current speed beside a player or goalkeeper only while a speed estimate is available.

## Calculation and coverage

Only observed players and goalkeepers with positive track IDs, direct calibration and a finite `pitch.xy_m` with status `valid` are eligible. The calibration output uses metres. The eligibility check excludes off-support extrapolation (status `estimated`), experimental temporal recovery and unavailable coordinates. Track identity is scoped by `segment` (camera shot), `class_id` and `track_id`. The runner uses the existing tracker cut signal.

For adjacent eligible observations, distance is the Euclidean displacement in metres. Elapsed seconds come from increasing source video timestamps (`CAP_PROP_POS_MSEC`) when available. A missing timestamp clock, including an initially constant zero, uses frame indices and validated source FPS. Duplicate or backward timestamps on an active clock and negative or non-finite timestamps break measurement continuity. Processing time is never used.

Each accepted segment contributes displacement and elapsed time to its track totals. An implied segment speed above the configured limit resets the smoothing window and discards the rejected endpoint. Missing observations, shot boundaries, invalid timing and ineligible coordinates also break continuity. Totals from earlier accepted segments remain available. Distance is the sum of the accepted displacements and represents only the observed, measurable portions of a track.

Current speed divides total displacement in the most recent consecutive accepted intervals by their total duration and multiplies by 3.6. It uses whole intervals and retains enough to span the smoothing target when available. It can report a shorter window once it has the minimum observation count, and the minimum count can extend the window at low frame rates. The average uses all accepted segments, and the maximum uses only the smoothed estimates.

Movement measurement coverage is valid measurement duration divided by observed tracking duration, as a percentage. The denominator sums timed intervals between adjacent frames in the same shot with observed boxes for that track, including intervals without valid calibration or with rejected movement. Frames without that track, invalid timing and shot boundaries contribute no duration. This differs from frame-level calibration coverage, which counts directly calibrated frames over all decoded frames.

## Configuration

| Field | Default | Meaning |
| --- | ---: | --- |
| `max_speed_kmh` | 45.0 | Reject segments with implied speed above this value in km/h. |
| `smoothing_seconds` | 0.2 | Target duration of the trailing speed window in seconds. |
| `min_observations` | 3 | Minimum consecutive eligible positions for current or maximum speed. Minimum allowed is 3. |

## Structured output

Frame JSONL includes `movement_timing` with `elapsed_seconds` and a `source` of `video_timestamp`, `frame_index_fps`, or `null`. Each eligible class and valid ID gets a `movement` object. The final summary includes `movement.config` and `movement.tracks`, with the same statistics plus the shot, class and track ID. Summary current speed is available only for tracks with an estimate on the final frame.

| Field | Units and availability |
| --- | --- |
| `speed_kmh`, `maximum_speed_kmh` | km/h from smoothed intervals. Null until the required observations exist. |
| `average_speed_kmh` | Accepted metres per valid second times 3.6. Null before the first accepted segment. |
| `observed_distance_m` | Sum of accepted displacements in metres. Null before the first accepted segment. |
| `valid_measurement_duration_seconds` | Seconds across accepted segments. |
| `observed_tracking_duration_seconds` | Seconds across observed adjacent track pairs with valid timing. |
| `measurement_coverage_pct` | Valid duration as a percentage of observed tracking duration. Null when the denominator is zero. |
| `accepted_segments`, `rejected_segments` | Counts of measured segments passing or exceeding the speed limit. Missing inputs are gaps. |

## Verification

All 148 automated tests passed. Synthetic trajectories verified 10.00 m and 18.00 km/h over two seconds at 10, 24 and 60 FPS. The tests also exercise irregular timestamps, FPS fallback, stationary observations, class-scoped and shot-scoped IDs, rejected jumps and null serialization. The runner's synthetic camera-cut test checks movement state on both sides of each cut.

The existing desktop clip ran end to end on CUDA with temporal calibration enabled to exercise its exclusion. A second run used the strict default for the first 96 frames. Both generated MP4s decoded fully and matched their JSONL frame counts.

| Desktop run | Frames | Observed tracks | Tracks with accepted movement | Tracks with smoothed speed | Accepted / rejected segments | Valid / observed tracking seconds | Movement coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full clip, temporal calibration enabled | 928 | 189 | 21 (20 players, 1 goalkeeper) | 17 | 185 / 43 | 3.08 / 220.08 | 1.40% |
| Strict default smoke run | 96 | 40 | 6 (5 players, 1 goalkeeper) | 6 | 25 / 0 | 0.42 / 33.72 | 1.25% |

The full clip contained 143 direct, 101 temporal and 684 unavailable calibration frames. A segment-by-segment audit reproduced the accepted distance and duration totals, confirmed unavailable current speed on ineligible observations and found no motion counted across gaps, temporal calibration or the cut at frame 925. Tracking and team records matched the archived pre-movement run on all 928 frames. The first 96 frames had identical movement output with and without experimental temporal calibration.

At desktop frame 545, player track 32 in shot 0 showed 18.6 km/h. Across the clip that track accumulated 2.41 m over 0.55 s of accepted movement, with a 15.8 km/h average, a 30.4 km/h smoothed maximum, and 4.05% measurement coverage across its observed tracking duration. These are sample output values, and their real-world accuracy is unverified. Local evidence is under `outputs/feature05-desktop-*`.

## Limitations

Observed distance covers the accepted intervals only. It cannot represent full-match distance. Strict direct calibration coverage is low, and filtering extrapolated coordinates reduces the eligible sample further. The speed limit is an engineering guardrail. Calibration jitter, approximate field dimensions, contact-point error and undetected track ID switches can still affect estimates below the limit. Shot identities cannot be linked into one player's match total. Real-world speed and distance accuracy remain unverified without independent ground truth.

Feature #05 was previously approved with these limitations for the MVP. It remains under scope review with camera calibration. The review accepted the synthetic verification and the 148 passing tests. The 1.40% desktop measurement coverage and 21 of 189 tracks with accepted segments describe the limited observed sample. Real-world speed accuracy remains unverified.
