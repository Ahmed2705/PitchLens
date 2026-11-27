# Automatic pass detection

Pass detection runs in the existing tracking runner with or without pitch calibration. It estimates transfers from observed ball and player tracks, team labels and, when available, current possession evidence.

## Contact and transfer evidence

The runner uses the observed ball box centre and the bottom centre of each observed player or goalkeeper box. Contact distance is measured in player box heights. The nearest in-range candidate must be separated from other in-range candidates by the configured margin. Referees, invalid boxes, stale tracks and ambiguous ball observations provide no contact evidence. A current confirmed owner can corroborate an unambiguous candidate. Otherwise, a short run of consecutive contact observations supports the candidate independently of possession.

1. Support a source track, then wait for the first observed ball sample separated from that player and outside all contact candidates. This is the release frame.
2. Require intermediate observed ball samples. Measure displacement both in the image and relative to the source player's feet, normalized by the source's box height. Each measure must clear the travel threshold by first reception. Consecutive observed intermediate samples must also accumulate at least a quarter of that threshold in each measure. This prevents a gap or endpoint jump from supplying all the apparent motion and reduces sensitivity to camera pans.
3. Require contact with a different, unambiguous receiver inside the configured time window. Emit once the receiver is supported. Continued contact remains part of that reception, and the supported receiver may become a source when new release and travel evidence follows.

The source track must remain observed until reception is confirmed. Short missing or predicted-ball gaps are allowed during flight when bracketed by observed travel samples. Predicted points cannot establish contact, start a flight or supply movement. A gap ending directly at reception is rejected. Ambiguity, source loss, return to the same source, an unsupported receiver, a timing discontinuity, an excessive gap, a cut or the end of the clip can terminate a flight. Rejection counts include only flights that started, so brief contacts and direct proximity switches can produce no event and no rejection.

## Configuration

Use `--pass-config PATH` with a JSON object to override these provisional engineering defaults.

| Field | Default | Unit |
| --- | ---: | --- |
| `max_normalized_contact_distance` | 0.5 | Player box heights |
| `min_contact_margin` | 0.1 | Normalized distance |
| `min_contact_seconds` | 0.05 | Seconds |
| `min_normalized_travel` | 0.75 | Source box heights |
| `min_transfer_seconds` | 0.08 | Seconds |
| `max_transfer_seconds` | 3.0 | Seconds |
| `max_observation_gap_seconds` | 0.2 | Seconds between observed samples |
| `min_travel_samples` | 2 | Observed separated samples, including release |

Increasing video timestamps supply elapsed time. The shared video clock falls back to validated FPS when the source clock is unavailable. The first frame and invalid or nonconsecutive intervals contribute no elapsed time. Event duration is reception timestamp minus release timestamp. The confirmation frame can follow the first reception frame.

## Classification and outputs

- `likely_successful_pass` requires two different players with the same known team label.
- `possible_turnover` means the players have different known team labels. Opponent reception alone cannot establish that an attempted pass failed or was intercepted.
- `unclassified_transfer` means either endpoint lacks a supported team label or the labels conflict within the evidence for that endpoint.

Each JSONL frame adds `passes.events`, `passes.rejection_reason` and `passes.running`. Newly confirmed events contain an ID unique within the run, classification, shot and player identities, nullable teams, release/reception frames and timestamps, confirmation frame, duration, observed travel sample count, largest observation gap and evidence. Evidence includes normalized distances and the original observed ball coordinates in pixels. The summary adds the complete event list, rejected-flight reasons, per-team and per-track counts, ball coverage and analyzed duration. Team turnovers are counted separately by losing and receiving side. Observed ball coverage counts tracker-observed balls, including detections flagged as ambiguous. Usable frames exclude ambiguous and invalid observed boxes. Track totals are scoped by shot, class and track ID.

The overlay shows the type of a newly confirmed transfer, source and receiver IDs, and detected team pass counts for up to 1.25 seconds. It clears on cuts or missing interval timing. The existing tracker trail provides ball-position context.

## Verification

All 181 automated tests passed, including 18 pass tests. Synthetic cases verified same-team, opponent and unknown-team transfers, stable dribbling, camera pans, timestamp and frame-rate differences, short and long observation gaps, and cuts. A runner fixture verified that event records appear in JSONL and the final summary, an event overlay clears on a cut, and the next overlay expires after 1.25 seconds. A separately rendered synthetic same-team transfer from track 1 to track 2 released at frame 2 (0.10 s), reached frame 5 (0.25 s) and confirmed at frame 6. It had three observed travel samples and lasted 0.15 s.

Both existing clips ran on CUDA without pitch calibration. Each output MP4 decoded to the same number of frames as its JSONL. All existing per-frame tracker, team and possession fields matched the Feature #06 runs exactly.

| Clip | Frames | Analyzed time | Observed ball frames | Usable ball frames | Likely passes / turnovers / unclassified | Rejected flights | Mean pass update |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Desktop | 928 | 15.450 s | 569 (61.3%) | 528 | 0 / 0 / 0 | 4 | 0.063 ms/frame |
| Inter–Ajax U19 | 390 | 16.208 s | 91 (23.3%) | 90 | 0 / 0 / 0 | 0 | 0.041 ms/frame |

Team 1 and Team 2 each had zero detected passes in both runs. Update timings exclude inference, overlays, encoding and JSON writes. The four desktop flights ended as source recontact at zero-based frames 216, 535, 547 and 554. Overlay spot checks around 534–535 and 547 showed the ball close to the same group of players, with no distinct supported reception. U19 frames 122–125 showed apparent free-kick travel, with no supported source-to-receiver sequence. U19 opened no flight candidates. No real event was available for source and receiver verification. The synthetic overlay verified display of a completed event. No third suitable raw development clip was identified in the project footage location.

The footage artifacts are `outputs/feature07-desktop.{mp4,jsonl,json}` and `outputs/feature07-u19.{mp4,jsonl,json}`. The controlled overlay is in `outputs/feature07-synthetic.mp4`, with samples in `outputs/feature07-overlay-review.jpg`. Feature #07 is ready for human review as an implementation. Practical pass-detection usefulness on the supplied clips remains unverified.

## Limitations

Detected transfers are heuristic estimates. They do not necessarily represent complete or accurate match statistics, and a clip with zero detections may still contain real passes. Camera motion, perspective, ball occlusion, false detections and tracking ID changes can obscure or mimic contact or travel. Requiring the source to remain visible and requiring image displacement can miss passes when the camera follows the ball or the source leaves the frame. Automatic team labels can swap after a cut. The output cannot measure intent, calibrated pass distance, ball speed or pass completion rate. Neither verification clip has annotated pass ground truth.

## Passing-clip review

Across four short clips, source-footage review verified 12 completed passes. YOLO26s alone emitted none. The SoccerNet run emitted one likely pass but credited a striped defender as the passer on a red-team pass and assigned him the receiver's team. Neither configuration correctly attributed a short-clip pass.

On the long SoccerNet clip, the manual sample verified six completed passes. The event confirmed at frame 5656 credited the correct player pair, with release recorded seven frames after the end of the manual release window. The possible turnover at frame 4555 credited a red presser as the receiver of a ball that rolled past him untouched. The manual review was a sample, so full-clip pass totals and recall remain unknown.

Pass detection remains unreliable. Missing ball observations around release or reception, fragmented person tracks, and incorrect contact and team attribution explain observed failures. In one short-clip pass, a two-frame source-track dropout canceled the flight despite observed ball positions on every travel and reception frame. [Detector comparison and evidence location](SOCCERNET_BALL.md#passing-clip-evaluation)
