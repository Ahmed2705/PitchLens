# Ball possession estimation

Possession runs in the existing tracking runner, including when pitch calibration is disabled. It uses current image-space boxes and the existing team classifier. The runner highlights the confirmed owner's box and displays the team when available. The overlay includes known-team coverage alongside any team percentage.

## Candidate selection and confirmation

The tracker keeps its existing ball selector. An observed ball can provide evidence when it was the only viable detection after the tracker's confidence and position gates. Multiple viable ball detections are treated as ambiguous for possession, including on initialization when there is no predicted position to gate against. Missing, predicted and invalid ball boxes provide no evidence.

For each observed player or goalkeeper with a positive track ID, measure the distance from the observed ball centre to the full-width lower 20% of the person box. The distance is zero inside that region. Outside, scale the vertical gap by box height and the horizontal gap by the greater of box height and twice box width. This makes comparable contacts less sensitive to image scale. Referees and invalid boxes are excluded. The nearest in-range candidate must clear the margin from the next in-range player, even when both players have the same team label. A goalkeeper with an unknown team can earn individual possession time.

An in-range, unambiguous candidate starts as pending. Consecutive observations compare the ball's normalized offset from the player's lower centre. Motion shared by the player and ball cancels out of that offset. Relative travel above the configured rate resets the confirmation timer. Pending confirmation also checks displacement from the first supporting sample, limiting the effect of per-frame jitter slack on a ball gradually traveling through the region.

A confirmed owner becomes unknown or pending as soon as current evidence is lost, contested, or indicates travel. Brief interruptions retain the previous identity internally and let that same player resume on a fresh, eligible observation within the hysteresis window. Missing or predicted balls cannot confirm a new owner, and the gap and first recovered interval earn no credit. Sustained separation or travel expires the hold, and a new owner requires its own confirmation. Camera cuts reset pending evidence, motion history and the hold. Track identity is scoped by camera shot, class and track ID. The pass detector retains its existing contact and travel rules and can still use a confirmed owner as corroboration.

## Configuration

Run `scripts.track_video` as usual. Optionally pass `--possession-config PATH` with a JSON object containing overrides.

| Field | Default | Meaning |
| --- | ---: | --- |
| `max_normalized_distance` | 0.30 | Maximum normalized distance outside the lower box region. |
| `min_distance_margin` | 0.1 | Minimum separation from the next in-range candidate, in normalized distance units. |
| `confirmation_seconds` | 0.1 | Required continuous support for the same candidate, in source video seconds. |
| `hysteresis_seconds` | 0.12 | Maximum time since the last supported observation for the same owner to resume. Set to zero to disable the hold. |
| `max_relative_speed` | 1.5 | Maximum normalized player-relative movement per second, plus 0.025 normalized units per check for measurement jitter. |
| `max_observation_gap_seconds` | 0.25 | Largest adjacent source time step allowed to carry evidence. A larger valid gap contributes only unknown time. |

These are provisional engineering defaults. They were not fitted to these clips.

## Duration accounting

Increasing video timestamps supply elapsed time. When the source timestamp clock is unavailable, adjacent frame indices and validated FPS supply elapsed time through the same clock as movement analytics. The first frame has no incoming duration. Invalid timing and nonconsecutive frame indices exclude the affected intervals. A cut with valid timing adds an unknown interval and starts a new evidence sequence.

Each valid interval is counted once. Player duration is credited only when the same owner is confirmed at both ends. No duration is credited back to the pending period or to the interval arriving at the first confirmation. Team duration additionally requires the same known label at both ends. Other player-confirmed intervals count as player possession with unknown team. An episode is counted once at its first confirmed sample. If it ends before the next supported sample, it can have zero credited duration.

- Team possession percentage divides each team's duration by the total known-team duration. Shares are null when that denominator is zero.
- Team possession coverage divides known-team duration by total analyzed duration. Player possession coverage uses confirmed player duration over the same denominator. Both coverages are null when no interval has been analyzed.
- Unknown team duration is analyzed duration minus known-team duration. It includes unknown player possession and player-confirmed time with unknown team.
- Unknown player duration includes missing, contested, traveling-ball and pending intervals. The overlay displays the likely current owner, Team A and Team B shares when available, and known-team coverage.

## Structured output

Each JSONL record adds `possession` with `state` (`confirmed`, `pending` or `unknown`), `reason`, `owner`, `pending_candidate`, `normalized_distance`, `ball_status`, `timing`, and running `totals`. Owner and pending identity objects include `segment`, `class_id`, `track_id` and nullable `team`. Timing includes elapsed seconds, the source clock, credited player seconds and the credited team. Running totals contain analyzed, player-confirmed, known-team and unknown durations in seconds, both coverage percentages, and a list of team durations and percentages. The tracker also emits `ball_observation_ambiguous` as an additive provenance flag.

The final summary adds possession settings, analysis time per frame, the final duration totals and a track list with durations, episode counts and final status. A track's summary team is null if no confirmed frame had a known label or if known labels disagreed. Track IDs cannot establish that the same individual appears in a different shot.

## Possession upgrade verification

All 213 automated tests passed, including tests for controlled motion, a ball traveling past a player, transfers, brief tracking interruptions, predicted positions, ambiguity, cuts and interval accounting. The CUDA smoke test processed and decoded all 762 frames of `passes2.mp4` at 7.70 processing FPS. Tracking and team records matched the previous integration on every frame. The interval audit reconciled all totals and found no possession credit on predicted ball positions or across unsupported intervals.

| Measurement | `passes2.mp4` |
| --- | ---: |
| Analyzed duration | 12.684 s |
| Confirmed player possession | 2.667 s (21.03%) |
| Unknown or contested player possession | 10.017 s |
| Known-team possession | 2.618 s (20.64% coverage) |
| Unknown team possession | 10.066 s |
| Team A share of known-team time | 0.00% (0.000 s) |
| Team B share of known-team time | 100.00% (2.618 s) |

Manual review inspected three-frame strips around eight selected moments. All five sampled confirmed owners looked consistent with visible control, zero sampled assignments looked incorrect, and three moments had unknown possession. The unknown examples covered a traveling ball, an overlap where the red player appeared to retain control, and a kick with only a predicted ball position. This small sample cannot establish clip-wide accuracy.

The annotated MP4, JSONL, summary, interval verification and manual review are in `E:\PitchLensStorage\outputs\possession-integration-v1`.

## Initial baseline verification

All 163 automated tests passed. The 10-second synthetic example, including confirmation and transition gaps, credited 6.00 s to Team A, 2.00 s to Team B and 2.00 s to unknown possession. The corresponding known-team shares were 75% and 25%, at 80% known-team coverage. Synthetic tests also cover goalkeepers, same-team and opposing-team ambiguity, invalid boxes and timing, and owner resets across cuts.

Both existing clips ran on CUDA with pitch calibration disabled and default automatic team classification. The output videos decoded to the same number of frames as the JSONL records. The interval audit reconciled player totals, team totals and unknown duration and found no credit across a cut or on estimated or ambiguous ball observations. The desktop track, team and ball trajectory fields matched the archived prior runner on all 928 frames.

| Clip | Frames | Analyzed time | Tracker-observed ball frames | Confirmed player coverage | Known-team coverage | Unknown player / team time | Episodes / tracks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Desktop | 928 | 15.450 s | 569 (61.3%) | 2.05% | 2.05% | 15.133 / 15.133 s | 5 / 3 |
| Inter–Ajax U19 | 390 | 16.208 s | 91 (23.3%) | 0% | 0% | 16.208 / 16.208 s | 0 / 0 |

Desktop credited 0.317 s to Team 1 and 0.000 s to Team 2. Their known-team shares were 100% and 0% within that 0.317 s sample. One episode ended before any duration was credited, so two of the three confirmed tracks accumulated positive time. U19 credited 0.000 s to both teams and exported null shares. Tracker-observed ball coverage includes frames where the selector had multiple candidates and measures tracker output availability. Mean possession update time was 0.041 ms/frame on desktop and 0.027 ms/frame on U19, excluding inference, rendering, encoding and JSON writes.

Overlay samples used zero-based frame indices. Desktop player 12 was highlighted close to the ball at frames 514–520. Frame 97 left a ball between converging players unknown. Frame 117 had no ball detection. Player 12 returned from pending at 559 to confirmed at 561, then became unknown with multiple ball candidates at 570. During the desktop ball movement around 594–605, the brief candidate and subsequent estimated positions earned no credit. U19 frames 122–125 show the observed ball travelling from the kicker toward the wall, with possession unknown. No goalkeeper episode was confirmed. At desktop frame 459 the ball was near a goalkeeper with overlapping person detections, and the estimate remained unknown. The visual checks showed plausible proximity and missed apparent control. They could not establish definite false confirmed owners or measure accuracy.

Local verification outputs and representative overlays are under `outputs/feature06-*`. Feature #06 was provisionally approved with significant limitations. Real-world possession accuracy remains unverified. Possession accuracy and ball detection reliability are deferred to the improvement phase.

## Limitations

Image-space possession estimates likely ownership. Ball flight can project near an unrelated player's lower box, and slow relative travel or shared movement can pass confirmation. Zoom, parallax, occlusion, false ball detections and track fragmentation can affect the estimate. Team percentages inherit classifier errors and missing labels. The default SigLIP classifier fixes the two team identities for the clip, while the color fallback can change labels after cuts. Verified physical control requires independent possession annotations.

## Passing-clip review

SoccerNet increased confirmed player coverage on each of the four paired short clips, but the highest coverage was 12.36%. On the long clip, confirmed player and known-team coverage were 14.65% and 14.44%, respectively. The displayed 99.91% share for one team used only the 18.79 seconds with a known team, leaving 85.56% of analyzed time uncredited to either team.

Possession remains unreliable. Fragmented player tracks interrupt evidence, and a nearby opponent can be incorrectly credited with contact as the ball travels past. These coverage measurements describe how often the estimator credited possession. Frame-level possession accuracy remains unmeasured. [Detector comparison and evidence location](SOCCERNET_BALL.md#passing-clip-evaluation)
