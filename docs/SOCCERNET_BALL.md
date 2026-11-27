# SoccerNet-v3D ball detector integration

The dedicated model is available for local evaluation through `--ball-weights`. The single-model YOLO26s runner remains the production default pending human review.

## Checkpoint and provenance

| Item | Verified value |
|---|---|
| Release | SoccerNet-v3D `v1.0.0`, GitHub asset ID `246873738` |
| Local path | `E:\PitchLensStorage\models\pretrained\yolo-sn-ball-opt.pt` |
| Size | 51,266,130 bytes, matching the official release API |
| SHA-256 | `a3082fb435a8501ae17cfe4ac78e66ca7041205e115feda344d34d1693064f36` |
| Architecture | Ultralytics YOLO11l (`yolo11l.yaml`), 25,311,251 parameters before fusion, object detection task |
| Class map | Checkpoint `0: ball` → PitchLens `3: ball` |
| Runtime | Python 3.13.7, PyTorch 2.11.0+cu128, CUDA runtime 12.8, Ultralytics 8.4.218, NVIDIA GeForce RTX 2060 SUPER (8 GiB) |

GitHub did not publish a digest for this asset. The repository links to the selected asset, and the official release API matched the tag, name, URL and byte count. The SHA-256 above was computed locally after download. [Repository](https://github.com/mguti97/SoccerNet-v3D#weights) · [Official release](https://github.com/mguti97/SoccerNet-v3D/releases/tag/v1.0.0) · [Release metadata](https://api.github.com/repos/mguti97/SoccerNet-v3D/releases/tags/v1.0.0) · [Checkpoint download](https://github.com/mguti97/SoccerNet-v3D/releases/download/v1.0.0/yolo-sn-ball-opt.pt)

This is an externally pretrained model by Marc Gutiérrez-Pérez and Antonio Agudo, whose SoccerNet-v3D paper appeared at the 2025 CVPR Workshops. The authors trained the detector on SoccerNet-v3 and fine-tuned it with optimized SoccerNet-v3D ball boxes. PitchLens supplied no training or fine-tuning data for this model. The original PitchLens-trained YOLO26s remains responsible for players, goalkeepers and referees. [Authors’ training provenance and citation](https://github.com/mguti97/SoccerNet-v3D#weights)

The repository’s tagged license is GPL-2.0. The checkpoint embeds an Ultralytics AGPL-3.0 notice and records Ultralytics 8.3.64 as its training version. The release supplies no separate explanation reconciling those notices for the checkpoint. SoccerNet also gates some protected video datasets under an NDA. Checkpoint redistribution or commercial use remains unresolved and outside this local evaluation. [Tagged repository license](https://github.com/mguti97/SoccerNet-v3D/blob/v1.0.0/LICENSE) · [Ultralytics licensing](https://www.ultralytics.com/license) · [SoccerNet data access](https://www.soccer-net.org/data)

## Running and configuring the detector

With `--ball-weights`, PitchLens runs both models on each original BGR frame, keeps the YOLO26s person detections, discards its ball boxes, and supplies the dedicated ball boxes to the existing tracker. Invalid or incompatible dedicated checkpoints fail before output creation. Omitting `--ball-weights` selects the original single-model runner.

The general detector runs at 960 pixels. The dedicated model defaults to 960 pixels and a confidence cutoff of 0.15. To override dedicated inference, pass `--ball-config` with a JSON object such as `{"image_size": 1280, "confidence": 0.15}`. Image size must be a positive multiple of 32 and confidence must be in [0, 1]. Both models use the selected `--device`. The production tracker still requires a ball confidence of 0.25, so candidates from 0.15 to below 0.25 can appear in raw diagnostics only under the evaluation settings.

Ultralytics reports `xyxy` in original image pixels, and the detector clips them to the original frame boundaries. On a 1920 × 1080 FCB frame 150, the dedicated model returned approximately `(907.01, 610.46, 920.90, 624.46)` at confidence 0.16585, matching the visible ball in the review image. Scores are the model’s original [0, 1] confidence values. Dual-mode JSONL adds `raw_ball_detections` with post-NMS candidates at the dedicated inference threshold. The tracker can select an observed ball, carry a prediction with null confidence, or report no ball. The possession and pass logic continues to consume the tracked provenance.

The following Windows CMD example writes a new output stem. It uses the stock Ultralytics safe checkpoint loader and matches the evaluation’s CUDA, strict pitch, tracking and analytics settings.

```bat
cd /d "C:\Users\Ahmad\Desktop\Coding\PitchLens"
set "ULTRALYTICS_SAFE_LOAD=1"
"C:\Users\Ahmad\Desktop\Coding\PitchLens\.venv\Scripts\python.exe" "C:\Users\Ahmad\Desktop\Coding\PitchLens\scripts\track_video.py" --input "E:\PitchLensStorage\footage\FCB.mp4" --output "E:\PitchLensStorage\outputs\soccernet-ball-v1\FCB-manual-dual.mp4" --weights "C:\Users\Ahmad\Desktop\Coding\PitchLens\runs\detection-baseline-640-v2\weights\best.pt" --ball-weights "E:\PitchLensStorage\models\pretrained\yolo-sn-ball-opt.pt" --pitch-weights "E:\PitchLensStorage\models\football-pitch-detection-v9.pt" --config "C:\Users\Ahmad\Desktop\Coding\PitchLens\configs\tracking.json" --device cuda
```

## Verification

The five focused tests and all 186 automated tests passed in the existing virtual environment. A mocked runner test passed real tracker output through possession and pass detection, confirmed a synthetic same-team pass, distinguished observed, predicted and missing balls, and verified the baseline branch and output schema. CUDA footage runs enabled Ultralytics safe loading and used the archived baseline’s settings, including automatic team classification and strict direct pitch calibration with temporal recovery disabled. The four footage hashes matched the archive, and every exported person track record, including team and movement data, matched its archived counterpart. All five generated videos decoded to the expected 1920 × 1080 frame counts. An isolated YOLO26s inference audit recovered raw boxes for FCB frames 116–187 because the archive contains selected tracks only.

### Known FCB pass

Frame indices are zero-based. The light-blue player releases around 139–140 and the receiver is visible around 177–178.

| Window | General 960 raw / observed | Dual 960 raw / observed | Dual 1280 raw / observed |
|---|---:|---:|---:|
| Before release, 116–138 (23 frames) | 12 / 12 | 16 / 13 | 23 / 16 |
| Release, 139–143 (5 frames) | 0 / 0 | 4 / 4 | 5 / 4 |
| Travel, 144–176 (33 frames) | 30 / 30 | 23 / 13 | 30 / 28 |
| Reception, 177–179 (3 frames) | 2 / 2, both visually false | 0 / 0 | 0 / 0 |

The raw columns count frames with at least one candidate. The longest travel stretch without a selected observation was 2 frames in the baseline, 12 at dedicated 960, and 2 at dedicated 1280.

At frame 140 both dedicated resolutions picked up the visible ball leaving the player’s foot, while the baseline continued the prior prediction. At frame 150 the dedicated 960 raw score was about 0.166, below the tracker cutoff, and the prediction fell behind. At frame 170 dedicated 960 produced a false candidate near the assistant referee’s flag at confidence 0.32, which the tracker rejected. The baseline selected a false box well above the receiver’s foot at frames 178–179. Both dedicated resolutions avoided that false selection and kept predicting near the receiver, but neither produced any raw reception detections on frames 177–179. Neither recovered the pass.

The 1280 comparison processed the first 210 frames at 6.43 FPS, took 53.03 ms per frame for dedicated inference, and peaked at 649 MiB allocated and 736 MiB reserved by PyTorch. Timing from this short run includes startup during the first inference and overlapped briefly with the isolated baseline inference audit.

### Four-clip comparison

Each cell shows the archived general detector result followed by the new dual-960 result. Missing means no ball track was emitted. Unobserved includes predictions and missing tracks. Possession coverages are percentages of analyzed time.

| Clip | Observed ball | Longest missing (frames) | Longest unobserved (frames) | Player possession | Known-team possession | Processing FPS |
|---|---:|---:|---:|---:|---:|---:|
| EGYPT (928 frames) | 61.31% → 52.05% | 31 → 28 | 46 → 43 | 2.05% → 1.74% | 2.05% → 1.74% | 11.56 → 8.36 |
| MAN (809 frames) | 68.36% → 61.43% | 60 → 15 | 75 → 30 | 0.62% → 0.49% | 0.62% → 0.49% | 11.42 → 8.47 |
| FCB (797 frames) | 49.81% → 53.95% | 31 → 54 | 46 → 69 | 7.68% → 2.76% | 1.26% → 0.25% | 10.68 → 8.11 |
| LIV (727 frames) | 55.16% → 76.62% | 149 → 16 | 153 → 31 | 3.29% → 4.40% | 3.29% → 4.40% | 11.29 → 8.44 |

Likely successful passes, possible turnovers and unclassified transfers were each zero for all four clips in both modes. Dedicated 960 inference took 30.3–30.8 ms per frame. End-to-end throughput fell about 24–28%. The peak PyTorch CUDA allocator readings at 960 were about 555 MiB allocated and 620 MiB reserved for the pipeline. These figures exclude other GPU allocations, and the archive did not report comparable memory readings.

In the EGYPT dribble sample, both models observed the ball at frame 590. At frame 565 the dedicated tracker could only predict it. In MAN, the dedicated model picked up the visibly airborne ball at frame 301 during a baseline tracking gap. At frame 331 it produced two low-confidence distractor candidates alongside the selected candidate. At frame 340 the 0.29 raw candidate near the striped player was away from the ball near the blue player and was rejected by the track gate. In LIV frame 365, the baseline selected a spot on the dark-shirted player away from the visible ball, while the dedicated model selected the ball. At frame 390 the dedicated raw score was about 0.19, leaving only a prediction. The samples and coverage figures do not establish precision, recall or real-world possession accuracy.

The first evaluation supported keeping dual-model mode optional. The FCB pass was still missed, FCB possession coverage fell, and the extra inference reduced throughput.

## Passing-clip evaluation

The four new 1080p short clips were evaluated at about 60 source FPS with pitch calibration disabled. Configuration A used YOLO26s for all classes. Configuration B used SoccerNet-v3D for the ball and YOLO26s for people. Both models ran at 960 pixels, with a 0.25 tracker confidence cutoff. SoccerNet also retained raw candidates down to 0.15 for diagnosis. Exported person tracks and team labels matched in all four paired runs.

| Clip | Frames | Observed ball A | Observed ball B | Longest unobserved gap A → B (frames) |
| --- | ---: | ---: | ---: | ---: |
| passes-1 | 628 | 43.9% | 90.3% | 50 → 12 |
| passes-2 | 762 | 42.0% | 79.9% | 122 → 41 |
| passes-3 | 950 | 27.7% | 90.0% | 134 → 15 |
| passes-4 | 600 | 56.3% | 71.7% | 40 → 27 |
| Pooled | 2,940 | 40.7% | 83.7% | |

Frames without a ball track fell from 655 to 54, and processing throughput fell from 16.20 to 11.28 FPS. The developer's review of the output videos found that B tracked the ball significantly better. Manual diagnostics also found one frame in every short clip where B selected the visible ball while A selected a different location. B still produced false candidates, sometimes causing ambiguity. Observed coverage counts selected detector observations and does not measure precision or recall.

Only B ran on the 130.1-second long validation clip. It observed the ball in 5,951 of 7,806 frames (**76.2%**), predicted it in 1,443 and had no ball track in 412. Usable observations covered 76.0%. The longest gap without an observation was 106 frames (1.77 s). Processing ran at 11.34 FPS.

Recommend SoccerNet-v3D as the production ball detector pending final developer approval. YOLO26s remains the default and continues to handle person detections in both modes. [Pass detection](PASSES.md#passing-clip-review) and [possession](POSSESSION.md#passing-clip-review) remain unreliable.

The saved evidence is under `E:\PitchLensStorage\outputs\pass-evaluation-v2`. See `comparison-metrics.json`, `long-metrics.json`, `long-configuration-selection-reviewed.json` and the manual review files in `manual/`.

## Review outputs

Artifacts are under `E:\PitchLensStorage\outputs\soccernet-ball-v1`. Each of the `FCB-dual-960`, `EGYPT-dual-960`, `MAN-dual-960`, `LIV-dual-960` and `FCB-dual-1280-first210` stems has an MP4, a JSON summary and a frame-by-frame JSONL. `comparison-metrics.json` contains full-precision before-and-after metrics, the FCB window counts and absolute file paths. `FCB-baseline-raw-116-187.json` contains the isolated raw baseline detections.

The FCB sheets are `FCB-release-and-travel.jpg`, `FCB-reception.jpg` and `FCB-raw-false-candidate.jpg`. The other sheets are `EGYPT-ball-review.jpg`, `MAN-ball-review.jpg` and `LIV-ball-review.jpg`. Yellow boxes mark raw candidates, green circles mark observed selections, and pink circles mark predictions.
