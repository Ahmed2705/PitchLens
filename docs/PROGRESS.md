# Development progress

PitchLens has completed foundational setup and is beginning the authorized object detection feature.

## Delegated development workflow

- The project owner authorized autonomous execution of the complete detection feature, including compatible free dependencies, dataset preparation, local training, testing, documentation, and verified milestone commits and pushes.
- Work stays within the approved scope, $0 budget and 1 GB total dataset download limit. External data paths remain configurable; raw datasets, model weights and generated experiments remain outside Git.
- Detection is not yet complete or verified. Stop at the detection feature boundary for architectural review before beginning tracking.

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

The two manually downloaded YOLO26 exports in Windows Downloads total 136,835,956 bytes, below the 1 GB limit. Both archives passed CRC validation, every image decoded successfully, and every image had a nonempty label file. No orphaned label files were found.

| Export | ZIP bytes | Extracted bytes | Extracted files | Train / validation / test | Image resolution |
| --- | ---: | ---: | ---: | --- | --- |
| Football Players Detection v10 | 71,896,874 | 72,158,240 | 627 | 250 / 43 / 19 | 312 at 1920 × 1080 |
| Football Object Detection v1 | 64,939,082 | 65,312,489 | 2,303 | 1,000 / 100 / 50 | 1,150 at 640 × 640 |

- Football Players Detection v10 contains 7,454 bounding boxes. Source IDs and counts are `0: ball` (272), `1: goalkeeper` (235), `2: player` (6,229), and `3: referee` (718). One referee box extends less than one pixel past the image edge. The export reports no preprocessing or augmentation. [Roboflow version 10](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/10).
- Football Object Detection v1 contains 28,275 bounding boxes and 506 polygons. Source IDs and counts are `0: ball` (977 boxes), `1: corner` (961 boxes), `2: goal` (506 polygons), `3: goalkeeper` (760 boxes), `4: player` (22,964 boxes), and `5: referee` (2,613 boxes). The ZIP retains all three byte-identical `data.yaml` entries. One copy was extracted to their shared path without overwriting. Roboflow applied auto-orientation, resized to 640 × 640, and added crop augmentation. [Roboflow version 1](https://universe.roboflow.com/football-detect/football-object-detection/dataset/1).
- Image SHA-256 checks found three exact duplicate groups in v1. All were within individual splits. No exact image matches were found between splits or between exports.
- Extraction is complete in `football-players-detection-v10` and `football-object-detection-v1` under `E:\PitchLensStorage\datasets\raw`. SHA-256 comparisons matched all 2,930 extracted files to their ZIP entries and reconfirmed both original ZIPs in Downloads. Recounts matched the original split, image resolution, and annotation totals. The external folders are outside any Git worktree, and Git ignored all corresponding paths under the repository's `data/raw/`.

## Detection dataset preparation and splitting — verified

- Prepared both exports without modifying raw data. The v1 export contains variants of 242 v10 frames, rather than independent frames. Fifteen source prefixes were reduced to 12 conservative groups after a visual review and perceptual-hash investigation; the exact original matches are not identified in the exports.
- Audited 1,462 input images and annotations. Removed 961 corner boxes and 506 goal polygons, clipped one sub-pixel edge overflow, found three exact duplicate groups and no duplicate annotation lines. Two duplicate images were removed from training and the remaining duplicate was among excluded held-out augmentations.
- Split the 312 original frames into 251 training, 31 validation and 30 test images. The training set also contains augmented copies, for 1,203 total training images; validation and test contain original images only. Excluded 196 held-out augmentations. Each split contains all four classes, and no known source group, frame ID or exact image hash crosses splits.
- Training class counts are 24,033 player, 806 goalkeeper, 2,738 referee and 1,008 ball boxes. Of 272 original ball boxes, 270 are under 8 pixels wide and high when scaled to a 640-pixel input. The small, correlated held-out sets and uncertain match provenance limit generalization claims.
- Two independent preparations produced identical manifests and reports. Seven automated preparation tests passed. Dataset details and commands are documented in [DETECTION.md](DETECTION.md).

## Detection training and evaluation — in progress

- Official pretrained `yolo26s.pt` downloaded locally; SHA-256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`. It is excluded from Git.
- A one-epoch CUDA smoke test on 60 training images and 31 validation images completed with finite losses and metrics, AMP enabled, automatic batch size 12, and peak PyTorch allocated GPU memory of 3,518,951,424 bytes. The smoke results are not model performance claims.
- The 640-pixel baseline has started with a maximum of 100 epochs and patience 20. Final training, final evaluation, detector and inference verification are not yet complete.

## Pending setup and release work

- Additional dependency selection.
- Automatic CPU fallback implementation and testing.
- Fresh-clone setup testing.
- Final integration and testing.
- Installation and usage documentation.
- Final portability and reproducibility testing.

## Pending core features

1. Player, goalkeeper, referee and ball detection
2. Player and ball tracking
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
