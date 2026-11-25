# Technical decisions

## Approved project scope

PitchLens is a solo university project. The approved core scope consists of these ten features.

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

Optional future features are tracked separately in [PROGRESS.md](PROGRESS.md#optional-future-features). They can enter implementation after the core features are complete, integrated, and tested. Each optional feature requires explicit approval and separate implementation authorization.

## Approved development workflow

- VS Code is the development environment. The developer runs Codex CLI in Windows CMD for implementation and testing. ChatGPT serves as technical advisor, architect, and reviewer.
- The developer owns final architectural and project decisions and has delegated routine technical decisions and execution within authorized feature scope.
- Codex sessions use workspace permissions with approval on request. The developer manually verifies these settings when starting a session.
- Git and GitHub provide version control, with `main` as the primary branch. Codex may stage reviewed, relevant files, commit verified milestones, and push to the configured repository; force-pushes and destructive history rewrites are prohibited.

## Approved repository layout

- The Python source package uses the `src/pitchlens/` layout. Feature-specific modules and subdirectories are added incrementally as their implementations are authorized.
- The repository uses the approved `.gitignore` strategy to exclude local environments and caches, credentials, large dataset payloads and match footage, model weights, and generated outputs. Dataset and model documentation and configuration can be tracked outside ignored payload folders.

## Approved Python environment

- Python 3.13.7 is the approved development version. The supported range is `>=3.13,<3.14`.
- Use `venv` to manage the virtual environment and `pip` to manage packages.
- The virtual environment lives in `.venv/` at the repository root. The `.venv/` rule in `.gitignore` excludes it from Git.

## Approved packaging and dependency management

- Use Setuptools to package `pitchlens` from `src/pitchlens/`. The initial version is `0.1.0`, and development uses editable installation.
- Add compatible free direct dependencies as needed within authorized scope and test them. Keep runtime pins consistent across `pyproject.toml` and `requirements.txt`.
- Pin PyTorch to `2.11.0` and TorchVision to `0.26.0` using public versions. The local environment uses the CUDA 12.8 (`cu128`) builds. The README documents separate installation commands for CUDA and CPU wheels.
- Use an NVIDIA GPU as the primary computation device where supported. CPU execution support is required. The current CUDA wheel setup requires a compatible NVIDIA driver, with no separate CUDA Toolkit or cuDNN installation required.
- Keep storage locations for large machine learning assets configurable so developers can use external storage.

## Approved detection dataset acquisition

- Acquire and inspect only the two approved Roboflow Universe projects for YOLO26s detection on main-camera football broadcasts. [Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) and [Football Object Detection](https://universe.roboflow.com/football-detect/football-object-detection).
- Use free YOLO object detection exports with a combined download limit of 1 GB. Verify both archive sizes before downloading. Prefer non-augmented, original-resolution versions where available.
- Preserve each export separately in the project's approved raw storage, including its images, annotations, metadata, class IDs, and source splits. The required final class mapping for a later preparation phase is `0: player`, `1: goalkeeper`, `2: referee`, `3: ball`.
- The developer supplied the Football Players Detection v10 and Football Object Detection v1 archives. Their embedded Roboflow metadata identifies both as YOLO26 exports under CC BY 4.0. The v1 export includes provider-generated crop augmentation and polygon annotations for `goal`. [Version 10](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/10) and [version 1](https://universe.roboflow.com/football-detect/football-object-detection/dataset/1).
- The approved local raw dataset directory is `E:\PitchLensStorage\datasets\raw`. Keep the raw storage location configurable for each machine. The two exports are stored separately in `football-players-detection-v10` and `football-object-detection-v1`, and the original ZIPs are retained in Windows Downloads.

## Delegated object detection implementation

- The complete object detection feature is authorized through dataset preparation, leakage-aware splitting, training, evaluation, module implementation and verification. Continue between its routine stages without manual approval; stop before object tracking for architectural review.
- Use local computation, a $0 project budget, a total dataset download limit of 1 GB, and the RTX 2060 SUPER with 8 GB VRAM as the primary hardware. Keep external storage paths configurable and never commit datasets, weights or generated experiments.
- Prepare the approved exports with the mapping `0: player`, `1: goalkeeper`, `2: referee`, `3: ball`, removing goal and corner annotations. Preserve the raw exports. Target a reproducible 80/10/10 split while keeping known related frames and augmentations together and documenting provenance limits.
- Train YOLO26s initialized from free pretrained `yolo26s.pt` at 640 pixels, with CUDA, AMP, automatic batch sizing and optimizer selection, at most 100 epochs, patience 20, and seed 42. Run a smoke test first; consider 960 or 1280 only when justified and feasible.
- Treat the supplied performance targets as provisional. Never tune with the held-out test set; report measured results and limitations.
- The v1 export's 242 distinct source-frame identifiers all occur in the 312-image v10 export. Group all copies by source prefix across both exports; merge visually identified stadiums conservatively and merge cross-prefix perceptual-hash candidates. Use only original v10 images for validation and test. Select held-out groups using original-frame and class counts before training, favoring held-out group diversity when feasible. The target split applies to unique original frames; augmented images are training-only.
- A closer review discovered that the first split placed footage from the same stadium across splits. The first baseline was stopped after 11 epochs; its checkpoints are preserved but will not be used for the final model. The revised final test excludes stadium groups previously used for validation, and the new baseline starts from the original pretrained weights.
- Two frames in the revised test group were inadvertently used to smoke-test an earlier checkpoint. Exclude those preselected frame IDs from final scoring; no model or threshold was selected from their predictions. The final dataset's training and validation images and labels were verified byte-for-byte identical to those used by the running baseline.

## Preliminary recommendations

The candidates are YOLO, ByteTrack, OpenCV, scikit-learn, K-means, NumPy, pandas, mplsoccer, and Streamlit. This list is subject to developer approval and does not establish dependencies.

## Open decisions

The project license and implementation details outside the authorized detection scope remain open.
