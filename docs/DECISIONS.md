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
- The developer owns architectural and project decisions and explicitly authorizes each implementation task.
- Codex sessions use workspace permissions with approval on request. The developer manually verifies these settings when starting a session.
- Git and GitHub provide version control, with `main` as the primary branch. The developer stages, commits, and pushes manually through VS Code.

## Approved repository layout

- The Python source package uses the `src/pitchlens/` layout. Feature-specific modules and subdirectories are added incrementally as their implementations are authorized.
- The repository uses the approved `.gitignore` strategy to exclude local environments and caches, credentials, large dataset payloads and match footage, model weights, and generated outputs. Dataset and model documentation and configuration can be tracked outside ignored payload folders.

## Approved Python environment

- Python 3.13.7 is the approved development version. The supported range is `>=3.13,<3.14`.
- Use `venv` to manage the virtual environment and `pip` to manage packages.
- The virtual environment lives in `.venv/` at the repository root. The `.venv/` rule in `.gitignore` excludes it from Git.

## Approved packaging and dependency management

- Use Setuptools to package `pitchlens` from `src/pitchlens/`. The initial version is `0.1.0`, and development uses editable installation.
- Add direct dependencies incrementally as the developer approves them and they are tested. Keep the runtime pins consistent across `pyproject.toml` and `requirements.txt`. The developer manages package installation manually through the VS Code terminal.
- Pin PyTorch to `2.11.0` and TorchVision to `0.26.0` using public versions. The local environment uses the CUDA 12.8 (`cu128`) builds. The README documents separate installation commands for CUDA and CPU wheels.
- Use an NVIDIA GPU as the primary computation device where supported. CPU execution support is required. The current CUDA wheel setup requires a compatible NVIDIA driver, with no separate CUDA Toolkit or cuDNN installation required.
- Keep storage locations for large machine learning assets configurable so developers can use external storage.

## Preliminary recommendations

The candidates are YOLO, ByteTrack, OpenCV, scikit-learn, K-means, NumPy, pandas, mplsoccer, and Streamlit. This list is subject to developer approval and does not establish dependencies.

## Open decisions

The license, additional dependencies and their versions, model selection, and feature implementation details remain open.
