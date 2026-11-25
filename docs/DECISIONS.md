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

## Preliminary recommendations

The candidates are Python, YOLO, ByteTrack, OpenCV, PyTorch, scikit-learn, K-means, NumPy, pandas, mplsoccer, and Streamlit. This list is subject to developer approval and does not establish dependencies.

## Open decisions

Python environment setup, package configuration, dependency management, model selection, dependency versions, and feature implementation details remain open.
