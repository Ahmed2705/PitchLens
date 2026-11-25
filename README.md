# PitchLens

PitchLens is a solo university project for AI-powered football tracking and match analysis. It aims to turn match footage into player, team, and match statistics using computer vision and machine learning.

## Project objectives

- Detect and track the people and ball in match footage.
- Map movement onto a 2D pitch to measure player activity and identify match events.
- Present the resulting statistics and visualizations in an interactive dashboard.
- Make experiments and evaluation reproducible within the scope of a solo university project.

## Approved core features

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

## Development status

PitchLens is in the foundational setup stage. The initial source layout and project documentation are in place. The football analytics features have not yet been implemented. See the [progress tracker](docs/PROGRESS.md) for upcoming work.

## Preliminary technology stack

Candidates subject to approval include Python, YOLO, ByteTrack, OpenCV, PyTorch, scikit-learn, K-means, NumPy, pandas, mplsoccer, and Streamlit. The stack, model selections, and dependency versions remain open.

## Repository layout

| Path | Responsibility |
| --- | --- |
| `src/pitchlens/` | Main Python application package |
| `data/` | Datasets and input match footage |
| `models/` | Model weights and associated documentation |
| `configs/` | Application, model, and experiment configuration |
| `notebooks/` | Exploratory analysis and research experiments |
| `scripts/` | Standalone training, evaluation, and data preparation utilities |
| `tests/` | Automated tests and test fixtures |
| `outputs/` | Generated videos, predictions, statistics, and visualizations |
| `docs/` | Technical documentation, approved decisions, and development progress |

## Development approach

Each feature will be authorized, implemented, and verified incrementally. The project will prioritize accuracy, reliability, and reproducibility, including documented dataset splits and experiment settings when applicable. Approved architectural decisions are recorded in the [technical decision record](docs/DECISIONS.md).
