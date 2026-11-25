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

The object detection baseline has been trained and evaluated on held-out broadcast frames, and the detector has been tested on CPU and CUDA. Several provisional accuracy targets remain unmet, and video-level verification is pending because no footage was available. See the [progress tracker](docs/PROGRESS.md) and [detection results](docs/DETECTION.md) before using the model.

## Preliminary technology stack

Python 3.13, PyTorch 2.11.0, TorchVision 0.26.0, Ultralytics YOLO26s, NumPy and OpenCV are used for the authorized detection work. Other proposed features and technologies remain outside the current implementation task.

## Preliminary environment setup

These instructions are preliminary until independently tested. PitchLens requires Python `>=3.13,<3.14`, and the development version is 3.13.7. From the repository root in Windows PowerShell, check that `python` points to a supported interpreter, then create and activate the environment.

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Select exactly one PyTorch installation option for your hardware.

**NVIDIA GPU with CUDA 12.8.** This option requires compatible NVIDIA hardware and drivers. The PyTorch wheels provide the CUDA runtime dependencies, so a standalone CUDA Toolkit is currently unnecessary.

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

**CPU-only.** Use this option for a CPU installation. The frame detector automatically falls back to CPU when CUDA is unavailable.

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

After installing the CPU or CUDA builds, install the project requirements and install PitchLens in editable mode.

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify that Python can locate the package and that the environment's dependencies are consistent.

```powershell
python -c "import pitchlens; print(pitchlens.__file__)"
python -m pip check
```

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

The current delegated task covers detection through verification, followed by architectural review before tracking. The project prioritizes accuracy, reliability, and reproducibility, including documented dataset splits and experiment settings. Decisions are recorded in the [technical decision record](docs/DECISIONS.md), and the detection data workflow is in [DETECTION.md](docs/DETECTION.md).
