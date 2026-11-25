# Development progress

PitchLens is in its foundational setup stage.

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

## Pending setup and release work

- Dependency management and package configuration.
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
