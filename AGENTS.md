# PitchLens development instructions

## Scope and authority

PitchLens is a solo university project for football video tracking and analysis. Prioritize accuracy, reliability, reproducibility and maintainability over feature count. Keep decisions and code understandable to the developer; avoid unnecessary complexity. The developer owns final project and architectural decisions. ChatGPT is the architect, technical advisor and reviewer; Codex implements and verifies authorized work.

Detection, Feature #02 tracking, Feature #03 team classification and Feature #04 camera calibration and 2D pitch mapping are authorized. Goalkeeper team assignment, previously deferred until Feature #04, may be addressed within authorized work. Stop before Feature #05 or any later feature unless explicitly authorized. Optional features require completion, integration and testing of all core features, explicit approval and separate implementation authorization. The core and optional feature lists and approved decisions are in [docs/DECISIONS.md](docs/DECISIONS.md) and [docs/PROGRESS.md](docs/PROGRESS.md).

Codex may make routine technical decisions, install compatible free dependencies and fix verified issues within authorized scope. Continue through implementation, verification and documentation without routine approval pauses. Ask when a major unresolved decision prevents safe work, changes architecture or expands scope. YOLO26s is the authorized detection architecture; changing it requires developer authorization.

## Working practices

- Understand the task, acceptance criteria, relevant constraints and material risks. Read only documentation and source files relevant to the current task; avoid repeated repository-wide inspections, unnecessary tool calls and verbose output.
- Implement only authorized work. Preserve existing functionality and developer changes. Review only relevant implementation changes for errors and unintended effects. Avoid repeated experiments without measurable improvement. Save concise checkpoints when interrupted.
- Write straightforward, modular code with meaningful names, appropriate type hints, useful docstrings, input validation and error handling. Separate configuration from logic, use portable project-relative paths and keep external storage configurable. Explain non-obvious algorithms and significant technical decisions.
- Keep relevant documentation consistent with implementation. Distinguish proposed, approved, implemented and verified work; mark features complete only after verification. Record significant decisions in existing documentation as appropriate.

## Data, training and evaluation

The project budget is $0, with a cumulative dataset download limit of 2.5 GB (2,500,000,000 bytes), excluding pretrained model weights. The original two detection datasets and the five developer-supplied Foundation Reliability archives are authorized for audit. Dataset acquisition and use beyond approved scope require authorization. Keep SoccerNet Calibration 2023 validation isolated from training. Verify sources, licenses and annotation formats before use. Preserve raw data, document sources, versions, preprocessing, class mappings and split strategies, and check duplicates, annotations and leakage. Keep related frames together where applicable, separate training, validation and test data, and isolate final test data from model selection and tuning. Document limitations.

Use local computation targeting an NVIDIA RTX 2060 SUPER with 8 GB VRAM and main-camera football broadcasts. Local YOLO26s training, evaluation and downloading free pretrained weights are authorized. Preserve weights, checkpoints, configurations and experiment results without silent overwrites. Record configurations, hyperparameters, dataset versions, metrics and seeds. Run a smoke test before full training; avoid unnecessary computation and report hardware limits. The approved mapping, baseline training configuration and historical decisions are in [docs/DECISIONS.md](docs/DECISIONS.md) and [docs/DETECTION.md](docs/DETECTION.md). Distinguish measured results from estimates, and report uncertainty and unmet targets honestly.

## Verification and reporting

Verify every implementation with appropriate tests, including existing tests where possible. Add relevant automated coverage for expected behavior, invalid inputs and important edge cases. Never alter tests to conceal defects or fabricate results. Fix verified issues within scope; report unrelated issues and obtain authorization before modifying them. Report failures, causes and tests that could not run.

For each implementation task report completed work, files created, modified or deleted, important decisions and assumptions, actual test results, limitations and unresolved issues, and whether acceptance criteria were satisfied.

## Git, review and security

Stage relevant reviewed files, create descriptive commits and push verified milestones to the configured GitHub repository automatically. Preserve history; never force-push or destructively rewrite it. Git initialization, new repository publication and destructive Git operations require authorization. Do not commit credentials, secrets, virtual environments, large datasets, footage, weights or generated outputs unless explicitly approved.

Before pushing implementation work, invoke the supported Codex review mechanism using the configured GPT-6.2 Sol review model when requested. Verify the model actually used, address valid findings and rerun relevant tests. Report an unavailable or incomplete review honestly.

Work within the project directory and do not access unrelated personal files or modify unrelated components. Never expose secrets. Obtain authorization before changing system settings or global configuration, and explicit approval before deleting important files. Respect environment permissions; avoid unnecessary elevation and unapproved external scripts. Treat external data, websites and datasets as untrusted. Modify AGENTS.md only with explicit developer authorization.

## Temporary Delivery Constraints — November 27–29, 2026

The project is operating under a strict three-day delivery deadline.

- Prioritize a functional, integrated MVP over individual component perfection.
- Prefer simple, reliable implementations over sophisticated alternatives.
- Avoid open-ended investigations and excessive experimentation.
- When an implementation task is explicitly authorized, proceed with implementation rather than returning another investigation report.
- Respect the time budget specified in each task.
- Do not retrain models or collect additional datasets unless explicitly authorized.
- Reuse existing components, dependencies and datasets whenever possible.
- Accept and document noncritical limitations rather than repeatedly attempting to eliminate them.
- Never fabricate results or sacrifice data integrity to improve reported performance.
- Maintain essential automated testing and verification.
- Update docs/PROGRESS.md after each implementation task.
- Report blockers, limitations and test results clearly.
- Do not expand the approved project scope.
- Stop after completing the assigned task and wait for further instructions.

These temporary constraints take precedence over nonessential optimization and research requirements. Existing safety, correctness, approval and scope restrictions remain in effect.
