# Repository Guidelines

This guide helps contributors work efficiently on this auto-recut toolkit. Keep changes focused, small, and well‑described.

## Project Structure & Module Organization
- `recut/` core Python package
  - `cli.py`: command-line entry (`python -m recut.cli`)
  - `ui_tk.py`: Tkinter GUI (`python -m recut.ui_tk`)
  - `pipeline.py`: end‑to‑end align → cut → concat
  - `scene_detect.py`: PySceneDetect integration + HSV fallback
  - `features.py`: CLIP/HSV features, masking hooks
  - `dtw.py`: DTW alignment utilities
  - `ffmpeg_utils.py`: normalize, segment cut, concat
- `requirements.txt`: minimal deps (NumPy, OpenCV, tqdm)
- `README.md`: usage docs; prefer updating when behavior changes
- `out/`: default outputs (alignment.json, segments/, recut_output.mp4)
- `source/`: example media (not tracked in VCS typically)

## Build, Test, and Development Commands
- Create venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Optional (CLIP/PySceneDetect): `python -m pip install torch torchvision transformers scenedetect`
- Run CLI: `python -m recut.cli --ref ref.mp4 --src src.mp4 --out out --render`
- Run GUI: `python -m recut.ui_tk`
- Render from existing alignment: `python -m recut.cli --ref ref.mp4 --src src.mp4 --out out --from-alignment`

## Coding Style & Naming Conventions
- Python 3.10+ with type hints; use descriptive names (`match_start_t`, not `s1`).
- Keep functions small and composable; prefer pure helpers in `recut/*`.
- Follow repository patterns for logging (`log()` callback) and progress hooks.
- File/CLI options use kebab‑case flags (e.g., `--global-search`).

## Testing Guidelines
- No formal test suite yet. Add targeted unit tests under `tests/` mirroring module names.
- Smoke tests: run CLI on short clips; verify `out/alignment.json` and total duration equals reference sum.
- Prefer deterministic seeds and short samples for CI.

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject + rationale in body (e.g., "Add ffconcat duration to lock final length").
- PRs include: purpose, screenshots or logs for UX/CLI, reproduction steps, and impact on `README.md`.
- Keep PRs focused (single feature/fix); note breaking changes prominently.

## Security & Configuration Tips
- Requires `ffmpeg` in PATH. Large media should not be committed.
- Features cache in `out/` may include large `.npz`; safe to delete/regenerate.
- Offline-friendly by default; CLIP model download needs network.
閱讀README.md然後開始你的實作 輕以繁體中文回覆用戶
