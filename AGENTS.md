# Repository Guidelines

## Project Structure & Module Organization
This repository focuses on converting Fourier Dora-Record episodes into LeRobot v3.0 datasets.

- `scripts/convert_tools/`: core conversion code and docs.
- `scripts/convert_tools/convert_dora_to_lerobot.py`: main pipeline (scan episodes, resample, encode video, write LeRobot format).
- `scripts/convert_tools/run_convert.sh`: preset conversion command.
- `pick_bottle_and_place_into_box/`: Dora input dataset (raw episodes).
- `*_lerobot*/`: generated outputs (`meta/`, `data/`, `videos/`).

Keep new conversion logic under `scripts/convert_tools/` and keep dataset-specific experiments in separate scripts/files.

## Build, Test, and Development Commands
No build system is required; this is script-driven.

- `conda activate lerobot`: activate expected environment.
- `bash scripts/convert_tools/run_convert.sh`: run the preset GR2 conversion workflow.
- `python scripts/convert_tools/convert_dora_to_lerobot.py --input ./pick_bottle_and_place_into_box --output ./pick_bottle_and_place_into_box_lerobot_gr2 --task "pick bottle and place into box" --fps 30 --robot-type fourier_gr2 --video-codec libopenh264 --workers 4`: run manually with explicit parameters.
- `python scripts/convert_tools/convert_dora_to_lerobot.py ... --no-video`: fast validation run without image/video encoding.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants.
- Prefer `pathlib.Path`, type hints, and small focused helper functions.
- Keep comments concise and technical (data shape, ordering, assumptions).
- Shell scripts should remain POSIX-compatible where possible and executable (`chmod +x`).

## Testing Guidelines
There is no formal automated test suite yet. Validate changes with:

1. A small-scope conversion run (`--no-video` first, then full run if needed).
2. Output checks: ensure `meta/info.json`, `meta/stats.json`, `data/chunk-000/file-000.parquet` are generated.
3. Schema sanity: confirm action/state dimensions match robot type expectations (for GR2: action 35D, state 45D).

## Commit & Pull Request Guidelines
Recent commits use short, imperative Chinese summaries (for example: `添加...`, `修复...`, `更新...`). Follow that style and keep each commit focused.

For PRs, include:
- What changed and why.
- Exact command(s) used for validation.
- Any dataset path assumptions or environment constraints (`ffmpeg` codec, conda env).
- Before/after behavior notes (for example conversion speed, memory, or output schema changes).

## Data & Configuration Notes
- Do not commit large raw/generated datasets. `.gitignore` already excludes `pick_bottle_and_place_into_box/` and `*_lerobot*/`.
- Prefer relative paths in new scripts; avoid user-home absolute paths in shared automation.
