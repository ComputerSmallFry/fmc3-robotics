#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${SCRIPT_DIR}/convert_dora_to_lerobot.py" \
    --input "${REPO_ROOT}/gr2-pick-3-4" \
    --output "${REPO_ROOT}/gr2-pick-3-4_lerobot_gr2" \
    --task "pick bottle and place into box" \
    --fps 30 \
    --robot-type fourier_gr2 \
    --video-codec libopenh264 \
    --workers 8
