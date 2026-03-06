#!/usr/bin/env bash
set -euo pipefail

# Resume PI0 training from step 60000 and target step 111000
# (estimated to run until around next morning 10:00 based on current speed).

cd /home/phl/workspace/lerobot-versions/lerobot

RESUME_CONFIG_PATH="/home/phl/workspace/lerobot-versions/lerobot/outputs/train/pi0_gr2_pick_3_4_20260304_172720/checkpoints/060000/pretrained_model/train_config.json" \
STEPS="111000" \
SAVE_FREQ="20000" \
LOG_FREQ="50" \
BATCH_SIZE="8" \
bash scripts/train/train_pi0_gr2_pick.sh "$@"
