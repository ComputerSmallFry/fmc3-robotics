#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/phl/workspace/lerobot-versions/lerobot"
CONDA_ENV="lerobot-pi0"
CHECKPOINT_PATH="/home/phl/workspace/lerobot-versions/lerobot/outputs/train/pi0_gr2_pick_3_4_20260304_172720/checkpoints/060000/pretrained_model"
TASK="pick bottle and place into box"
ROBOT_TYPE="fourier_gr2"

FPS="${FPS:-30}"
DOMAIN_ID="${DOMAIN_ID:-123}"
ROBOT_NAME="${ROBOT_NAME:-gr2}"
FSM_STATE="${FSM_STATE:-11}"
ENABLE_GUI="${ENABLE_GUI:-0}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/scripts/outputs/deploy_gr2_pi0_multivis}"
SAVE_EVERY="${SAVE_EVERY:-30}"
CLIENT_INIT_RETRIES="${CLIENT_INIT_RETRIES:-10}"
CLIENT_RETRY_INTERVAL_S="${CLIENT_RETRY_INTERVAL_S:-2.0}"
MAX_STEPS="${MAX_STEPS:-0}"
ACTION_EMA_ALPHA="${ACTION_EMA_ALPHA:-0.35}"
MAX_ARM_DELTA="${MAX_ARM_DELTA:-0.06}"
MAX_HAND_DELTA="${MAX_HAND_DELTA:-0.12}"
MAX_HEAD_WAIST_DELTA="${MAX_HEAD_WAIST_DELTA:-0.08}"
MAX_BASE_DELTA="${MAX_BASE_DELTA:-0.15}"
SLOW_LOOP_WARN_MS="${SLOW_LOOP_WARN_MS:-120}"

cd "${ROOT_DIR}"

if [[ ! -e "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] checkpoint not found: ${CHECKPOINT_PATH}"
  exit 1
fi

EXTRA_ARGS=()
if [[ "${ENABLE_GUI}" != "1" ]]; then
  EXTRA_ARGS+=(--no-gui)
fi

set -x
conda run --no-capture-output -n "${CONDA_ENV}" python scripts/deploy_gr2_pi0_multivis.py \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --task "${TASK}" \
  --robot-type "${ROBOT_TYPE}" \
  --fps "${FPS}" \
  --domain-id "${DOMAIN_ID}" \
  --robot-name "${ROBOT_NAME}" \
  --fsm-state "${FSM_STATE}" \
  --camera-key "observation.images.camera_top" \
  --client-init-retries "${CLIENT_INIT_RETRIES}" \
  --client-retry-interval-s "${CLIENT_RETRY_INTERVAL_S}" \
  --action-ema-alpha "${ACTION_EMA_ALPHA}" \
  --max-arm-delta "${MAX_ARM_DELTA}" \
  --max-hand-delta "${MAX_HAND_DELTA}" \
  --max-head-waist-delta "${MAX_HEAD_WAIST_DELTA}" \
  --max-base-delta "${MAX_BASE_DELTA}" \
  --slow-loop-warn-ms "${SLOW_LOOP_WARN_MS}" \
  --save-dir "${SAVE_DIR}" \
  --save-every "${SAVE_EVERY}" \
  --max-steps "${MAX_STEPS}" \
  "${EXTRA_ARGS[@]}"
