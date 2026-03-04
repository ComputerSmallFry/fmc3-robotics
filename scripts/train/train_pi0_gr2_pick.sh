#!/usr/bin/env bash
set -euo pipefail

# GR2 pi0 training config (local dataset first)
CONDA_ENV="${CONDA_ENV:-lerobot-pi0}"
DATASET_REPO_ID="${DATASET_REPO_ID:-puheliang/gr2-pick-3-4-lerobot-gr2}"
DATASET_ROOT="${DATASET_ROOT:-/home/phl/workspace/dataset/fourier/gr2-pick-3-4_lerobot_gr2}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/home/phl/workspace/models/pi0}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="100000"
# Fixed: save checkpoint every 20000 steps.
SAVE_FREQ="20000"
LOG_FREQ="${LOG_FREQ:-50}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-35}"
MAX_STATE_DIM="${MAX_STATE_DIM:-45}"
DRY_RUN="${DRY_RUN:-false}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-/home/phl/workspace/lerobot-versions/lerobot/outputs/train/pi0_gr2_pick_3_4_${TS}}"
JOB_NAME="${JOB_NAME:-pi0_gr2_pick_3_4}"
LOG_DIR="${LOG_DIR:-/home/phl/workspace/lerobot-versions/lerobot/outputs/train_logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/pi0_gr2_pick_3_4_${TS}.log}"

cd /home/phl/workspace/lerobot-versions/lerobot
mkdir -p "${LOG_DIR}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "[ERROR] Dataset root not found: ${DATASET_ROOT}"
  exit 1
fi

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "[ERROR] Missing ${DATASET_ROOT}/meta/info.json"
  exit 1
fi

if [[ ! -e "${PRETRAINED_PATH}" ]]; then
  echo "[ERROR] Pretrained path not found: ${PRETRAINED_PATH}"
  exit 1
fi

FREE_GB="$(df --output=avail -BG /home/phl | tail -n1 | tr -dc '0-9')"
if [[ -n "${FREE_GB}" && "${FREE_GB}" -lt 25 ]]; then
  echo "[ERROR] Free disk is only ${FREE_GB}G (<25G). Please free space before training."
  exit 1
fi

echo "[INFO] env=${CONDA_DEFAULT_ENV:-base} target_env=${CONDA_ENV}"
echo "[INFO] dataset_repo_id=${DATASET_REPO_ID}"
echo "[INFO] dataset_root=${DATASET_ROOT}"
echo "[INFO] pretrained_path=${PRETRAINED_PATH}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] log_file=${LOG_FILE}"
echo "[INFO] free_disk_gb=${FREE_GB:-unknown}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
  TRAIN_CMD=(lerobot-train)
else
  TRAIN_CMD=(conda run --no-capture-output -n "${CONDA_ENV}" lerobot-train)
fi

TRAIN_ARGS=(
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.streaming=false"
  "--dataset.video_backend=${VIDEO_BACKEND}"
  "--policy.type=pi0"
  "--policy.pretrained_path=${PRETRAINED_PATH}"
  "--policy.device=cuda"
  "--policy.dtype=bfloat16"
  "--policy.gradient_checkpointing=true"
  "--policy.compile_model=false"
  "--policy.max_action_dim=${MAX_ACTION_DIM}"
  "--policy.max_state_dim=${MAX_STATE_DIM}"
  "--policy.push_to_hub=false"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--wandb.enable=false"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${JOB_NAME}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, command preview:"
  printf '%q ' PYTHONUNBUFFERED=1 stdbuf -oL -eL "${TRAIN_CMD[@]}" "${TRAIN_ARGS[@]}"
  echo
  exit 0
fi

set -x
PYTHONUNBUFFERED=1 stdbuf -oL -eL "${TRAIN_CMD[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
