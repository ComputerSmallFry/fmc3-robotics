#!/bin/bash
# ============================================================
# pi0 微调训练脚本 - 抓取物体放入盒子
# 数据集: puheliang/lerobot_fmc3_grab_box_v2
# 基座模型: lerobot/pi0
# 机械臂: SO-101 (leader + follower)
# 相机: top + wrist
# 任务: Pick up the tape and place it in the box
# ============================================================

set -e

# ---------- 环境配置 ----------
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/huggingface_cache
source /etc/network_turbo 2>/dev/null || true

# ---------- 训练参数（可按需修改） ----------
DATASET_REPO="puheliang/lerobot_fmc3_grab_box_v2"
PRETRAINED_PATH="lerobot/pi0"
OUTPUT_DIR="/root/output_lerobot_train/grab_box/pi0_fmc3"
JOB_NAME="grab_box_pi0_fmc3"
STEPS=50000
BATCH_SIZE=8
DTYPE="bfloat16"
SAVE_FREQ=10000

# Hub 推送配置
PUSH_TO_HUB=true
HUB_REPO_ID="puheliang/pi0_fmc3_grab_box_v2"

# WandB 配置
WANDB_ENABLE=true
WANDB_PROJECT="Lerobot_Fmc_Project"

# ---------- 清理旧输出（首次训练） ----------
if [ "$1" != "--resume" ]; then
    echo "清理旧输出目录: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
fi

# ---------- 构建训练命令 ----------
CMD="lerobot-train \
    --dataset.repo_id=${DATASET_REPO} \
    --dataset.streaming=false \
    --policy.type=pi0 \
    --output_dir=${OUTPUT_DIR} \
    --job_name=${JOB_NAME} \
    --policy.pretrained_path=${PRETRAINED_PATH} \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=${DTYPE} \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=${STEPS} \
    --policy.device=cuda \
    --policy.push_to_hub=${PUSH_TO_HUB} \
    --policy.repo_id=${HUB_REPO_ID} \
    --batch_size=${BATCH_SIZE} \
    --save_freq=${SAVE_FREQ} \
    --wandb.enable=${WANDB_ENABLE} \
    --wandb.project=${WANDB_PROJECT}"

# 断点续训
if [ "$1" == "--resume" ]; then
    CMD="${CMD} --resume=true"
    echo "断点续训模式"
fi

echo "========================================"
echo "开始训练: ${JOB_NAME}"
echo "数据集: ${DATASET_REPO}"
echo "基座模型: ${PRETRAINED_PATH}"
echo "输出目录: ${OUTPUT_DIR}"
echo "总步数: ${STEPS}"
echo "========================================"

eval ${CMD}
