#!/bin/bash
# 恢复PI0训练 - 在tmux中运行防止终端断开
# 从checkpoint 020000恢复，训练到85000步
# 预计用时约15小时 (18:00 -> 09:00)

set -e

# 初始化conda
source /home/phl/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-pi0

cd /home/phl/workspace/lerobot-versions/lerobot

echo "=========================================="
echo "  PI0 恢复训练"
echo "  开始时间: $(date)"
echo "  目标步数: 85000"
echo "  恢复自: checkpoint 020000"
echo "=========================================="

lerobot-train \
    --policy.path=/home/phl/workspace/models/pi0 \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.compile_model=true \
    --policy.compile_mode=max-autotune \
    --dataset.repo_id=fourier_gr2_pick_place \
    --dataset.root=/home/phl/workspace/dataset/fourier/pick_and_place \
    --dataset.video_backend=torchcodec \
    --output_dir=outputs/train/fourier_gr2_pi0 \
    --batch_size=12 \
    --steps=85000 \
    --save_freq=10000 \
    --log_freq=100 \
    --num_workers=8 \
    --resume=true \
    2>&1 | tee outputs/train/fourier_gr2_pi0/train.log

echo "=========================================="
echo "  训练完成: $(date)"
echo "=========================================="
