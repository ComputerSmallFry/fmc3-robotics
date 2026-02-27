#!/bin/bash
# ACT 策略训练 — 抓取胶带放入盒子任务
# 数据集: puheliang/lerobot_fmc3_grab_box_v2 (SO-101, 401 episodes)
# 硬件: RTX 4090

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=~/output_lerobot_train/grab_box/act
KEEP_LAST=2

# 后台清理：每60秒检查一次，只保留最新的 KEEP_LAST 个 checkpoint
(
    while true; do
        sleep 60
        checkpoints=($(ls -dt "$OUTPUT_DIR"/checkpoints/*/  2>/dev/null))
        if [ ${#checkpoints[@]} -gt $KEEP_LAST ]; then
            for ckpt in "${checkpoints[@]:$KEEP_LAST}"; do
                rm -rf "$ckpt"
            done
        fi
    done
) &
CLEANUP_PID=$!
trap "kill $CLEANUP_PID 2>/dev/null" EXIT

lerobot-train \
    --policy.type=act \
    --policy.device=cuda \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --dataset.repo_id=puheliang/lerobot_fmc3_grab_box_v2 \
    --dataset.video_backend=torchcodec \
    --output_dir=$OUTPUT_DIR \
    --job_name=grab_box_act \
    --batch_size=64 \
    --steps=100000 \
    --save_freq=5000 \
    --log_freq=100 \
    --num_workers=8 \
    --wandb.enable=false
