#!/bin/bash
# ACT 策略训练 — 抓瓶子放盒子再拿出来
# 数据集: pick_bottle_and_place_into_box_lerobot (GR-2, 211 episodes, 62874 frames)
# 硬件: RTX 4090

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUTPUT_DIR=~/output_lerobot_train/pick_bottle/act
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
    --dataset.repo_id=pick_bottle_and_place_into_box_lerobot \
    --dataset.root=/home/phl/workspace/dataset/fourier/pick_bottle_and_place_into_box_lerobot \
    --dataset.video_backend=torchcodec \
    --output_dir=$OUTPUT_DIR \
    --job_name=pick_bottle_act \
    --batch_size=64 \
    --steps=100000 \
    --save_freq=5000 \
    --log_freq=100 \
    --num_workers=8 \
    --wandb.enable=false
