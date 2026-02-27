#!/bin/bash
# ACT 策略推理 — 抓取胶带放入盒子任务 (SO-101)
# 用法: bash scripts/inference/infer_act_grab_box.sh

CHECKPOINT=~/output_lerobot_train/grab_box/act/checkpoints/last/pretrained_model

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM2 \
    --robot.id=zihao_follower_arm \
    --robot.cameras='{"top":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":2,"width":640,"height":480,"fps":30}}' \
    --policy.path=$CHECKPOINT \
    --display_data=true \
    --dataset.repo_id=puheliang/eval_lerobot_fmc3_grab_box_v2 \
    --dataset.single_task="Pick up the tape and place it in the box" \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=0 \
    --dataset.num_episodes=20 \
    --dataset.push_to_hub=false
