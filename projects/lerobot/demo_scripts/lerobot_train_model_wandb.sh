lerobot-train \
  --mixed_precision=fp16 \
  --dataset.repo_id=${HF_USER}/pick_place_with_rgbd  \
  --policy.type=act \
  --output_dir=outputs/train/pick_place_with_rgbd \
  --job_name=pick_and_place_bottle_from_A_to_B \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/lerobot_pick_and_place_act_rgbd
