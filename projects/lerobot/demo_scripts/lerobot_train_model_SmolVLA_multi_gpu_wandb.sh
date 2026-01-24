lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=${HF_USER}/lerobot_pick_and_place_smolvla_rgbd \
  --dataset.repo_id=${HF_USER}/pick_place_with_rgbd \
  --batch_size=32 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
