lerobot-train \
  --dataset.repo_id=${HF_USER}/lerobot_dataset  \
  --policy.type=act \
  --output_dir=outputs/train/lerobot_model \
  --job_name=pickup_pen \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/lerobot
