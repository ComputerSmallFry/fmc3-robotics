accelerate launch \
  --multi_gpu \
  --num_processes=2 \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/lerobot_dataset  \
  --policy.type=act \
  --output_dir=outputs/train/act_multi_gpu/lerobot_model \
  --job_name=pickup_pen \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=50000 \
  --optimizer.lr=2e-4 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/lerobot
