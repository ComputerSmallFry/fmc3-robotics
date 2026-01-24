lerobot-train  \
  --dataset.repo_id=${HF_USER}/pick_place_with_rgbd  \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_multi_gpu/lerobot_model \
  --job_name=pick_and_place_bottle_from_A_to_B \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=50000 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/lerobot_pick_and_place_diffusion
