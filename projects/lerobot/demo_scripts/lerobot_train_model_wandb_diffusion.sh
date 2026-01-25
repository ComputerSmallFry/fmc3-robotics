export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train  \
  --dataset.repo_id=${HF_USER}/pick_place_with_rgbd  \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion/my_diffusion_lerobot_model \
  --job_name=pick_and_place_diffusion \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=50000 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/lerobot_pick_and_place_diffusion
