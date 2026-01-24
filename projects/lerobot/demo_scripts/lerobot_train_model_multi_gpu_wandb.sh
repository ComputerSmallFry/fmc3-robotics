export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
  --mixed_precision=fp16 \
  --multi_gpu \
  --num_processes=2 \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/pick_place_with_rgbd  \
  --policy.type=act \
  --output_dir=outputs/train/act_multi_gpu/lerobot_model \
  --job_name=pick_and_place_bottle_from_A_to_B \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.precision=fp16 \
  --batch_size=16 \
  --steps=50000 \
  --optimizer.lr=2e-4 \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/lerobot_pick_and_place_act_rgbd
