export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-train \
  --dataset.repo_id=${HF_USER}/pick_place_with_rgbd  \
  --policy.type=act \
  --output_dir=outputs/train/pick_place_with_rgbd \
  --job_name=pick_and_place_act \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/lerobot_pick_and_place_act_rgbd
