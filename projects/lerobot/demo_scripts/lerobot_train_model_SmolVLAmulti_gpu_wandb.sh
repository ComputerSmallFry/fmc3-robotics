accelerate launch \
  --multi_gpu \
  --num_processes=1 \
  $(which lerobot-train) \
  --dataset.repo_id=${HF_USER}/my_eval_smolvla_single_cam  \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id=${HF_USER}/lerobot_pick_and_place_smolvla \
  --output_dir=outputs/train/act_multi_gpu/lerobot_model \
  --job_name=pick_and_place_bottle_from_A_to_B \
  --policy.device=cuda \
  --batch_size=8 \
  --steps=50000 \
  --optimizer.lr=2e-4 \
  --wandb.enable=true 
  
