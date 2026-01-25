#huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_arm \
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: 420222071192, fps: 30, width: 640, height: 480, color_mode: RGB, use_depth: true}}" \
  --display_data=false \
  --dataset.root=./hugging_face_smolvla \
  --dataset.num_episodes=30 \
  --dataset.reset_time_s=5 \
  --dataset.repo_id=${HF_USER}/eval_pick_place_smolvla_rgbd \
  --policy.path=${HF_USER}/lerobot_pick_and_place_smolvla_rgbd \
  --dataset.single_task="put box from A to B"
