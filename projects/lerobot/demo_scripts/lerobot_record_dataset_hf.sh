#huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: 420222071192, fps: 30, width: 640, height: 480, color_mode: RGB, use_depth: true}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_arm \
  --display_data=true \
  --dataset.push_to_hub=True \
  --dataset.root=./hugging_face \
  --dataset.repo_id=${HF_USER}/pick_place_with_rgbd \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=30 \
  --dataset.single_task="pick bottle from A and place to B"
