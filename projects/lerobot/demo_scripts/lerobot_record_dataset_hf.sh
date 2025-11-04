#huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_arm \
  --display_data=true \
  --dataset.push_to_hub=True \
  --dataset.root=./hugging_face \
  --dataset.repo_id=${HF_USER}/lerobot_dataset \
  --dataset.num_episodes=3 \
  --dataset.single_task="Grab the pencil"
