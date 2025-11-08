#huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}}" \
  --display_data=false \
  --dataset.root=./hugging_face \
  --dataset.repo_id=${HF_USER}/eval_lerobot_dataset \
  --policy.path=${HF_USER}/lerobot \
  --dataset.single_task="Grab the pencil"
