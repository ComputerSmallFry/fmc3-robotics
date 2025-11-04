lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_arm \
    #--dataset.root=./record_dataset \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.episode=0 \
    
