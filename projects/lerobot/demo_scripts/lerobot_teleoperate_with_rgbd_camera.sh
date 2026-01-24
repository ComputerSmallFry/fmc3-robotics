lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_arm \
    --robot.cameras="{front: {type: intelrealsense, serial_number_or_name: 420222071192, fps: 30, width: 640, height: 480, color_mode: RGB, use_depth: true}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_arm  \
    --display_data=true
