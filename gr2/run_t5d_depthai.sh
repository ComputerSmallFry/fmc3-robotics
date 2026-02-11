xhost + local:$USER >/dev/null 2>&1
machine_id=$(echo "$(hostname)" | sed 's/[^0-9]//g')
robot_type=$1
camera=$2
task=$3


echo "Running Data Collection..."
echo "Machine ID: ${machine_id}"
echo "Robot Type: ${robot_type}"
echo "Camera: ${camera}"
echo "Task: ${task}"
echo "Namespace: gr/daq-${machine_id}"
echo "-----------------------------------"
shift 3

echo "Additional arguments: $@"

docker run --rm -it --name daq \
    --user $(id -u):$(id -g) \
    --privileged \
    -v /dev:/dev \
    --device-cgroup-rule='c 189:* rmw' \
    -e DISPLAY=$DISPLAY \
    -e "HOSTNAME=$(cat /etc/hostname)" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/Data:/app/data:rw \
    -v ~/.certs:/app/certs:ro \
    -v ~/.farts/outputs:/app/outputs:rw \
    -v ~/teleoperation/configs:/app/configs:rw \
    -v ~/teleoperation/server_config:/app/server_config:rw \
    --network=host \
    --ipc=host \
    -e HYDRA_FULL_ERROR=1 \
    -e PB_ENDPOINT="https://pocketbase.fftaicorp.com/" \
    docker.fftaicorp.com/farts/depthai-deploy-dds:depthai \
    python -m teleoperation \
    --config-name daq \
    robot=${robot_type} \
    task_name=factory_$(date +%m_%d)_"${task}" \
    camera="${camera}" \
    hand=fourier_dexpilot_dhx \
    $@
