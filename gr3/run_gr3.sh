#!/bin/bash

# Local mount path for data storage
LOCAL_MOUNT_PATH="$HOME/data"

# Default values
DEFAULT_SELECTED_DOCKER="docker.fftaicorp.com/farts/farther:onekey-260123060517-462cac7b"
DEFAULT_SELECTED_GRAPH="agv.yml"
DEFAULT_DAQ_NOTES="grxtest"
DEFAULT_DAQ_PILOT="-1"
DEFAULT_DAQ_OPERATOR="-1"
DEFAULT_STATION_ID=$(echo "$(hostname)" | sed 's/[^0-9]//g')
DEFAULT_MACHINE_ID="GRx"
DEFAULT_DOMAIN_ID="123"
DEFAULT_ENV_FILE=""

# Initialize variables with default values
SELECTED_DOCKER="${DEFAULT_SELECTED_DOCKER}"
SELECTED_GRAPH="${DEFAULT_SELECTED_GRAPH}"
DAQ_NOTES="${DEFAULT_DAQ_NOTES}"
DAQ_PILOT="${DEFAULT_DAQ_PILOT}"
DAQ_OPERATOR="${DEFAULT_DAQ_OPERATOR}"
STATION_ID="${DEFAULT_STATION_ID}"
MACHINE_ID="${DEFAULT_MACHINE_ID}"
DOMAIN_ID="${DEFAULT_DOMAIN_ID}"
ENV_FILE="${DEFAULT_ENV_FILE}"
DEBUG_MODE=false
LOCAL_MODE=false

# Display usage information
usage() {
    echo "Usage: $0 [--docker DOCKER_IMAGE] [--tag TAG] [--graph GRAPH_PATH] [--notes DAQ_NOTES] [--pilot DAQ_PILOT] [--operator DAQ_OPERATOR] [--station-id STATION_ID] [--machine-id MACHINE_ID] [--domain-id DOMAIN_ID] [--env-file ENV_FILE_PATH] [--debug] [--local-mode]"
    echo "  --docker DOCKER_IMAGE    Set Docker image (default: $DEFAULT_SELECTED_DOCKER)"
    echo "  --tag TAG                Set Docker image tag, higher priority than --docker, effective as docker.fftaicorp.com/farts/farther:TAG"
    echo "  --graph GRAPH_PATH       Set graph path (default: $DEFAULT_SELECTED_GRAPH)"
    echo "  --notes DAQ_NOTES        Set DAQ_NOTES environment variable (default: $DEFAULT_DAQ_NOTES)"
    echo "  --pilot DAQ_PILOT        Set DAQ_PILOT environment variable (default: $DEFAULT_DAQ_PILOT)"
    echo "  --operator DAQ_OPERATOR  Set DAQ_OPERATOR environment variable (default: $DEFAULT_DAQ_OPERATOR)"
    echo "  --station-id STATION_ID  Set station ID (default: numeric part from hostname - $DEFAULT_STATION_ID)"
    echo "  --machine-id MACHINE_ID  Set machine ID (default: $DEFAULT_MACHINE_ID)"
    echo "  --domain-id DOMAIN_ID    Set domain ID (default: $DEFAULT_DOMAIN_ID)"
    echo "  --env-file ENV_FILE_PATH Set environment variable file path (default: empty)"
    echo "  --debug                  Debug mode: print Docker command without executing"
    echo "  --local-mode             Local mode: use local livekit parameters"
    echo "  -h, --help               Display this help information"
    echo ""
    echo "Notes:"
    echo "  - Local data storage path LOCAL_MOUNT_PATH is set to '$HOME/data' by default, please ensure this path exists"
    echo "  - DAQ_EQUIPMENT_TYPE is automatically set based on SELECTED_GRAPH: 'agv' if contains 'agv', 't5d' if contains 't5d', otherwise 'none'"
    echo "  - When SELECTED_GRAPH contains 'robot-nuc', the --with-aurora parameter will be added automatically"
    echo ""
    echo "Examples:"
    echo "  $0 --graph agv.yml --notes grxtest --pilot 123 --operator 456 --machine-id gr3 --domain-id 143"
    echo "  $0 --debug  # Debug mode, only print command without executing"
    echo ""
    echo "Functions:"
    echo "  $0 --graph agv.yml  # AGV 遥操 AGV teleoperation"
    echo "  $0 --graph agv_gr2.yml  # AGV 遥操 GR2 AGV teleoperation for GR2"
    echo "  $0 --graph agv_opencv.yml  # AGV 遥操, OpenCV相机 AGV teleoperation with OpenCV camera"
    echo "  $0 --graph agv_gr2_opencv.yml  # AGV 遥操 GR2, OpenCV相机 AGV teleoperation for GR2 with OpenCV camera"
    echo "  $0 --graph daq_t5d.yml  # T5D 数采 T5D data collection"
    echo "  $0 --graph daq_t5d_opencv.yml  # T5D 数采, OpenCV相机 T5D data collection with OpenCV camera"
    echo "  $0 --graph exo-debug.yml  # 测试外骨骼输出, 不控机器人 Test exoskeleton output, without control"
}

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --docker)
            SELECTED_DOCKER="${2}"
            shift 2
            ;;
        --tag)
            SELECTED_DOCKER="docker.fftaicorp.com/farts/farther:${2}"
            shift 2
            ;;
        --graph)
            SELECTED_GRAPH="${2}"
            shift 2
            ;;
        --notes)
            DAQ_NOTES="${2}"
            shift 2
            ;;
        --pilot)
            DAQ_PILOT="${2}"
            shift 2
            ;;
        --operator)
            DAQ_OPERATOR="${2}"
            shift 2
            ;;
        --station-id)
            STATION_ID="${2}"
            shift 2
            ;;
        --machine-id)
            MACHINE_ID="${2}"
            shift 2
            ;;
        --domain-id)
            DOMAIN_ID="${2}"
            shift 2
            ;;
        --env-file)
            ENV_FILE="${2}"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --local-mode)
            LOCAL_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# Automatically set DAQ_EQUIPMENT_TYPE based on SELECTED_GRAPH
if [[ "$SELECTED_GRAPH" == *"agv"* ]]; then
    DAQ_EQUIPMENT_TYPE="agv"
elif [[ "$SELECTED_GRAPH" == *"t5d"* ]]; then
    DAQ_EQUIPMENT_TYPE="t5d"
else
    DAQ_EQUIPMENT_TYPE="none"
fi

# Set hand dof based on SELECTED_GRAPH
if [[ "$SELECTED_GRAPH" == *"12H"* ]]; then
    HAND_DOF="12"
else
    HAND_DOF="6"
fi

# Check STATION_ID, set to "-1" if empty
if [[ -z "$STATION_ID" ]]; then
    STATION_ID="-1"
fi

# Check MACHINE_ID, set to GR2 if DEFAULT_MACHINE_ID and SELECTED_GRAPH contains gr2/GR2, otherwise GR3
if [[ "$MACHINE_ID" == "$DEFAULT_MACHINE_ID" ]]; then
    if [[ "$SELECTED_GRAPH" == *"gr2"* ]] || [[ "$SELECTED_GRAPH" == *"GR2"* ]]; then
        MACHINE_ID="GR2"
    else
        MACHINE_ID="GR3"
    fi
fi


# Display current configuration
echo "=== Docker Container Startup Configuration ==="
echo "SELECTED_DOCKER:      ${SELECTED_DOCKER}"
echo "SELECTED_GRAPH:       ${SELECTED_GRAPH}"
echo "DAQ_EQUIPMENT_TYPE:   ${DAQ_EQUIPMENT_TYPE}"
echo "LOCAL_MOUNT_PATH:     ${LOCAL_MOUNT_PATH}"
echo "DAQ_NOTES:            ${DAQ_NOTES}"
echo "DAQ_PILOT:            ${DAQ_PILOT}"
echo "DAQ_OPERATOR:         ${DAQ_OPERATOR}"
echo "STATION_ID:           ${STATION_ID}"
echo "MACHINE_ID:           ${MACHINE_ID}"
echo "DOMAIN_ID:            ${DOMAIN_ID}"
echo "ENV_FILE:             ${ENV_FILE}"
echo "DEBUG_MODE:           ${DEBUG_MODE}"
echo "LOCAL_MODE:           ${LOCAL_MODE}"
echo "========================"

# Set container name based on MACHINE_ID and STATION_ID
CONTAINER_NAME="farther-daqdeploy"

# Build Docker command template
docker_command="docker run --rm --privileged --name ${CONTAINER_NAME} \
  -v /dev:/dev \
  -v "${LOCAL_MOUNT_PATH}":$HOME/data \
  -v $HOME/dataset/fourier/dora-record:/tmp \
  --network host"

# If environment file is specified, add --env-file parameter
if [[ -n "${ENV_FILE}" ]]; then
    docker_command="${docker_command} \
  --env-file "${ENV_FILE}""
fi

# 添加通用环境变量和参数
docker_command="${docker_command} \
  -e DAQ_NOTES="${DAQ_NOTES}" \
  -e DAQ_PILOT="${DAQ_PILOT}" \
  -e DAQ_OPERATOR="${DAQ_OPERATOR}" \
  -e DAQ_MACHINE_ID="${MACHINE_ID}" \
  -e DAQ_STATION_ID="${STATION_ID}" \
  -e DAQ_EQUIPMENT_TYPE="${DAQ_EQUIPMENT_TYPE}" \
  -e DAQ_RECORD_DIR=$HOME/data/$(date +"%Y-%m")/$(date +"%Y-%m-%d")/${STATION_ID}/${DAQ_EQUIPMENT_TYPE} \
  ${SELECTED_DOCKER} \
  -g graphs/"${SELECTED_GRAPH}" \
  --left-hand-type FDH-${HAND_DOF}L \
  --right-hand-type FDH-${HAND_DOF}R \
  --room-name farther-${STATION_ID} \
  --domain-id ${DOMAIN_ID}"

# If SELECTED_GRAPH contains robot-nuc, add --with-aurora parameter
if [[ "${SELECTED_GRAPH}" == *"robot-nuc"* ]]; then
    docker_command="${docker_command} --with-aurora"
fi

# If local mode is enabled, add local livekit parameters
if [ "${LOCAL_MODE}" = true ]; then
    docker_command="${docker_command} \
    -e LIVEKIT_URL=ws://localhost:7880 \
    -e LIVEKIT_API_KEY=a1b2c3d4e5f678901234567890 \
    -e LIVEKIT_API_SECRET=a1b2c3d4e5f6789012345678901234567890abcdef123456789012345678901234"
fi

# Determine whether to execute or print the command based on debug mode
if [ "${DEBUG_MODE}" = true ]; then
    echo "=== Debug Mode: Docker Command (Not Executing) ==="
    echo "${docker_command}"
    echo "====================================="
else
    echo "Checking for containers with the same name..."
    # Stop and remove container with the same name if it exists
    if docker ps -a | grep -q "${CONTAINER_NAME}"; then
        echo "Found container with the same name ${CONTAINER_NAME}, stopping and removing..."
        docker rm -f "${CONTAINER_NAME}" || echo "Failed to remove container, but will continue to start new container"
    fi
    
    echo "Starting Docker container ${CONTAINER_NAME}..."
    # Use eval to properly parse parameters with quotes
    eval "${docker_command}"
fi
