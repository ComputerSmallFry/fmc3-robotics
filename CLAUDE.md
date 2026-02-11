# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository deploys teleoperation services for GRx series robots (GR2 and GR3). The system enables remote operation and data collection through Docker-based containers, LiveKit real-time streaming, multiple camera backends (OAK/DepthAI, Orbbec, OpenCV), and background daemon services for data processing/upload.

## Repository Structure

```
daq-deploy/
├── gr3/              # GR3 robot teleoperation scripts and GUI
├── gr2/              # GR2 robot teleoperation scripts
├── livekit/          # LiveKit server for streaming (local mode)
├── daemon/           # Background data processing services
│   ├── gr2/          # GR2 daemon docker-compose
│   ├── gr3/          # GR3 daemon docker-compose
│   ├── deploy/       # Per-host start/stop/log scripts
│   └── fabfile.py    # Fabric deployment automation
├── setup/            # System provisioning scripts
├── udev/             # USB device rules (cameras, exoskeletons, etc.)
└── s3/               # S3 mounting utilities
```

## Common Commands

### Initial Setup (one-time per machine)

```bash
bash setup/provisioning.sh          # Provision fresh Ubuntu 22.04
sudo bash udev/setup_udev.sh        # Setup udev rules for USB devices
sudo bash setup/setup_user_group.sh  # Setup user permissions (input, docker groups)
bash gr3/install_gui.sh              # (Optional) Install GR3 GUI desktop launcher
```

### Docker Registry Access

```bash
sudo docker login docker.fftaicorp.com
# username: devops
# password: fftai@2025
```

### GR3 Teleoperation

```bash
bash gr3/run_gr3.sh --graph daq-qnexo-depthai.yml --notes grxtest        # Company network (default)
bash gr3/run_gr3.sh --local-mode --graph daq-quest-depthai.yml            # Local network mode (start LiveKit first)
bash gr3/run_gr3.sh --debug --graph daq-qnexo-depthai.yml                 # Debug: show command without executing
bash gr3/run_gr3.sh --help                                                 # All options and graph file descriptions

# GUI interface
bash gr3/launch_gui.sh
# Or: uv run python gr3/gr3_gui.py
```

### GR2 Teleoperation

```bash
bash gr2/run_t5d_depthai.sh gr2t2d oak_97 t5dtest use_depth=true robot.visualize=true    # OAK camera
bash gr2/run_t5d_orbbec.sh gr2t2d orbbec t5dtestorbbec use_depth=false robot.visualize=true  # Orbbec camera
```

### LiveKit Server (Local Mode)

```bash
bash livekit/start_local_server.sh   # Start local LiveKit server
bash livekit/stop_local_server.sh    # Stop local LiveKit server
bash livekit/connect_quest.sh        # Connect Quest headset via USB (ADB port forward)
# Viewer: http://localhost:8081/  |  VR Client: http://localhost:8080/
```

### Daemon Services

```bash
# Per-host (from daemon/gr3/ or daemon/gr2/)
STATION_ID=$(hostname | sed 's/[^0-9]//g') docker compose up -d
docker logs daemon -f
docker compose down

# Multi-host via Fabric (daemon/ directory)
fab rput --hosts=11,12,13 --local=/path/to/file --remote=/remote/path   # Deploy files
fab run --hosts=11,12,13 --cmd="docker ps"                              # Run commands
fab du --hosts=11,12,13                                                  # Check disk usage
fab rm_data --hosts=11,12,13                                             # Clean recording data
```

### S3 Mounting

```bash
bash s3/mount_s3.sh                              # Mount default bucket (farther-data) to ~/s3/
bash s3/mount_s3.sh --bucket my-bucket -m /mnt/custom  # Custom bucket and mount path
```

## Architecture Notes

### Container Execution Pattern

All teleoperation sessions run as ephemeral Docker containers:

- **GR3**: Image `docker.fftaicorp.com/farts/farther` with graph-based config
- **GR2**: Image `docker.fftaicorp.com/farts/depthai-deploy-dds`
- Containers are auto-removed after exit (`--rm` flag)

**GR3 volume mounts** (`run_gr3.sh`):
- `-v /dev:/dev` — full device access (cameras, exoskeletons)
- `-v ~/data:~/data` — recording data storage
- `-v /tmp:/tmp` — temporary data exchange (camera frames, etc.)

**GR2 volume mounts** (`run_t5d_*.sh`):
- `-v /dev:/dev` — full device access
- `-v /mnt/Data:/app/data` — recording data storage
- `-v /tmp/.X11-unix:/tmp/.X11-unix` — X11 forwarding for visualization

All containers use `--privileged` and `--network host` (low-latency DDS communication).

### Graph Files

Graph files (`.yml`) define the teleoperation pipeline. Key properties are auto-detected from filename:

- **Equipment type**: `qnexo`, `quest`, `pedal`, `visionpro` (sets `DAQ_EQUIPMENT_TYPE`)
- **Camera backend**: `depthai` (OAK), `orbbec`, or `opencv`
- **Robot model**: `GR2` or `GR3` (default GR3 unless filename contains `GR2`)
- **Hand DOF**: 6 (default) or 12 (if filename contains `12H`)
- **Mode**: `remote-on-client` / `remote-on-robot-nuc` / `remote-on-robot-bag` for wireless

Run `bash gr3/run_gr3.sh --help` for the full list of graph files with descriptions.

### Runtime Configuration (`gr3/config.env`)

Operator-facing configuration file loaded via `--env-file`. Key parameter groups:
- **Camera**: `DAQ_CAMERA_ROTATE`, `DAQ_DRAW_TIMESTAMP`, `DAQ_CAMERA_CAPTURE_PATH` (OpenCV device)
- **Retargeting**: `RETARGETING_HAND_SCALING`, `RETARGETING_BODY_SCALING` (height-dependent, 0.8~1.3)
- **Exoskeleton (Qnexo)**: joystick range/deadzone, trigger range, press timing
- **Gripper**: open/closed positions per finger joint (`EXO_GRIPPER_*_POS_*`)
- **Motion**: head/waist joint speeds, base velocity coefficients
- **Safety**: `EXO_MAX_RAW_DATA_DELAY` (max delay before stopping)

### Environment Variables

Key environment variables set during container launch:

- `DAQ_NOTES`: Session identifier/notes
- `DAQ_PILOT` / `DAQ_OPERATOR`: Pilot and operator IDs (`-1` for anonymous)
- `DAQ_MACHINE_ID`: Robot ID (e.g., `GR2`, `GR3`, `gr301aa0017`)
- `DAQ_STATION_ID`: Station/computer ID (extracted from hostname numeric suffix)
- `DAQ_EQUIPMENT_TYPE`: Auto-detected from graph filename
- `DAQ_RECORD_DIR`: Auto-set to `~/data/YYYY-MM/YYYY-MM-DD/STATION_ID/EQUIPMENT_TYPE/`
- `DOMAIN_ID`: DDS domain ID (default: 123)

For local mode, additional LiveKit variables: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`.

### LiveKit Services (Local Mode)

`livekit/docker-compose.yaml` runs three services (all host network):
- **livekit-server** (v1.9.0): WebRTC signaling on port 7880, TCP fallback 7881, media ports 50000-60000
- **teleop-client**: Receives VR input and camera data
- **teleop-viewer**: Web-based video viewer (port 8081)

### Daemon Services

`daemon/gr3/docker-compose.yml` runs:
- **redis**: Cache (port 6379)
- **daemon**: Monitors recording directory, processes data, uploads to S3/PocketBase. Runs at low CPU/IO priority (`ionice -c 3 nice -n 12`)
- **delete-worker**: RQ worker for data cleanup tasks

### Data Flow

1. **Teleoperation Input**: Exoskeleton/VR headset -> Container -> Robot NUC
2. **Video Stream**: Camera -> Container -> LiveKit -> Web clients
3. **Data Recording**: Container -> `~/data/YYYY-MM/YYYY-MM-DD/STATION_ID/EQUIPMENT_TYPE/`
4. **Background Processing**: Daemon monitors recording directory -> Processes data -> Uploads to S3/PocketBase

### Multi-Host Deployment

Fabric (`daemon/fabfile.py`) manages stations 11-18, 21-24 (format: `{id:03d}.farts.com`). Uses `ThreadingGroup` for parallel SSH execution.

### Robot NUC Connection

Before running teleoperation, start AuroraCore on the robot's NUC:

```bash
ssh gr301aa0017@192.168.137.220   # password: fftai2015
deploy_aurora tag Algorithm-aurora_unified-305-GR3
AuroraCore
```

## Development Patterns

### Python/Tooling

- Python >=3.13, managed with `uv` (no pip/requirements.txt)
- `pyproject.toml` defines entry point: `daq-gui = "gr3.gr3_gui:main"`
- No formal build, lint, or test framework
- GUI validation: `python gr3/test_gui.py`

### GUI Application (`gr3/gr3_gui.py`)

- Built with tkinter, bilingual (Chinese/English)
- Wraps `run_gr3.sh` with graphical controls
- Manages container lifecycle (start/stop/monitor) with real-time log display
- ANSI color parsing for log output

### udev Rules

USB device permissions are critical for camera and exoskeleton access:
- `80-depthai.rules`: OAK cameras (vendor `03e7`)
- `99-obsensor-libusb.rules`: Orbbec cameras (vendor `2bc5`)
- `51-android.rules`: Quest headset (ADB)
- `qnbot.rules`: Qnbot devices

After modifying rules: `sudo bash udev/setup_udev.sh`

## Important Notes

- **Hostname convention**: Station ID extracted from hostname numeric suffix (e.g., `gr3-station-17` -> `17`)
- **MACHINE_ID**: Should be set in `~/.bashrc` as `MACHINE_ID=$(echo $HOSTNAME | sed 's/[^0-9]//g')`
- **Data path**: `/mnt/Data` for GR2, `~/data` for GR3
- **Container naming**: `farther-daqdeploy` (auto-stops existing container on restart)

## Troubleshooting

- **Permission denied for USB devices**: Run `sudo bash udev/setup_udev.sh` and reboot
- **Docker pull errors**: Check login to `docker.fftaicorp.com` registry
- **Container conflicts**: Existing containers are auto-stopped in `run_gr3.sh`
- **Local LiveKit not working**: Ensure `livekit/start_local_server.sh` ran successfully, check `docker ps`
- **OpenCV camera not found**: Set `DAQ_CAMERA_CAPTURE_PATH=/dev/video0` in env file
