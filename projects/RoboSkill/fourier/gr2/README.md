# Fourier GR2 Robot Skill Server

This skill server provides MCP-based control for the Fourier GR2 humanoid robot.

## Prerequisites

- Python 3.10+
- Fourier GR2 robot connected to the network
- DDS domain ID: 123 (configurable in skill.py)

## Installation

```bash
# Create conda environment
conda create -n fourier-robot python=3.10 -y
conda activate fourier-robot

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `skill.py` to configure:
- `DOMAIN_ID`: DDS domain ID (default: 123)
- `ROBOT_NAME`: Robot name (default: "gr2")

## Running the Service

```bash
python skill.py
```

This starts the MCP skill server at `http://0.0.0.0:8000`.

## Available Tools

| Tool | Description |
|------|-------------|
| `connect_robot()` | Connect to robot and enter control mode |
| `disconnect_robot()` | Disconnect and return to stand mode |
| `wave_hand(wave_count, wave_speed)` | Make robot wave its hand |
| `thumbs_up()` | Make robot give thumbs up gesture |
| `move_arm_to_position(positions)` | Move arm to 7-DOF joint positions |
| `set_hand_gesture(positions)` | Set 6-DOF hand finger positions |

## RoboOS Integration

To use with RoboOS, set in `RoboOS/slaver/config.yaml`:

```yaml
robot:
  name: fourier_gr2
  call_type: remote
  path: "http://<ROBOT_IP>:8000"
```

## Example Usage

```python
# Via MCP client
await connect_robot()
await wave_hand(wave_count=3, wave_speed=0.3)
await thumbs_up()
await disconnect_robot()
```
