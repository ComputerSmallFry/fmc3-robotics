RoboBrain + RoboOS + RoboSkill + LeRobot (SO101) integration

Overview
- This repo provides an SO101 adapter for LeRobot plus an MCP skill server for RoboOS.
- The skill server exposes tools over MCP at `http://<robot_ip>:8000/mcp`.

Prerequisites
- LeRobot installed at `/home/haoanw/workspace/lerobot` (or on PYTHONPATH).
- SO101 motors calibrated and reachable on the configured port.
- Handeye camera available (configured in `configs/lerobot_robot.yaml`).

Quick start
1) Configure LeRobot robot
   - `configs/lerobot_robot.yaml` (port, camera, normalized control)

2) Start the skill server
   - `src/roboskill/fmc3-robotics/so101/scripts/run_skill_server.sh`

3) Smoke test MCP tools
   - `python src/roboskill/fmc3-robotics/so101/scripts/smoke_test.py --robot-config configs/lerobot_robot.yaml --gripper`

4) Configure RoboOS slaver
   - Update `configs/roboos_slaver_config.yaml` with the skill server URL.
   - Ensure RoboOS slaver points to that config.

Notes
- The server uses MCP (FastMCP) so RoboOS can list and call tools remotely.
