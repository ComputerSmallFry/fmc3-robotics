# roboskill/<vendor>/so101/skill.py
from typing import Any, Dict, List, Optional, Literal
import time

from mcp.server.fastmcp import FastMCP

from so101_bridge.lerobot_adapter import So101LeRobotAdapter
from so101_bridge.safety import JointSafety

mcp = FastMCP(name="robots", stateless_http=True, host="0.0.0.0", port=8000)

ADAPTER: Optional[So101LeRobotAdapter] = None
SAFETY: Optional[JointSafety] = None


def result(ok: bool, data: Any = None, code: str = "", message: str = "", detail: Any = None) -> Dict[str, Any]:
    return {
        "ok": ok,
        "t": time.time(),
        "data": data if ok else {},
        "error": ({"code": code, "message": message, "detail": detail or {}} if not ok else None),
    }


def _require_connected() -> None:
    if not ADAPTER:
        raise RuntimeError("Call connect first")


@mcp.tool()
def health() -> Dict[str, Any]:
    return result(True, {"alive": True})


@mcp.tool()
def connect(
    robot_config_path: str,
    camera_name: str = "handeye",
    enable_camera: bool = True,
) -> Dict[str, Any]:
    global ADAPTER, SAFETY
    ADAPTER = So101LeRobotAdapter(robot_config_path, camera_name, enable_camera)
    ADAPTER.connect()
    SAFETY = JointSafety(dof=ADAPTER.dof, joint_limits=ADAPTER.joint_limits)
    return result(
        True,
        {
            "robot": "so101",
            "dof": ADAPTER.dof,
            "has_gripper": True,
            "camera_name": camera_name,
            "obs_keys": ["state.q", f"images.{camera_name}"],
        },
    )


@mcp.tool()
def disconnect() -> Dict[str, Any]:
    global ADAPTER
    if ADAPTER:
        ADAPTER.disconnect()
    ADAPTER = None
    return result(True, {"disconnected": True})


@mcp.tool()
def get_observation(
    image_format: Literal["jpeg_base64", "raw"] = "jpeg_base64",
    max_width: int = 640,
    max_height: int = 480,
    include: List[Literal["state", "image"]] = ["state", "image"],
) -> Dict[str, Any]:
    _require_connected()
    obs = ADAPTER.get_observation(image_format, max_width, max_height, include)
    return result(True, obs)


@mcp.tool()
def move_joints(
    q: List[float],
    duration_s: float = 1.0,
    rate_hz: int = 50,
    max_delta_per_step: float = 0.05,
) -> Dict[str, Any]:
    _require_connected()
    if SAFETY is None:
        raise RuntimeError("Safety not initialized")
    target = SAFETY.sanitize_target(q, max_delta_per_step, ADAPTER.get_joint_position())
    steps = ADAPTER.move_joints_interpolated(target, duration_s, rate_hz)
    return result(True, {"executed": True, "steps": steps})


@mcp.tool()
def gripper(command: Literal["open", "close"], timeout_s: float = 3.0) -> Dict[str, Any]:
    _require_connected()
    if command == "open":
        ADAPTER.open_gripper(timeout_s)
    else:
        ADAPTER.close_gripper(timeout_s)
    return result(True, {"executed": True})


@mcp.tool()
def stop() -> Dict[str, Any]:
    if not ADAPTER:
        return result(True, {"stopped": True})
    ADAPTER.stop()
    return result(True, {"stopped": True})


@mcp.tool()
def start_record(session: Dict[str, Any]) -> Dict[str, Any]:
    _require_connected()
    info = ADAPTER.start_record(session)
    return result(True, info)


@mcp.tool()
def stop_record() -> Dict[str, Any]:
    _require_connected()
    info = ADAPTER.stop_record()
    return result(True, info)


@mcp.tool()
def replay_episode(episode_path: str, speed_scale: float = 1.0) -> Dict[str, Any]:
    _require_connected()
    ADAPTER.replay_episode(episode_path, speed_scale)
    return result(True, {"replayed": True})


@mcp.tool()
def pick_and_place(
    pick_q: List[float],
    place_q: List[float],
    approach_duration_s: float = 1.0,
    lift_duration_s: float = 1.0,
    rate_hz: int = 50,
    max_delta_per_step: float = 0.05,
) -> Dict[str, Any]:
    _require_connected()
    if SAFETY is None:
        raise RuntimeError("Safety not initialized")

    pick_target = SAFETY.sanitize_target(pick_q, max_delta_per_step, ADAPTER.get_joint_position())
    ADAPTER.move_joints_interpolated(pick_target, approach_duration_s, rate_hz)
    ADAPTER.close_gripper(1.0)

    place_target = SAFETY.sanitize_target(place_q, max_delta_per_step, ADAPTER.get_joint_position())
    ADAPTER.move_joints_interpolated(place_target, lift_duration_s, rate_hz)
    ADAPTER.open_gripper(1.0)

    return result(True, {"executed": True})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
