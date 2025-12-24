# roboskill/<vendor>/so101/skill.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
import time

from so101_bridge.lerobot_adapter import So101LeRobotAdapter
from so101_bridge.safety import JointSafety

app = FastAPI(title="SO101 RoboSkill Server", version="0.1.0")

ADAPTER: Optional[So101LeRobotAdapter] = None
SAFETY: Optional[JointSafety] = None

def result(ok: bool, data: Any = None, code: str = "", message: str = "", detail: Any = None):
    return {
        "ok": ok,
        "t": time.time(),
        "data": data if ok else {},
        "error": ({"code": code, "message": message, "detail": detail or {}} if not ok else None)
    }

class ConnectReq(BaseModel):
    robot_config_path: str
    camera_name: str = "handeye"
    enable_camera: bool = True

class ObsReq(BaseModel):
    image_format: Literal["jpeg_base64", "raw"] = "jpeg_base64"
    max_width: int = 640
    max_height: int = 480
    include: List[Literal["state", "image"]] = ["state", "image"]

class JointTarget(BaseModel):
    type: Literal["joint_position"] = "joint_position"
    q: List[float]

class MovePolicy(BaseModel):
    interp: Literal["linear"] = "linear"
    rate_hz: int = 50
    max_delta_per_step: float = 0.05

class MoveReq(BaseModel):
    target: JointTarget
    duration_s: float = 1.0
    policy: MovePolicy = Field(default_factory=MovePolicy)

class GripperReq(BaseModel):
    command: Literal["open", "close"]
    timeout_s: float = 3.0

class StartRecordReq(BaseModel):
    session: Dict[str, Any]

class ReplayReq(BaseModel):
    episode_path: str
    speed_scale: float = 1.0

@app.get("/health")
def health():
    return result(True, {"alive": True})

@app.post("/robot_connect")
def robot_connect(req: ConnectReq):
    global ADAPTER, SAFETY
    try:
        ADAPTER = So101LeRobotAdapter(req.robot_config_path, req.camera_name, req.enable_camera)
        ADAPTER.connect()
        SAFETY = JointSafety(dof=ADAPTER.dof, joint_limits=ADAPTER.joint_limits)
        return result(True, {
            "robot": "so101",
            "dof": ADAPTER.dof,
            "has_gripper": True,
            "camera_name": req.camera_name,
            "obs_keys": ["state.q", f"images.{req.camera_name}"]
        })
    except Exception as e:
        return result(False, code="CONNECT_FAILED", message=str(e))

@app.post("/robot_disconnect")
def robot_disconnect():
    global ADAPTER
    try:
        if ADAPTER:
            ADAPTER.disconnect()
        ADAPTER = None
        return result(True, {"disconnected": True})
    except Exception as e:
        return result(False, code="DISCONNECT_FAILED", message=str(e))

@app.post("/get_observation")
def get_observation(req: ObsReq):
    try:
        if not ADAPTER:
            return result(False, code="NOT_CONNECTED", message="Call robot_connect first")
        obs = ADAPTER.get_observation(req.image_format, req.max_width, req.max_height, req.include)
        return result(True, obs)
    except Exception as e:
        return result(False, code="OBS_FAILED", message=str(e))

@app.post("/move_joints")
def move_joints(req: MoveReq):
    try:
        if not ADAPTER or not SAFETY:
            return result(False, code="NOT_CONNECTED", message="Call robot_connect first")
        q = SAFETY.sanitize_target(req.target.q, req.policy.max_delta_per_step, ADAPTER.get_joint_position())
        steps = ADAPTER.move_joints_interpolated(q, req.duration_s, req.policy.rate_hz)
        return result(True, {"executed": True, "steps": steps})
    except Exception as e:
        return result(False, code="MOVE_FAILED", message=str(e))

@app.post("/gripper")
def gripper(req: GripperReq):
    try:
        if not ADAPTER:
            return result(False, code="NOT_CONNECTED", message="Call robot_connect first")
        if req.command == "open":
            ADAPTER.open_gripper(req.timeout_s)
        else:
            ADAPTER.close_gripper(req.timeout_s)
        return result(True, {"executed": True})
    except Exception as e:
        return result(False, code="GRIPPER_FAILED", message=str(e))

@app.post("/stop")
def stop():
    try:
        if not ADAPTER:
            return result(True, {"stopped": True})
        ADAPTER.stop()
        return result(True, {"stopped": True})
    except Exception as e:
        return result(False, code="STOP_FAILED", message=str(e))

@app.post("/start_record")
def start_record(req: StartRecordReq):
    try:
        if not ADAPTER:
            return result(False, code="NOT_CONNECTED", message="Call robot_connect first")
        info = ADAPTER.start_record(req.session)
        return result(True, info)
    except Exception as e:
        return result(False, code="RECORD_START_FAILED", message=str(e))

@app.post("/stop_record")
def stop_record():
    try:
        if not ADAPTER:
            return result(False, code="NOT_CONNECTED", message="Call robot_connect first")
        info = ADAPTER.stop_record()
        return result(True, info)
    except Exception as e:
        return result(False, code="RECORD_STOP_FAILED", message=str(e))

@app.post("/replay_episode")
def replay_episode(req: ReplayReq):
    try:
        if not ADAPTER:
            return result(False, code="NOT_CONNECTED", message="Call robot_connect first")
        ADAPTER.replay_episode(req.episode_path, req.speed_scale)
        return result(True, {"replayed": True})
    except Exception as e:
        return result(False, code="REPLAY_FAILED", message=str(e))
