# src/so101_bridge/lerobot_adapter.py
from typing import Any, Dict, List, Optional, Literal
import time
import base64
from pathlib import Path
import numpy as np
import cv2
import yaml

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from so101_bridge.dataset_adapter import So101DatasetAdapter

class So101LeRobotAdapter:
    def __init__(self, robot_config_path: str, camera_name: str, enable_camera: bool):
        self.robot_config_path = robot_config_path
        self.camera_name = camera_name
        self.enable_camera = enable_camera

        self.robot = None
        self.dof = 6
        self.joint_limits = None  # fill with (min,max) per joint if available
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
        self.gripper_open = 100.0
        self.gripper_closed = 0.0

        self._dataset = So101DatasetAdapter()
        # recording state (optional)
        self._recording = False
        self._record_session = None

    def connect(self) -> None:
        cfg = self._load_yaml(self.robot_config_path)
        robot_cfg = cfg.get("robot", cfg)
        if not isinstance(robot_cfg, dict):
            raise ValueError("robot config must be a dict")

        cameras_cfg = {}
        for name, cam_cfg in (robot_cfg.get("cameras") or {}).items():
            cam_type = cam_cfg.get("type", "opencv")
            if cam_type != "opencv":
                raise ValueError(f"unsupported camera type: {cam_type}")
            index_or_path = cam_cfg.get("index_or_path", 0)
            if isinstance(index_or_path, str):
                index_or_path = int(index_or_path) if index_or_path.isdigit() else Path(index_or_path)
            cameras_cfg[name] = OpenCVCameraConfig(
                index_or_path=index_or_path,
                fps=cam_cfg.get("fps"),
                width=cam_cfg.get("width"),
                height=cam_cfg.get("height"),
            )

        robot_config = SO101FollowerConfig(
            port=robot_cfg["port"],
            id=robot_cfg.get("id"),
            use_degrees=bool(robot_cfg.get("use_degrees", False)),
            max_relative_target=robot_cfg.get("max_relative_target"),
            cameras=cameras_cfg,
        )
        self.robot = SO101Follower(robot_config)
        self.robot.connect()

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def get_joint_position(self) -> List[float]:
        obs = self._raw_get_obs()
        return self._extract_q(obs)

    def get_observation(self,
                        image_format: Literal["jpeg_base64", "raw"],
                        max_w: int,
                        max_h: int,
                        include: List[str]) -> Dict[str, Any]:
        obs = self._raw_get_obs()
        out: Dict[str, Any] = {}

        if "state" in include:
            q = self._extract_q(obs)
            out["state"] = {"q": q}

        if "image" in include and self.enable_camera:
            img = self._extract_image(obs)  # np.uint8 HxWx3
            img = self._resize_keep_aspect(img, max_w, max_h)
            if image_format == "raw":
                out["image"] = {
                    "camera": self.camera_name,
                    "format": "raw",
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "data": img.tolist()
                }
            else:
                jpg = self._encode_jpeg(img)
                out["image"] = {
                    "camera": self.camera_name,
                    "format": "jpeg_base64",
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "data": base64.b64encode(jpg).decode("ascii")
                }
        return out

    def move_joints_interpolated(self, target_q: List[float], duration_s: float, rate_hz: int) -> int:
        rate_hz = max(5, min(rate_hz, 200))
        steps = max(1, int(duration_s * rate_hz))
        q0 = np.array(self.get_joint_position(), dtype=float)
        q1 = np.array(target_q, dtype=float)

        for i in range(1, steps + 1):
            alpha = i / steps
            qi = (1 - alpha) * q0 + alpha * q1
            self._raw_send_joint_position(qi.tolist())
            if self._recording:
                obs = self._raw_get_obs()
                self._dataset.add_step(obs, {"q": qi.tolist()})
            time.sleep(1.0 / rate_hz)
        return steps

    def open_gripper(self, timeout_s: float) -> None:
        q = self.get_joint_position()
        q[-1] = self.gripper_open
        self._raw_send_joint_position(q)
        if self._recording:
            obs = self._raw_get_obs()
            self._dataset.add_step(obs, {"q": q})
        time.sleep(max(0.0, timeout_s))

    def close_gripper(self, timeout_s: float) -> None:
        q = self.get_joint_position()
        q[-1] = self.gripper_closed
        self._raw_send_joint_position(q)
        if self._recording:
            obs = self._raw_get_obs()
            self._dataset.add_step(obs, {"q": q})
        time.sleep(max(0.0, timeout_s))

    def stop(self) -> None:
        if self.robot is None:
            return
        q = self.get_joint_position()
        self._raw_send_joint_position(q)

    def start_record(self, session: Dict[str, Any]) -> Dict[str, Any]:
        self._recording = True
        self._record_session = session
        return self._dataset.start(session)

    def stop_record(self) -> Dict[str, Any]:
        self._recording = False
        return self._dataset.stop()

    def replay_episode(self, episode_path: str, speed_scale: float) -> None:
        # TODO: 调用 LeRobot 的 replay/eval 入口
        pass

    # --- internal helpers ---
    def _raw_get_obs(self) -> Any:
        if self.robot is None:
            raise RuntimeError("robot not connected")
        return self.robot.get_observation()

    def _raw_send_joint_position(self, q: List[float]) -> None:
        if self.robot is None:
            raise RuntimeError("robot not connected")
        if len(q) != self.dof:
            raise ValueError(f"q length must be {self.dof}")
        action = {f"{name}.pos": float(val) for name, val in zip(self.joint_names, q)}
        self.robot.send_action(action)

    def _extract_q(self, obs: Any) -> List[float]:
        q = []
        for name in self.joint_names:
            key = f"{name}.pos"
            if key not in obs:
                raise KeyError(f"missing joint in observation: {key}")
            val = obs[key]
            q.append(float(val))
        return q

    def _extract_image(self, obs: Any) -> np.ndarray:
        if self.camera_name not in obs:
            raise KeyError(f"missing camera in observation: {self.camera_name}")
        img = obs[self.camera_name]
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255.0).astype(np.uint8)
        return img

    @staticmethod
    def _encode_jpeg(img: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", img)
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return buf.tobytes()

    @staticmethod
    def _resize_keep_aspect(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale == 1.0:
            return img
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
