# src/so101_bridge/lerobot_adapter.py
from typing import Any, Dict, List, Optional, Literal
import time
import base64
import numpy as np
import cv2
import yaml

class So101LeRobotAdapter:
    def __init__(self, robot_config_path: str, camera_name: str, enable_camera: bool):
        self.robot_config_path = robot_config_path
        self.camera_name = camera_name
        self.enable_camera = enable_camera

        self.robot = None
        self.dof = 6
        self.joint_limits = None  # fill with (min,max) per joint if available

        # recording state (optional)
        self._recording = False
        self._record_session = None

    def connect(self) -> None:
        cfg = self._load_yaml(self.robot_config_path)
        # TODO: 用你当前 LeRobot 的创建方式实例化 SO101Follower/robot
        # 典型方向：
        #   - 通过 LeRobot 的 config/hydra 工厂创建 robot
        #   - 或直接 import SO101Follower 并传入 cfg
        # self.robot = ...
        # self.robot.connect()
        pass

    def disconnect(self) -> None:
        if self.robot is not None:
            # self.robot.disconnect()
            self.robot = None

    def get_joint_position(self) -> List[float]:
        obs = self._raw_get_obs()
        # TODO: 从 LeRobot observation 中取 joint position
        # return obs["observation"]["state"] or similar
        raise NotImplementedError

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
            time.sleep(1.0 / rate_hz)
        return steps

    def open_gripper(self, timeout_s: float) -> None:
        # TODO: 调用 LeRobot 的 gripper action
        pass

    def close_gripper(self, timeout_s: float) -> None:
        # TODO
        pass

    def stop(self) -> None:
        # TODO: 如果 LeRobot 提供 stop/disable torque，调用之；否则发送保持位姿
        pass

    def start_record(self, session: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: 可调用 LeRobot 的 record pipeline，或你自己把 obs/action 落盘为 LeRobot dataset
        self._recording = True
        self._record_session = session
        return {"recording": True, "session_id": session.get("name", "session"), "output_dir": session.get("output_dir", "datasets/")}

    def stop_record(self) -> Dict[str, Any]:
        self._recording = False
        return {"recording": False, "dataset_path": "datasets/..."}

    def replay_episode(self, episode_path: str, speed_scale: float) -> None:
        # TODO: 调用 LeRobot 的 replay/eval 入口
        pass

    # --- internal helpers ---
    def _raw_get_obs(self) -> Any:
        if self.robot is None:
            raise RuntimeError("robot not connected")
        # TODO: LeRobot robot step / get_observation
        raise NotImplementedError

    def _raw_send_joint_position(self, q: List[float]) -> None:
        if self.robot is None:
            raise RuntimeError("robot not connected")
        # TODO: LeRobot send_action with joint target
        raise NotImplementedError

    def _extract_q(self, obs: Any) -> List[float]:
        # TODO: map LeRobot obs to q
        raise NotImplementedError

    def _extract_image(self, obs: Any) -> np.ndarray:
        # TODO: map LeRobot obs to RGB image for camera_name
        raise NotImplementedError

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
