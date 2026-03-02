#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pi0 策略部署到 Fourier GR-2 机器人 —— 抓瓶子放盒子任务
融合了稳定的 V4L2 相机驱动逻辑与 Tensor 维度自适应修复
"""

import logging
import time
import cv2
import numpy as np
import torch
import sys

from fourier_aurora_client import AuroraClient
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import prepare_observation_for_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================

HF_MODEL_ID = "/home/phl/workspace/mymodels/pick_bottle/pi0_gr2/checkpoints/last/pretrained_model"
HF_DATASET_ID = "/home/phl/workspace/dataset/fourier/pick_bottle_and_place_into_box_lerobot_gr2"

# 摄像头配置 (使用你提供的稳健配置)
CAMERA_TOP_INDEX = 0
CAMERA_TOP_WIDTH = 640
CAMERA_TOP_HEIGHT = 480
CAMERA_TOP_FPS = 30
CAMERA_WARMUP_FRAMES = 5
CAMERA_READ_FAIL_REOPEN_THRESHOLD = 5
CAMERA_REOPEN_WAIT_S = 0.2

# 控制参数
FPS = 10                  # 建议降低到10Hz确保推理来得及
EPISODE_TIME_S = 0
TRANSITION_TIME_S = 3.0   
TRANSITION_FREQ = 100     

# Aurora SDK
DOMAIN_ID = 123
ROBOT_NAME = "gr2"   
CONTROL_FSM_STATE = 2
TASK = "pick bottle and place into box"

# 灵巧手偏移与限位
ENABLE_TRIGGER_HAND_COUPLING = True
DEX_FINGER_OFFSET = 0.17
DEX_THUMB_PITCH_OFFSET = 0.12
DEX_HAND_SDK_MIN = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf], dtype=np.float32)
DEX_HAND_SDK_MAX = np.array([1.9226667, 1.9226667, 1.9226667, 1.9226667, 1.5, np.inf], dtype=np.float32)

# ==================== 关节映射与限位 ====================

ACTION_GROUP_SLICES = {
    "left_manipulator":  slice(0, 7),
    "right_manipulator": slice(7, 14),
    "left_hand":         slice(14, 20),
    "right_hand":        slice(20, 26),
    "head":              slice(26, 28),
    "waist":             slice(28, 29),
}

STATE_GROUP_ORDER = [
    ("left_manipulator", "position"),
    ("right_manipulator", "position"),
    ("left_hand", "position"),
    ("right_hand", "position"),
    ("head", "position"),
    ("waist", "position"),
]

STATE_BASE_DATA_ORDER = [
    ("pos_W", 3), ("quat_xyzw", 4), ("rpy", 3), ("acc_B", 3), ("omega_B", 3)
]

JOINT_LIMITS = {
    "left_manipulator": [(-2.9671, 2.9671), (-0.5236, 2.7925), (-1.8326, 1.8326), (-1.5272, 0.47997), (-1.8326, 1.8326), (-0.61087, 0.61087), (-0.95993, 0.95993)],
    "right_manipulator": [(-2.9671, 2.9671), (-2.7925, 0.5236), (-1.8326, 1.8326), (-1.5272, 0.47997), (-1.8326, 1.8326), (-0.61087, 0.61087), (-0.95993, 0.95993)],
    "head": [(-1.3963, 1.3963), (-0.5236, 0.5236)],
    "waist": [(-2.618, 2.618)],
}

# ==================== 工具函数 ====================

def clamp_action(action_np: np.ndarray) -> np.ndarray:
    action = action_np.copy()
    for group_name, limits in JOINT_LIMITS.items():
        slc = ACTION_GROUP_SLICES[group_name]
        vals = action[slc]
        for i, (lo, hi) in enumerate(limits):
            vals[i] = float(np.clip(vals[i], lo, hi))
        action[slc] = vals
    return action

def apply_trigger_hand_coupling(action_np: np.ndarray) -> np.ndarray:
    if not ENABLE_TRIGGER_HAND_COUPLING: return action_np
    action = action_np.copy()
    thumb_pitch = float(action[24])
    action[20:24] = float(np.clip(-2.0 * thumb_pitch, -1.9226667, 0.0))
    return action

def _align_one_hand_urdf_to_sdk(hand_urdf: np.ndarray) -> np.ndarray:
    hand_sdk = np.array([
        DEX_FINGER_OFFSET - float(hand_urdf[0]),
        DEX_FINGER_OFFSET - float(hand_urdf[1]),
        DEX_FINGER_OFFSET - float(hand_urdf[2]),
        DEX_FINGER_OFFSET - float(hand_urdf[3]),
        float(hand_urdf[4]) - DEX_THUMB_PITCH_OFFSET,
        -float(hand_urdf[5]),
    ], dtype=np.float32)
    return np.clip(hand_sdk, DEX_HAND_SDK_MIN, DEX_HAND_SDK_MAX).astype(np.float32)

def _align_one_hand_sdk_to_urdf(hand_sdk: np.ndarray) -> np.ndarray:
    return np.array([
        DEX_FINGER_OFFSET - float(hand_sdk[0]),
        DEX_FINGER_OFFSET - float(hand_sdk[1]),
        DEX_FINGER_OFFSET - float(hand_sdk[2]),
        DEX_FINGER_OFFSET - float(hand_sdk[3]),
        DEX_THUMB_PITCH_OFFSET + float(hand_sdk[4]),
        -float(hand_sdk[5]),
    ], dtype=np.float32)

def get_robot_state(client: AuroraClient) -> np.ndarray:
    parts = []
    for group_name, key in STATE_GROUP_ORDER:
        vals = np.array(client.get_group_state(group_name, key), dtype=np.float32)
        if group_name in ("left_hand", "right_hand"): vals = _align_one_hand_sdk_to_urdf(vals)
        parts.append(vals)
    for data_key, dim in STATE_BASE_DATA_ORDER:
        parts.append(np.array(client.get_base_data(data_key)[:dim], dtype=np.float32))
    return np.concatenate(parts)

def send_upper_body_action(client: AuroraClient, action_np: np.ndarray) -> None:
    action_send = action_np.copy()
    action_send[14:20] = _align_one_hand_urdf_to_sdk(action_send[14:20])
    action_send[20:26] = _align_one_hand_urdf_to_sdk(action_send[20:26])
    position_dict = { name: action_send[slc].tolist() for name, slc in ACTION_GROUP_SLICES.items() }
    client.set_joint_positions(position_dict)

def smooth_transition(client: AuroraClient, target_action: np.ndarray, duration: float = 3.0, frequency: int = 100) -> None:
    init_joints = get_robot_state(client)[:29]
    target_joints = target_action[:29]
    total_steps = int(frequency * duration)
    logger.info(f"平滑过渡到初始姿态 ({duration}s, {total_steps} steps)...")
    for step in range(total_steps + 1):
        alpha = step / total_steps
        send_upper_body_action(client, init_joints * (1.0 - alpha) + target_joints * alpha)
        time.sleep(1.0 / frequency)

def connect_aurora_client(domain_id: int, robot_name: str, retries: int = 3, retry_interval_s: float = 2.0) -> AuroraClient:
    for attempt in range(1, retries + 1):
        logger.info(f"连接 GR-2 (domain={domain_id}, robot={robot_name}), 尝试 {attempt}/{retries}...")
        client = AuroraClient.get_instance(domain_id=domain_id, robot_name=robot_name, serial_number=None)
        if client is not None:
            time.sleep(1.0)
            return client
        if attempt < retries:
            time.sleep(retry_interval_s)
    raise RuntimeError("AuroraClient 初始化失败。")

# --- 你的原生相机代码逻辑 ---
def open_top_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开 camera_top (index={index})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_TOP_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(CAMERA_WARMUP_FRAMES): cap.read()
    return cap

def reopen_top_camera(cap: cv2.VideoCapture | None) -> cv2.VideoCapture:
    if cap is not None:
        try: cap.release()
        except Exception: pass
    time.sleep(CAMERA_REOPEN_WAIT_S)
    return open_top_camera(CAMERA_TOP_INDEX, CAMERA_TOP_WIDTH, CAMERA_TOP_HEIGHT)

# ==================== 主程序 ====================

def main():
    logger.info(f"加载 Pi0 模型: {HF_MODEL_ID}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PI0Policy.from_pretrained(HF_MODEL_ID).to(device)
    policy.eval()

    logger.info(f"加载数据集 stats: {HF_DATASET_ID}")
    dataset = LeRobotDataset(HF_DATASET_ID)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy, pretrained_path=HF_MODEL_ID, dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    logger.info("连接摄像头...")
    cap_top = open_top_camera(CAMERA_TOP_INDEX, CAMERA_TOP_WIDTH, CAMERA_TOP_HEIGHT)
    cam_read_fail_count = 0

    client = connect_aurora_client(domain_id=DOMAIN_ID, robot_name=ROBOT_NAME)

    # ===== 关键加入：CUDA 模型预热，解决“回车没反应” =====
    logger.info("正在执行 CUDA 首帧预热推理...")
    with torch.inference_mode():
        dummy_obs = {"observation.images.camera_top": np.zeros((480, 640, 3), dtype=np.uint8), "observation.state": np.zeros(45, dtype=np.float32)}
        dummy_in = prepare_observation_for_inference(dummy_obs, device=device, task=TASK, robot_type="fourier_gr2")
        _ = policy.select_action(preprocessor(dummy_in))
    logger.info("预热完成。")
    # ===================================================

    try:
        input("确认机器人安全后, 按回车切换到 PdStand (FSM 2)...")
        print("[DBG] 正在 set_fsm_state(2)...", flush=True)
        client.set_fsm_state(2)
        time.sleep(1.0)
        print("[DBG] 已进入 PdStand 状态", flush=True)

        if CONTROL_FSM_STATE != 2:
            input(f"按回车切换到控制模式 (FSM {CONTROL_FSM_STATE})...")
            client.set_fsm_state(CONTROL_FSM_STATE)
            time.sleep(0.5)

        print("[DBG] 重置策略对象...", flush=True)
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        
        first_action = True
        step_count = 0
        episode_start = time.time()
        logger.info(f"进入控制循环, 目标频率 {FPS} Hz (Ctrl+C 停止)...")

        while True:
            if EPISODE_TIME_S > 0 and (time.time() - episode_start) >= EPISODE_TIME_S: break
            loop_start = time.time()

            # --- 稳定相机读取 ---
            ret, frame_bgr = cap_top.read()
            if not ret:
                cam_read_fail_count += 1
                if cam_read_fail_count >= CAMERA_READ_FAIL_REOPEN_THRESHOLD:
                    logger.warning(f"摄像头连续读取失败 {cam_read_fail_count} 次，尝试重连...")
                    cap_top = reopen_top_camera(cap_top)
                    cam_read_fail_count = 0
                time.sleep(0.02)
                continue
            cam_read_fail_count = 0
            
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            state = get_robot_state(client)

            # --- 策略推理 ---
            obs = {"observation.images.camera_top": frame_rgb, "observation.state": state}
            with torch.inference_mode():
                observation = prepare_observation_for_inference(obs, device=device, task=TASK, robot_type="fourier_gr2")
                observation = preprocessor(observation)
                action_tensor = policy.select_action(observation)
                action_tensor = postprocessor(action_tensor) 

            # ===== 关键修复：解决 IndexError 维度报错 =====
            # 将张量展平为二维 (Time, Dim)，无视它原来是 2D 还是 3D
            action_2d = action_tensor.view(-1, action_tensor.shape[-1])
            # 提取时间步为 0 的前 35 维数据
            action_np = action_2d[0, :35].cpu().numpy().astype(np.float32)
            # ===============================================

            # --- 打印与安全处理 ---
            if step_count % 10 == 0:
                print(f"--- step {step_count} ---", flush=True)
                print(f"  [state] left_arm= {state[0:7]} | right_arm= {state[7:14]}", flush=True)
                arm_delta_norm = float(np.linalg.norm(action_np[0:14] - state[0:14]))
                print(f"  [diag]  arm_delta_norm= {arm_delta_norm:.5f}", flush=True)

            action_np = apply_trigger_hand_coupling(action_np)
            action_np = clamp_action(action_np)

            # --- 发送指令 ---
            if first_action:
                smooth_transition(client, action_np, duration=TRANSITION_TIME_S, frequency=TRANSITION_FREQ)
                first_action = False
                episode_start = time.time()
                step_count = 0
            else:
                send_upper_body_action(client, action_np)
                step_count += 1

            # --- 频率控制 ---
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, 1.0 / FPS - elapsed)
            if sleep_time == 0 and step_count % 10 == 1:
                logger.warning(f"推理延迟报警: {1.0/elapsed:.1f} Hz < {FPS} Hz")
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("用户中断 (Ctrl+C)")
    finally:
        if cap_top is not None:
            try: cap_top.release()
            except: pass
        if client is not None: client.close()
        logger.info("控制结束，资源已释放。")

if __name__ == "__main__":
    main()