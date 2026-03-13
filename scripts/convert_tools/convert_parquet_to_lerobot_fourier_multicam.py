#!/usr/bin/env python3
"""
将 Dora parquet episode 转换为 LeRobot 数据集，默认支持 GR2 的三路 RGB 相机：

- observation.images.camera_top
- observation.images.camera_left_wrist
- observation.images.camera_right_wrist

该版本用于 Intel RealSense D435 多相机场景，同时支持 RGB 和 depth。
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


GR2_JOINT_ORDER = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_ring_proximal_joint",
    "L_pinky_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_ring_proximal_joint",
    "R_pinky_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "waist_yaw_joint",
]

ACTION_FILTER_JOINTS = {"waist_roll_joint", "waist_pitch_joint"}
BASE_ACTION_NAMES = [
    "base_vel_x",
    "base_vel_y",
    "base_vel_yaw",
    "base_vel_height",
    "base_vel_pitch",
    "base_base_yaw",
]
BASE_STATE_NAMES = [
    "base_pos_x",
    "base_pos_y",
    "base_pos_z",
    "base_quat_x",
    "base_quat_y",
    "base_quat_z",
    "base_quat_w",
    "base_rpy_roll",
    "base_rpy_pitch",
    "base_rpy_yaw",
    "imu_acc_x",
    "imu_acc_y",
    "imu_acc_z",
    "imu_omega_x",
    "imu_omega_y",
    "imu_omega_z",
]

DEFAULT_CAMERAS = ["camera_top", "camera_left_wrist", "camera_right_wrist"]
RAW_IMAGE_SHAPE_CANDIDATES = [
    (480, 640, 3),
    (480, 848, 3),
    (720, 1280, 3),
    (800, 1280, 3),
    (480, 640, 4),
    (480, 848, 4),
]


def log(message: str) -> None:
    print(message, flush=True)


def normalize_camera_keys(camera_keys: list[str]) -> list[str]:
    normalized = []
    for key in camera_keys:
        camera_key = key if key.startswith("camera_") else f"camera_{key}"
        if camera_key.endswith("_depth"):
            raise ValueError(f"暂不支持深度流作为 LeRobot 视频键: {camera_key}")
        normalized.append(camera_key)
    return normalized


def depth_stream_key(camera_key: str) -> str:
    return f"{camera_key}_depth"


def find_episode_dirs(raw_dir: Path) -> list[Path]:
    return sorted(path for path in raw_dir.rglob("episode_*") if path.is_dir())


def reorder_to_target(source_names: list[str], values: np.ndarray, target_names: list[str]) -> tuple[list[str], np.ndarray]:
    index_map = {name: idx for idx, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in index_map]
    if missing:
        raise ValueError(f"缺少目标关节: {missing}")
    reorder_indices = [index_map[name] for name in target_names]
    return list(target_names), values[:, reorder_indices]


def read_named_list_column(parquet_path: Path, column_name: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=["timestamp_utc", column_name])
    first_row = df[column_name].iloc[0]
    names = [item["name"] for item in first_row]

    values = np.zeros((len(df), len(names)), dtype=np.float32)
    for row_idx, row in enumerate(df[column_name]):
        name_to_value = {item["name"]: float(item["value"]) for item in row}
        for col_idx, name in enumerate(names):
            values[row_idx, col_idx] = name_to_value.get(name, 0.0)

    timestamps_ns = df["timestamp_utc"].astype("int64").to_numpy(dtype=np.int64)
    return names, values, timestamps_ns


def read_base_state_column(parquet_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet_path, columns=["timestamp_utc", "observation.base_state"])
    values = np.zeros((len(df), len(BASE_STATE_NAMES)), dtype=np.float32)

    for row_idx, row in enumerate(df["observation.base_state"]):
        if row is None or len(row) == 0:
            continue
        entry = row[0]
        base = entry.get("base", {})
        imu = entry.get("imu", {})

        flat = (
            list(base.get("position", [0.0, 0.0, 0.0]))
            + list(base.get("quat", [0.0, 0.0, 0.0, 0.0]))
            + list(base.get("rpy", [0.0, 0.0, 0.0]))
            + list(imu.get("acc_B", [0.0, 0.0, 0.0]))
            + list(imu.get("omega_B", [0.0, 0.0, 0.0]))
        )
        values[row_idx, :] = np.asarray(flat[: len(BASE_STATE_NAMES)], dtype=np.float32)

    timestamps_ns = df["timestamp_utc"].astype("int64").to_numpy(dtype=np.int64)
    return list(BASE_STATE_NAMES), values, timestamps_ns


def read_image_column(parquet_path: Path, column_name: str) -> tuple[list[bytes], np.ndarray]:
    images: list[bytes] = []
    timestamps: list[int] = []
    try:
        parquet_file = pq.ParquetFile(str(parquet_path))
        for batch in parquet_file.iter_batches(batch_size=64, columns=[column_name, "timestamp_utc"]):
            image_col = batch.column(column_name)
            timestamp_col = batch.column("timestamp_utc").cast(pa.int64())
            for row_idx in range(len(batch)):
                images.append(bytes(image_col[row_idx].as_py()))
                timestamps.append(int(timestamp_col[row_idx].as_py()))
    except Exception as exc:
        raise RuntimeError(f"读取图像 parquet 失败: {parquet_path}") from exc

    if not images:
        raise ValueError(f"图像 parquet 为空: {parquet_path}")
    return images, np.asarray(timestamps, dtype=np.int64)


def nearest_indices(source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    if source_ts.ndim != 1 or len(source_ts) == 0:
        raise ValueError("source_ts 必须是一维非空数组")

    right = np.searchsorted(source_ts, target_ts, side="left")
    right = np.clip(right, 0, len(source_ts) - 1)
    left = np.clip(right - 1, 0, len(source_ts) - 1)
    choose_left = np.abs(target_ts - source_ts[left]) <= np.abs(source_ts[right] - target_ts)
    return np.where((right == 0) | choose_left, left, right)


def sample_array_by_timestamps(values: np.ndarray, source_ts: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    return values[nearest_indices(source_ts, target_ts)]


def sample_list_by_timestamps(values: list[bytes], source_ts: np.ndarray, target_ts: np.ndarray) -> list[bytes]:
    return [values[idx] for idx in nearest_indices(source_ts, target_ts)]


def generate_target_timestamps(source_timestamps: list[np.ndarray], fps: int) -> np.ndarray:
    start_ns = max(int(ts.min()) for ts in source_timestamps)
    end_ns = min(int(ts.max()) for ts in source_timestamps)
    if start_ns > end_ns:
        raise ValueError("多路数据时间范围没有交集，无法对齐")

    step_ns = int(round(1e9 / fps))
    target_ts = np.arange(start_ns, end_ns + 1, step_ns, dtype=np.int64)
    if len(target_ts) == 0:
        target_ts = np.asarray([start_ns], dtype=np.int64)
    return target_ts


def detect_raw_image_shape(raw_bytes: bytes) -> tuple[int, int, int] | None:
    length = len(raw_bytes)
    for shape in RAW_IMAGE_SHAPE_CANDIDATES:
        if np.prod(shape) == length:
            return shape
    return None


def decode_rgb_frame(raw_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is not None:
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    try:
        return np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
    except Exception:
        pass

    try:
        image = iio.imread(raw_bytes, extension=".avif")
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[-1] == 4:
            image = image[:, :, :3]
        return image.astype(np.uint8)
    except Exception:
        pass

    raw_shape = detect_raw_image_shape(raw_bytes)
    if raw_shape is None:
        raise ValueError(f"无法识别图像字节格式，长度={len(raw_bytes)}")

    height, width, channels = raw_shape
    image = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(height, width, channels)
    if channels == 4:
        image = image[:, :, :3]
    return image


def decode_depth_frame(raw_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(raw_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if decoded is not None:
        return decoded

    try:
        return np.asarray(Image.open(io.BytesIO(raw_bytes)))
    except Exception:
        pass

    try:
        return iio.imread(raw_bytes, extension=".png")
    except Exception as exc:
        raise ValueError(f"无法识别深度图字节格式，长度={len(raw_bytes)}") from exc


def depth_to_rgb_frame(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[:, :, 0]

    if depth.ndim == 3 and depth.shape[-1] >= 3:
        rgb = depth[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    depth_float = depth.astype(np.float32)
    positive = depth_float[depth_float > 0]
    if positive.size == 0:
        depth_u8 = np.zeros(depth.shape[:2], dtype=np.uint8)
    else:
        depth_min = float(positive.min())
        depth_max = float(positive.max())
        if depth_max <= depth_min:
            depth_u8 = np.zeros(depth.shape[:2], dtype=np.uint8)
        else:
            normalized = np.clip((depth_float - depth_min) / (depth_max - depth_min), 0.0, 1.0)
            depth_u8 = (normalized * 255.0).astype(np.uint8)
        depth_u8[depth_float <= 0] = 0

    return np.repeat(depth_u8[:, :, None], 3, axis=2)


def resize_frame(image: np.ndarray, target_hw: tuple[int, int] | None) -> np.ndarray:
    if target_hw is None:
        return image
    target_h, target_w = target_hw
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def decode_stream_frames(
    sampled_bytes: list[bytes],
    resize_hw: tuple[int, int] | None,
    is_depth: bool,
    max_workers: int,
) -> list[np.ndarray]:
    def _decode(raw_bytes: bytes) -> np.ndarray:
        if is_depth:
            frame = depth_to_rgb_frame(decode_depth_frame(raw_bytes))
        else:
            frame = decode_rgb_frame(raw_bytes)
        frame = resize_frame(frame, resize_hw)
        return frame.astype(np.uint8)

    worker_count = max(1, min(max_workers, len(sampled_bytes)))
    if worker_count == 1:
        return [_decode(raw_bytes) for raw_bytes in sampled_bytes]

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_decode, sampled_bytes))


def parse_resize(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    width, height = map(int, value.lower().split("x"))
    return height, width


def inspect_dataset_schema(
    episode_dir: Path,
    camera_keys: list[str],
    use_videos: bool,
    include_depth: bool,
    resize_hw: tuple[int, int] | None,
) -> tuple[list[str], list[str], dict[str, tuple[int, int]]]:
    log(f"检查数据 schema: {episode_dir}")
    action_names, _, _ = read_named_list_column(episode_dir / "action.parquet", "action")
    keep_indices = [idx for idx, name in enumerate(action_names) if name not in ACTION_FILTER_JOINTS]
    action_names = [action_names[idx] for idx in keep_indices]
    action_names, _ = reorder_to_target(action_names, np.zeros((1, len(action_names)), dtype=np.float32), GR2_JOINT_ORDER)

    state_names, _, _ = read_named_list_column(episode_dir / "observation.state.parquet", "observation.state")
    state_names, _ = reorder_to_target(
        state_names,
        np.zeros((1, len(state_names)), dtype=np.float32),
        GR2_JOINT_ORDER,
    )

    combined_action_names = action_names + BASE_ACTION_NAMES
    combined_state_names = state_names + BASE_STATE_NAMES

    camera_shapes: dict[str, tuple[int, int]] = {}
    if use_videos:
        for camera_key in camera_keys:
            parquet_path = episode_dir / f"observation.images.{camera_key}.parquet"
            column_name = f"observation.images.{camera_key}"
            images, _ = read_image_column(parquet_path, column_name)
            first_frame = resize_frame(decode_rgb_frame(images[0]), resize_hw)
            camera_shapes[camera_key] = (int(first_frame.shape[0]), int(first_frame.shape[1]))
            if include_depth:
                depth_key = depth_stream_key(camera_key)
                depth_path = episode_dir / f"observation.images.{depth_key}.parquet"
                if depth_path.exists():
                    depth_images, _ = read_image_column(depth_path, f"observation.images.{depth_key}")
                    first_depth = resize_frame(depth_to_rgb_frame(decode_depth_frame(depth_images[0])), resize_hw)
                    camera_shapes[depth_key] = (int(first_depth.shape[0]), int(first_depth.shape[1]))

    return combined_action_names, combined_state_names, camera_shapes


def build_features(
    action_names: list[str],
    state_names: list[str],
    camera_shapes: dict[str, tuple[int, int]],
) -> dict[str, dict]:
    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
    }
    for camera_key, (height, width) in camera_shapes.items():
        features[f"observation.images.{camera_key}"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def discover_default_task(episode_dirs: list[Path]) -> str:
    for episode_dir in episode_dirs:
        metadata_path = episode_dir / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        notes = metadata.get("notes")
        if notes:
            return str(notes)
    return " "


def cleanup_temporary_images(output_dir: Path) -> None:
    images_dir = output_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)


def process_episode(
    episode_dir: Path,
    episode_index: int,
    fps: int,
    camera_keys: list[str],
    use_videos: bool,
    include_depth: bool,
    resize_hw: tuple[int, int] | None,
    decode_workers: int,
    task_override: str | None,
    default_task: str,
) -> tuple[dict[str, np.ndarray], str]:
    episode_start = time.perf_counter()
    log(f"  [{episode_index}] 读取状态和动作: {episode_dir.name}")
    metadata_path = episode_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    action_names, action_values, action_ts = read_named_list_column(episode_dir / "action.parquet", "action")
    keep_indices = [idx for idx, name in enumerate(action_names) if name not in ACTION_FILTER_JOINTS]
    action_values = action_values[:, keep_indices]
    action_names = [action_names[idx] for idx in keep_indices]
    action_names, action_values = reorder_to_target(action_names, action_values, GR2_JOINT_ORDER)

    state_names, state_values, state_ts = read_named_list_column(
        episode_dir / "observation.state.parquet",
        "observation.state",
    )
    state_names, state_values = reorder_to_target(state_names, state_values, GR2_JOINT_ORDER)

    _, base_action_values, base_action_ts = read_named_list_column(episode_dir / "action.base.parquet", "action.base")
    _, base_state_values, base_state_ts = read_base_state_column(episode_dir / "observation.base_state.parquet")

    timestamps_to_align = [action_ts, state_ts, base_action_ts, base_state_ts]
    raw_camera_frames: dict[str, list[bytes]] = {}
    raw_camera_timestamps: dict[str, np.ndarray] = {}
    if use_videos:
        log(f"  [{episode_index}] 读取相机流: {len(camera_keys)} 路 RGB，depth={'on' if include_depth else 'off'}")
        for camera_key in camera_keys:
            parquet_path = episode_dir / f"observation.images.{camera_key}.parquet"
            column_name = f"observation.images.{camera_key}"
            images, image_ts = read_image_column(parquet_path, column_name)
            raw_camera_frames[camera_key] = images
            raw_camera_timestamps[camera_key] = image_ts
            timestamps_to_align.append(image_ts)
            if include_depth:
                depth_key = depth_stream_key(camera_key)
                depth_path = episode_dir / f"observation.images.{depth_key}.parquet"
                if depth_path.exists():
                    depth_images, depth_ts = read_image_column(depth_path, f"observation.images.{depth_key}")
                    raw_camera_frames[depth_key] = depth_images
                    raw_camera_timestamps[depth_key] = depth_ts
                    timestamps_to_align.append(depth_ts)

    log(f"  [{episode_index}] 时间戳对齐和重采样")
    target_ts = generate_target_timestamps(timestamps_to_align, fps)
    frame_data: dict[str, np.ndarray | list[np.ndarray]] = {
        "action": np.concatenate(
            [
                sample_array_by_timestamps(action_values, action_ts, target_ts),
                sample_array_by_timestamps(base_action_values, base_action_ts, target_ts),
            ],
            axis=1,
        ).astype(np.float32),
        "observation.state": np.concatenate(
            [
                sample_array_by_timestamps(state_values, state_ts, target_ts),
                sample_array_by_timestamps(base_state_values, base_state_ts, target_ts),
            ],
            axis=1,
        ).astype(np.float32),
    }

    if use_videos:
        for camera_key in camera_keys:
            log(f"  [{episode_index}] 解码视频流: {camera_key} ({len(target_ts)} 帧)")
            sampled_bytes = sample_list_by_timestamps(
                raw_camera_frames[camera_key],
                raw_camera_timestamps[camera_key],
                target_ts,
            )
            frame_data[f"observation.images.{camera_key}"] = decode_stream_frames(
                sampled_bytes=sampled_bytes,
                resize_hw=resize_hw,
                is_depth=False,
                max_workers=decode_workers,
            )
            if include_depth:
                depth_key = depth_stream_key(camera_key)
                if depth_key in raw_camera_frames:
                    log(f"  [{episode_index}] 解码视频流: {depth_key} ({len(target_ts)} 帧)")
                    sampled_depth_bytes = sample_list_by_timestamps(
                        raw_camera_frames[depth_key],
                        raw_camera_timestamps[depth_key],
                        target_ts,
                    )
                    frame_data[f"observation.images.{depth_key}"] = decode_stream_frames(
                        sampled_bytes=sampled_depth_bytes,
                        resize_hw=resize_hw,
                        is_depth=True,
                        max_workers=decode_workers,
                    )

    task = task_override or metadata.get("notes") or default_task
    log(
        f"  [{episode_index}] episode 准备完成: {episode_dir.name}, "
        f"frames={len(target_ts)}, cost={time.perf_counter() - episode_start:.1f}s"
    )
    return frame_data, task


def process_episode_worker(
    episode_index: int,
    episode_dir_str: str,
    fps: int,
    camera_keys: list[str],
    use_videos: bool,
    include_depth: bool,
    resize_hw: tuple[int, int] | None,
    decode_workers: int,
    task_override: str | None,
    default_task: str,
) -> dict:
    episode_dir = Path(episode_dir_str)
    try:
        frame_data, task = process_episode(
            episode_dir=episode_dir,
            episode_index=episode_index,
            fps=fps,
            camera_keys=camera_keys,
            use_videos=use_videos,
            include_depth=include_depth,
            resize_hw=resize_hw,
            decode_workers=decode_workers,
            task_override=task_override,
            default_task=default_task,
        )
        return {
            "episode_index": episode_index,
            "episode_dir": str(episode_dir),
            "frame_data": frame_data,
            "task": task,
            "error": None,
        }
    except Exception as exc:
        return {
            "episode_index": episode_index,
            "episode_dir": str(episode_dir),
            "frame_data": None,
            "task": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def write_episode_to_dataset(
    dataset: LeRobotDataset,
    result: dict,
    camera_shapes: dict[str, tuple[int, int]],
    use_videos: bool,
) -> None:
    episode_dir = Path(result["episode_dir"])
    frame_data = result["frame_data"]
    task = result["task"]
    num_frames = len(frame_data["action"])

    log(f"  写入 LeRobot episode 并编码视频: {episode_dir.name}, frames={num_frames}")
    for frame_idx in range(num_frames):
        frame = {
            "action": frame_data["action"][frame_idx],
            "observation.state": frame_data["observation.state"][frame_idx],
            "task": task,
        }
        if use_videos:
            for camera_key in camera_shapes:
                feature_key = f"observation.images.{camera_key}"
                frame[feature_key] = frame_data[feature_key][frame_idx]
        dataset.add_frame(frame)

    dataset.save_episode(parallel_encoding=False)
    log(f"  episode 保存完成: {episode_dir.name}")


def convert(args: argparse.Namespace) -> Path:
    convert_start = time.perf_counter()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    camera_keys = normalize_camera_keys(args.camera_keys)
    use_videos = not args.no_video
    include_depth = not args.no_depth
    resize_hw = parse_resize(args.resize)

    episode_dirs = find_episode_dirs(raw_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"未找到 episode_* 目录: {raw_dir}")
    default_task = args.task or discover_default_task(episode_dirs)
    log(f"开始转换: raw_dir={raw_dir}")
    log(f"输出目录: {output_dir}")
    log(f"任务描述: {default_task}")
    log(f"检测到 episode 数量: {len(episode_dirs)}")
    log(
        "视频配置: "
        f"rgb_cameras={camera_keys}, depth={'on' if not args.no_depth else 'off'}, "
        f"fps={args.fps}, workers={args.workers}, decode_workers={args.decode_workers}, image_writer_threads={args.image_writer_threads}, "
        f"vcodec={args.vcodec}"
    )

    action_names, state_names, camera_shapes = inspect_dataset_schema(
        episode_dirs[0],
        camera_keys,
        use_videos,
        include_depth,
        resize_hw,
    )

    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"输出目录已存在: {output_dir}，如需覆盖请加 --force")
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id or output_dir.name,
        root=output_dir,
        fps=args.fps,
        robot_type="fourier_gr2",
        features=build_features(action_names, state_names, camera_shapes),
        use_videos=use_videos,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=0,
        vcodec=args.vcodec,
    )

    common_args = (
        args.fps,
        camera_keys,
        use_videos,
        include_depth,
        resize_hw,
        args.decode_workers,
        args.task,
        default_task,
    )
    worker_count = max(1, args.workers)
    saved_episode_count = 0
    skipped_episodes: list[tuple[Path, str]] = []

    if worker_count <= 1:
        for episode_idx, episode_dir in enumerate(episode_dirs):
            log(f"[{episode_idx + 1}/{len(episode_dirs)}] {episode_dir}")
            result = process_episode_worker(
                episode_idx,
                str(episode_dir),
                *common_args,
            )
            if result["error"] is not None:
                skipped_episodes.append((episode_dir, result["error"]))
                log(f"[WARN] 跳过 episode: {episode_dir} -> {result['error']}")
                continue
            write_episode_to_dataset(dataset, result, camera_shapes, use_videos)
            saved_episode_count += 1
    else:
        log(f"使用 {worker_count} 个进程并行预处理 episode")
        future_by_index = {}
        next_submit_index = 0

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            initial_submit = min(worker_count, len(episode_dirs))
            for _ in range(initial_submit):
                episode_dir = episode_dirs[next_submit_index]
                log(f"[submit {next_submit_index + 1}/{len(episode_dirs)}] {episode_dir}")
                future_by_index[next_submit_index] = executor.submit(
                    process_episode_worker,
                    next_submit_index,
                    str(episode_dir),
                    *common_args,
                )
                next_submit_index += 1

            for next_write_index, episode_dir in enumerate(episode_dirs):
                result = future_by_index.pop(next_write_index).result()
                if result["error"] is not None:
                    skipped_episodes.append((episode_dir, result["error"]))
                    log(f"[WARN] 跳过 episode: {episode_dir} -> {result['error']}")
                    if "traceback" in result:
                        log(result["traceback"].rstrip())
                else:
                    log(f"[write {next_write_index + 1}/{len(episode_dirs)}] {episode_dir}")
                    write_episode_to_dataset(dataset, result, camera_shapes, use_videos)
                    saved_episode_count += 1

                if next_submit_index < len(episode_dirs):
                    submit_dir = episode_dirs[next_submit_index]
                    log(f"[submit {next_submit_index + 1}/{len(episode_dirs)}] {submit_dir}")
                    future_by_index[next_submit_index] = executor.submit(
                        process_episode_worker,
                        next_submit_index,
                        str(submit_dir),
                        *common_args,
                    )
                    next_submit_index += 1

    if saved_episode_count == 0:
        raise RuntimeError("没有成功转换的 episode，请检查输入数据或日志中的跳过原因")

    dataset.finalize()
    cleanup_temporary_images(output_dir)
    if skipped_episodes:
        log(f"转换结束，跳过 {len(skipped_episodes)} 个 episode:")
        for episode_dir, reason in skipped_episodes[:20]:
            log(f"  - {episode_dir}: {reason}")
        if len(skipped_episodes) > 20:
            log(f"  ... 其余 {len(skipped_episodes) - 20} 个跳过 episode 已省略")
    log(f"转换完成: {output_dir} (total_cost={time.perf_counter() - convert_start:.1f}s)")
    return output_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 Fourier GR2 Dora parquet 转为 LeRobot，多相机版本。")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="/home/phl/workspace/dataset/fourier/gr2_test_muticam_dora",
        help="Dora 数据集根目录",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="输出 LeRobot 数据集目录")
    parser.add_argument("--repo-id", type=str, default=None, help="写入 meta 时使用的 repo_id，默认取输出目录名")
    parser.add_argument("--task", type=str, default=None, help="覆盖 metadata.json 中的 notes 作为任务文本")
    parser.add_argument("--fps", type=int, default=30, help="输出数据集帧率")
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="RGB 相机键，支持 camera_top / camera_left_wrist / camera_right_wrist，也支持省略 camera_ 前缀",
    )
    parser.add_argument("--resize", type=str, default=None, help="统一缩放到 WxH，例如 640x480")
    parser.add_argument("--no-video", action="store_true", help="仅转换 state/action，不写视频")
    parser.add_argument("--no-depth", action="store_true", help="仅写 RGB，不写 *_depth 视频")
    parser.add_argument("--workers", type=int, default=1, help="episode 级并行进程数")
    parser.add_argument("--decode-workers", type=int, default=8, help="单路视频帧解码线程数")
    parser.add_argument("--image-writer-threads", type=int, default=12, help="LeRobot 异步图片写线程数")
    parser.add_argument("--vcodec", type=str, default="h264", choices=["h264", "hevc", "libsvtav1"])
    parser.add_argument("--force", action="store_true", help="输出目录存在时先删除再重建")
    return parser


if __name__ == "__main__":
    convert(build_argparser().parse_args())
