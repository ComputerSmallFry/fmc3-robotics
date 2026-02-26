# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Always answer the user's questions in Chinese.

## 项目概述

Fourier 机器人数据集转换工具。将外骨骼遥操采集的 Dora-Record 格式数据转换为 LeRobot v3.0 格式，用于策略模型训练（ACT、Diffusion、PI0、SmolVLA 等）。

核心代码只有一个文件：`convert_tools/convert_dora_to_lerobot.py`。

## 运行命令

```bash
conda activate lerobot  # 依赖: pyarrow, numpy, pandas, PIL; 外部: ffmpeg

# 快速运行（预设参数）
bash convert_tools/run_convert.sh

# 手动运行
python convert_tools/convert_dora_to_lerobot.py \
    --input ./pick_bottle_and_place_into_box \
    --output ./pick_bottle_and_place_into_box_lerobot \
    --task "pick bottle and place into box" \
    --fps 30 \
    --robot-type fourier_gr2 \
    --video-codec libopenh264
```

注意：conda 环境的 ffmpeg 不带 libx264，用 `--video-codec libopenh264`。加 `--no-video` 可跳过图像/视频只转换关节数据。

## 转换流水线架构

`convert()` 主流程：

1. 扫描输入目录下所有 `episode_*` 子目录
2. `ProcessPoolExecutor` 并行调用 `_load_single_episode()` 处理每个 episode：
   - 读取各 parquet 文件（action/state/base/images，频率各不同）
   - 过滤关节（GR2 去掉 waist_roll/pitch，见 `ACTION_FILTER_JOINTS`）
   - 按 `GR2_JOINT_ORDER` 重排关节顺序对齐 SDK 控制组
   - 生成统一时间轴 → 最近邻重采样对齐所有传感器
   - 拼接 action (29关节+6底盘=35D) 和 state (29关节+16底盘IMU=45D)
3. 编码视频（ffmpeg subprocess）
4. 计算全局归一化统计（min/max/mean/std）
5. 写入 LeRobot v3.0 格式（parquet + mp4 + meta json）

## 关节维度映射

`GR2_JOINT_ORDER`（29个关节，对齐 SDK 控制组顺序）：
left_manipulator(7) → right_manipulator(7) → left_hand(6) → right_hand(6) → head(2) → waist(1)

- Action 35D = 29 关节 + 6 底盘速度
- State 45D = 29 关节 + 3 base_pos + 4 base_quat + 3 base_rpy + 3 imu_acc + 3 imu_omega

## 输入/输出格式

输入（Dora-Record）每个 episode 包含：`metadata.json`、`action.parquet`（~100Hz）、`action.base.parquet`（~100Hz）、`observation.state.parquet`（~60Hz）、`observation.base_state.parquet`（~60Hz）、`observation.images.camera_top.parquet`（~30Hz）、`observation.images.camera_top_depth.parquet`（~30Hz）。

输出（LeRobot v3.0）：`meta/info.json`、`meta/stats.json`、`meta/tasks.parquet`、`data/chunk-*/file-*.parquet`、`videos/{camera_key}/chunk-*/file-*.mp4`。

## 中文文档

`convert_tools/` 下：
- `CONVERT_GUIDE.md` — 完整转换指南和参数说明
- `dora-record结构.md` — 输入格式详解
- `lerobot-gr2结构.md` — 输出格式和维度映射
- `GR-2 各控制组参数详解.md` — 41自由度关节参数
- `GR2推理适配指南.md` — 模型推理部署代码示例
