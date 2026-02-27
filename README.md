# Fourier Dora-Record → LeRobot v3.0 数据集转换工具

将 Fourier 人形机器人（GR-2/GR-3）外骨骼遥操采集的 Dora-Record 格式数据转换为 [LeRobot v3.0](https://github.com/huggingface/lerobot) 格式，用于策略模型训练（ACT、Diffusion、PI0、SmolVLA 等）。

## 快速开始

```bash
# 环境要求：pyarrow, numpy, pandas, Pillow, ffmpeg
conda activate lerobot

# 使用预设参数运行
bash scripts/convert_tools/run_convert.sh

# 或手动指定参数
python scripts/convert_tools/convert_dora_to_lerobot.py \
    --input ./pick_bottle_and_place_into_box \
    --output ./pick_bottle_and_place_into_box_lerobot \
    --task "pick bottle and place into box" \
    --fps 30 \
    --robot-type fourier_gr2 \
    --video-codec libopenh264 \
    --workers 4
```

> conda 环境的 ffmpeg 不带 libx264，请使用 `--video-codec libopenh264`。

## 命令行参数

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必填 | Dora-Record session 目录（包含 `episode_*` 子目录） |
| `--output` | `-o` | 必填 | 输出 LeRobot 数据集目录 |
| `--task` | `-t` | `teleoperation` | 任务自然语言描述，VLA 训练时作为语言指令 |
| `--fps` | | `30` | 目标帧率，所有传感器数据重采样到此帧率 |
| `--robot-type` | | `fourier_gr3` | 机器人类型标识 |
| `--video-codec` | | `libx264` | 视频编码器（推荐 `libopenh264`） |
| `--workers` | `-w` | `1` | 并行进程数，设为 CPU 核数一半可加速 3-4 倍 |
| `--no-video` | | | 跳过图像/视频，只转换关节数据 |

## 转换流程

```
Dora-Record                          LeRobot v3.0
┌─────────────────┐                  ┌──────────────────────┐
│ action.parquet  │─┐                │ data/chunk-000/      │
│ (31D, ~100Hz)   │ │  重采样+拼接   │   file-000.parquet   │
│                 │ ├──────────────→ │   (action 35D,       │
│ action.base     │ │   统一到       │    state 45D,        │
│ (6D, ~100Hz)    │─┘   30fps       │    timestamps...)     │
│                 │                  │                      │
│ obs.state       │─┐               │ videos/              │
│ (29D, ~60Hz)    │ ├──────────────→ │   camera_top/        │
│ obs.base_state  │─┘               │     file-000.mp4     │
│ (16D, ~60Hz)    │                  │     file-001.mp4     │
│                 │  解码+编码       │   camera_top_depth/  │
│ camera_top      │──────────────→   │     file-000.mp4     │
│ (JPEG, ~30Hz)   │                  │                      │
│ camera_depth    │──────────────→   │ meta/                │
│ (PNG, ~30Hz)    │                  │   info.json          │
└─────────────────┘                  │   stats.json         │
                                     │   tasks.parquet      │
                                     └──────────────────────┘
```

每个 episode 独立处理：读取 → 时间对齐 → 关节重排 → 编码视频 → 释放内存，峰值约 2GB/进程。

## 关节维度映射

转换时按 GR-2 SDK 控制组顺序重排关节，部署时可直接切片下发：

**Action（35D）**= 29 关节 + 6 底盘速度

| 维度 | 控制组 | 内容 |
|------|--------|------|
| 0-6 | left_manipulator | 左臂 7 关节 |
| 7-13 | right_manipulator | 右臂 7 关节 |
| 14-19 | left_hand | 左手 6 关节 |
| 20-25 | right_hand | 右手 6 关节 |
| 26-27 | head | 头部偏航/俯仰 |
| 28 | waist | 腰部偏航 |
| 29-34 | base | 底盘速度 6D |

**State（45D）**= 29 关节 + 16 底盘状态

| 维度 | 内容 |
|------|------|
| 0-28 | 29 关节状态（同 action 顺序） |
| 29-31 | 底盘位置 (x, y, z) |
| 32-35 | 底盘四元数 (qx, qy, qz, qw) |
| 36-38 | 底盘欧拉角 (roll, pitch, yaw) |
| 39-41 | IMU 加速度 (ax, ay, az) |
| 42-44 | IMU 角速度 (wx, wy, wz) |

## 项目结构

```
.
├── scripts/convert_tools/
│   ├── convert_dora_to_lerobot.py   # 核心转换脚本
│   ├── run_convert.sh               # 快速运行脚本
│   ├── CONVERT_GUIDE.md             # 完整转换指南
│   ├── dora-record结构.md            # 输入格式详解
│   ├── lerobot-gr2结构.md            # 输出格式详解
│   ├── GR-2 各控制组参数详解.md       # 41 自由度关节参数
│   └── GR2推理适配指南.md            # 模型推理部署示例
└── CLAUDE.md
```

## 文档

详细文档位于 [scripts/convert_tools/](scripts/convert_tools/)：

- [CONVERT_GUIDE.md](scripts/convert_tools/CONVERT_GUIDE.md) — 完整转换指南、参数说明、常见问题
- [GR2推理适配指南.md](scripts/convert_tools/GR2推理适配指南.md) — 策略模型推理部署代码示例

## 依赖

- Python 3.10+
- pyarrow, numpy, pandas, Pillow
- ffmpeg（系统安装或 conda 安装）
