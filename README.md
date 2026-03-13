# Fourier Dora-Record -> LeRobot v3.0 转换说明

本仓库用于把 Fourier 机器人遥操作采集的 Dora-Record 数据，转换成 LeRobot v3.0 训练格式（PI0/ACT/Diffusion/SmolVLA 等可直接读取）。

## Dora 原始格式（输入）

### 目录结构

```text
pick_bottle_and_place_into_box/
├── <session_id_1>/
│   ├── episode_000000000/
│   │   ├── metadata.json
│   │   ├── action.parquet
│   │   ├── action.base.parquet
│   │   ├── observation.state.parquet
│   │   ├── observation.base_state.parquet
│   │   ├── observation.images.camera_top.parquet
│   │   └── observation.images.camera_top_depth.parquet
│   └── episode_000000001/
└── <session_id_2>/
    └── episode_...
```

### 各 parquet 的语义

| 文件 | 关键列 | 形状/维度 | 频率（约） | 说明 |
|---|---|---:|---:|---|
| `action.parquet` | `action` | 31D | 100Hz | 关节动作（`list<struct<name,value>>`） |
| `action.base.parquet` | `action.base` | 6D | 100Hz | 底盘动作（`vel_x/vel_y/vel_yaw/vel_height/vel_pitch/base_yaw`） |
| `observation.state.parquet` | `observation.state` | 29D | 60Hz | 关节状态（`list<struct<name,value>>`） |
| `observation.base_state.parquet` | `observation.base_state` | 16D（展开后） | 60Hz | 底盘位姿 + IMU 嵌套结构 |
| `observation.images.camera_top.parquet` | `observation.images.camera_top` | JPEG 字节流 | 30Hz | RGB 图像 |
| `observation.images.camera_top_depth.parquet` | `observation.images.camera_top_depth` | PNG 字节流 | 30Hz | 深度图像 |

> 公共时间列为 `timestamp_utc`（纳秒时间戳），转换时用它做跨模态对齐。

## LeRobot 格式（输出）

### 目录结构

```text
<dataset_name>/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-000.parquet
├── data/chunk-000/file-000.parquet
└── videos/
    ├── observation.images.camera_top/chunk-000/file-*.mp4
    └── observation.images.camera_top_depth/chunk-000/file-*.mp4
```

### 帧级数据（`data/chunk-000/file-000.parquet`）

| 列名 | 类型 | 含义 |
|---|---|---|
| `action` | `list<float64>` | 模型监督动作向量 |
| `observation.state` | `list<float64>` | 模型状态输入向量 |
| `timestamp` | `float64` | episode 内时间（秒） |
| `frame_index` | `int64` | episode 内帧号 |
| `episode_index` | `int64` | episode 编号 |
| `index` | `int64` | 全局帧号 |
| `task_index` | `int64` | 任务索引（映射到 `meta/tasks.parquet`） |

### 输出维度（按当前脚本）

- `--robot-type fourier_gr2`（你当前 `run_convert.sh` 使用的配置）
  - `action`: 35D = 29 关节 + 6 底盘动作
  - `observation.state`: 45D = 29 关节 + 16 底盘状态
- `--robot-type fourier_gr3`
  - `action`: 37D = 31 关节 + 6 底盘动作
  - `observation.state`: 45D = 29 关节 + 16 底盘状态

## 转换时具体做了什么

1. 自动扫描所有 `session/episode_*`。
2. 读取各模态 parquet，并取 `timestamp_utc`。
3. 生成统一时间轴（默认 `--fps 30`），对动作/状态/图像做最近邻重采样。
4. `fourier_gr2` 下过滤 `waist_roll_joint` 和 `waist_pitch_joint`，并按 GR2 SDK 控制组顺序重排关节。
5. 拼接：
   - `action = joint_action + base_action`
   - `observation.state = joint_state + base_state`
6. RGB/Depth 重新编码成 mp4，并写入 LeRobot `meta/*` + `data/*`。

## 快速运行

```bash
# 环境要求: pyarrow numpy pandas Pillow ffmpeg
conda activate lerobot

# 预设命令（GR2）
bash scripts/convert_tools/run_convert.sh

# 手动运行
python scripts/convert_tools/convert_dora_to_lerobot.py \
  --input ./pick_bottle_and_place_into_box \
  --output ./pick_bottle_and_place_into_box_lerobot_gr2 \
  --task "pick bottle and place into box" \
  --fps 30 \
  --robot-type fourier_gr2 \
  --video-codec libopenh264 \
  --workers 4
```

> conda 里的 ffmpeg 常不带 `libx264`，建议 `--video-codec libopenh264`。

## 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--input`/`-i` | 必填 | Dora 数据根目录（可包含 session 子目录） |
| `--output`/`-o` | 必填 | LeRobot 输出目录 |
| `--task`/`-t` | `teleoperation` | 任务文本 |
| `--fps` | `30` | 输出帧率 |
| `--video-codec` | `libx264` | 视频编码器 |
| `--robot-type` | `fourier_gr3` | 机器人类型 |
| `--workers`/`-w` | `1` | 并行进程数 |
| `--no-video` | 关闭 | 不处理图像/视频 |

## 相关文档

- `scripts/convert_tools/dora-record结构.md`
- `scripts/convert_tools/lerobot-gr2结构.md`
- `scripts/convert_tools/CONVERT_GUIDE.md`
