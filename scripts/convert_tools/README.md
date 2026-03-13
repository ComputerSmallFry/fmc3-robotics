# Convert Tools

本目录存放 Fourier Dora-Record 数据转 LeRobot 数据集的脚本和说明文档。

## 环境

```bash
conda activate lerobot
```

依赖要求：

- `ffmpeg`
- `pyarrow`
- `pandas`
- `numpy`
- `opencv-python`
- `Pillow`
- `imageio`

## 脚本说明

### 单相机 Dora -> LeRobot

脚本：

- `convert_dora_to_lerobot.py`
- `run_convert.sh`

适用数据：

- 顶视 RGB：`observation.images.camera_top.parquet`
- 顶视 depth：`observation.images.camera_top_depth.parquet`

默认输出：

- `fourier/gr2-pick-3-4_lerobot_gr2`

运行：

```bash
bash run_convert.sh
```

### 多相机 parquet -> LeRobot v3

脚本：

- `convert_parquet_to_lerobot_fourier_multicam.py`
- `run_convert_multicam_lerobotv3.sh`

适用数据：

- 顶视 RGB：`observation.images.camera_top.parquet`
- 顶视 depth：`observation.images.camera_top_depth.parquet`
- 左腕 RGB：`observation.images.camera_left_wrist.parquet`
- 左腕 depth：`observation.images.camera_left_wrist_depth.parquet`
- 右腕 RGB：`observation.images.camera_right_wrist.parquet`
- 右腕 depth：`observation.images.camera_right_wrist_depth.parquet`

默认输入：

- `/home/phl/workspace/dataset/fourier/gr2/muticams/dora/fmc3_gr2_grab_bottle_into_box_dora_ds`

默认输出：

- `/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds`

运行：

```bash
bash run_convert_multicam_lerobotv3.sh
```

## 多相机脚本参数

最常用命令：

```bash
python convert_parquet_to_lerobot_fourier_multicam.py \
  --raw-dir /home/phl/workspace/dataset/fourier/gr2/muticams/dora/fmc3_gr2_grab_bottle_into_box_dora_ds \
  --output-dir /home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds \
  --task "pick up the bottle from the grid cell and place it into the box" \
  --fps 30 \
  --camera-keys camera_top camera_left_wrist camera_right_wrist \
  --workers 4 \
  --decode-workers 4 \
  --image-writer-threads 12 \
  --vcodec h264 \
  --force
```

常用参数：

- `--raw-dir`: Dora 数据根目录
- `--output-dir`: LeRobot 输出目录
- `--task`: LeRobot 任务描述，写入 `meta/tasks.parquet`；建议显式传入，不要依赖 `metadata.json` 中的 `notes`
- `--fps`: 输出帧率，默认 `30`
- `--camera-keys`: RGB 相机列表，默认 `camera_top camera_left_wrist camera_right_wrist`
- `--no-video`: 只转 action/state，不写视频
- `--no-depth`: 只写 RGB，不写 `*_depth`
- `--workers`: `episode` 级并行进程数，行为上更接近旧版单相机脚本
- `--decode-workers`: 单路视频帧解码线程数
- `--image-writer-threads`: LeRobot 异步图片写线程数
- `--vcodec`: 输出视频编码，支持 `h264`、`hevc`、`libsvtav1`
- `--force`: 输出目录已存在时先删除

## 输出内容

转换完成后会生成标准 LeRobot v3 目录：

```text
<output_dir>/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/chunk-000/file-000.parquet
├── data/chunk-000/file-000.parquet
└── videos/
    ├── observation.images.camera_top/
    ├── observation.images.camera_top_depth/
    ├── observation.images.camera_left_wrist/
    ├── observation.images.camera_left_wrist_depth/
    ├── observation.images.camera_right_wrist/
    └── observation.images.camera_right_wrist_depth/
```

GR2 输出维度：

- `action`: 35D
- `observation.state`: 45D

## 上传到 Hugging Face

脚本：

- `upload_lerobot_dataset_to_hf.sh`

用法：

```bash
bash upload_lerobot_dataset_to_hf.sh \
  --dataset \
  /home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds
```

说明：

- 只需要传本地数据集目录
- 脚本会自动识别目录类型：
  - LeRobot：包含 `meta/`、`data/`、`videos/`
  - Dora：包含 `episode_*` 目录
- Hugging Face 仓库名默认取目录名
- 最终会上传到当前登录账号下的 `datasets/<your_username>/<dataset_dir_name>`
- 上传前先执行 `hf auth login`，或者保证环境变量里已有 `HF_TOKEN`
- 如果要建私有仓库，可在运行前加 `HF_PRIVATE=1`
- 如果只想允许上传 LeRobot，可在运行前加 `HF_REQUIRE_LEROBOT=1`
- 如果当前 shell 里设置了代理，脚本会先按当前代理环境上传；失败后会自动再试一次“无代理”
- 如果你明确知道当前环境不能走代理，可直接加 `HF_FORCE_NO_PROXY=1`

上传 Dora 原始数据示例：

```bash
bash upload_lerobot_dataset_to_hf.sh \
  --dataset \
  /home/phl/workspace/dataset/fourier/gr2/muticams/dora/fmc3_gr2_grab_bottle_into_box_dora_ds
```

如果你明确知道当前环境不能走代理，也可以强制脚本忽略代理：

```bash
HF_FORCE_NO_PROXY=1 \
  bash upload_lerobot_dataset_to_hf.sh \
    --dataset /home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds
```

## 实现说明

多相机脚本当前行为：

- 自动扫描所有 `episode_*`
- 用 `timestamp_utc` 对 action / state / RGB / depth 做最近邻对齐
- action 过滤 `waist_roll_joint`、`waist_pitch_joint`
- 按 GR2 SDK 顺序重排关节
- depth 图转换为 3 通道伪 RGB 视频，兼容当前 LeRobot 视频写入
- 支持 `episode` 级多进程并行预处理，行为上接近旧版单相机脚本
- 每个 episode 内部继续使用线程池并发解码图像
- 使用 LeRobot 异步 image writer 线程并发落盘
- 视频编码阶段使用串行 `save_episode(parallel_encoding=False)`，避免当前环境下的多进程权限错误

## 验证建议

先做一轮无视频验证：

```bash
python convert_parquet_to_lerobot_fourier_multicam.py \
  --raw-dir /home/phl/workspace/dataset/fourier/gr2/muticams/dora/fmc3_gr2_grab_bottle_into_box_dora_ds \
  --output-dir /tmp/fmc3_gr2_grab_bottle_into_box_lerobot_no_video \
  --no-video \
  --force
```

再做完整转换。

检查输出：

- `meta/info.json`
- `meta/stats.json`
- `data/chunk-000/file-000.parquet`
- `videos/.../*.mp4`

## 相关文档

- `CONVERT_GUIDE.md`
- `dora-record结构.md`
- `lerobot-gr2结构.md`
- `GR2推理适配指南.md`
