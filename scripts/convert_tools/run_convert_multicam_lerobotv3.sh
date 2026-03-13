#!/usr/bin/env bash
set -euo pipefail

# 多相机 Dora -> LeRobot v3 转换启动脚本。
#
# 当前配置对应数据集：
#   输入：/home/phl/workspace/dataset/fourier/gr2/muticams/dora/fmc3_gr2_grab_bottle_into_box_dora_ds
#   输出：/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds
#
# 使用前建议先进入 lerobot 环境：
#   source /home/phl/miniconda3/etc/profile.d/conda.sh
#   conda activate lerobot
#
# 运行方式：
#   bash run_convert_multicam_lerobotv3.sh
#
# 如需切换数据集，通常只需要改下面两个路径：
#   --raw-dir    Dora 原始数据根目录
#   --output-dir LeRobot 输出目录

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dora 原始数据根目录。脚本会递归扫描其中所有 episode_* 目录。
RAW_DIR="/home/phl/workspace/dataset/fourier/gr2/muticams/dora/fmc3_gr2_grab_bottle_into_box_dora_ds"

# 最终生成的 LeRobot 数据集目录。
# 输出结构保持为 LeRobot v3 标准形式：meta/ data/ videos/
OUTPUT_DIR="/home/phl/workspace/dataset/fourier/gr2/muticams/lerobot/fmc3_gr2_grab_bottle_into_box_lerobot_ds"

# LeRobot 数据集任务描述。
# 这个字段会写入 meta/tasks.parquet，并关联到每一帧样本。
# 建议写成训练时直接可用的英文或中英混合描述，不要留空。
TASK="pick up the bottle from the grid cell and place it into the box"

# 需要写入 LeRobot 的 RGB 相机键。
# depth 不需要单独写在这里，转换脚本会自动补对应的 *_depth 流。
CAMERA_KEYS=(
    camera_top
    camera_left_wrist
    camera_right_wrist
)

# 用数组组织参数，便于逐项加注释，也避免续行和注释互相干扰。
ARGS=(
    --raw-dir "${RAW_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --task "${TASK}"
    # 目标采样帧率。action/state/RGB/depth 都会按该频率做时间对齐。
    --fps 30
    --camera-keys "${CAMERA_KEYS[@]}"
    # episode 级并行进程数。
    # 多相机数据每个 episode 都比较重，先用 4 个进程更稳妥。
    --workers 4
    # 单个 episode 内部的图像解码线程数。
    # 它会和 --workers 叠加，占用总 CPU 线程数约为 workers * decode-workers。
    --decode-workers 4
    # LeRobot 内部异步图片写盘线程数，主要影响临时帧落盘速度。
    --image-writer-threads 12
    # 输出视频编码格式。h264 兼容性最好。
    --vcodec h264
    # 若输出目录已存在，先删除后重建，避免旧数据混入。
    --force
)

python -u "${SCRIPT_DIR}/convert_parquet_to_lerobot_fourier_multicam.py" "${ARGS[@]}"
