# Fourier GR2 LeRobot 训练指南

## 环境准备

```bash
conda activate lerobot
cd /home/phl/workspace/lerobot-versions/lerobot
pip install -e ".[dev,test]"
pip install num2words  # SmolVLA 需要
```

## 数据集

| 数据集 | 路径 | 格式 | episodes | frames |
|--------|------|------|----------|--------|
| pick_and_place | `/home/phl/workspace/dataset/fourier/pick_and_place` | v3.0 | 27 | 5922 |
| lerobot_output | `/home/phl/workspace/dataset/fourier/lerobot_output` | v3.0 | 3 | 835 |

数据集特征：
- **robot_type**: fourier_gr2
- **action**: 37 维（双臂14关节 + 双手12关节 + 头2 + 腰3 + 底盘6）
- **observation.state**: 45 维（关节 + 底盘位姿 + IMU）
- **observation.images.camera_top**: 480x640 RGB 视频
- **observation.images.camera_top_depth**: 480x640 深度视频

## 本地模型权重

| 模型 | 路径 |
|------|------|
| SmolVLM2-500M | `/home/phl/workspace/models/SmolVLM2-500M-Video-Instruct` |
| PI0 | `/home/phl/workspace/models/pi0` |

---

## 训练命令

### 1. ACT 策略

轻量经典策略，训练快，适合快速验证。

```bash
# 脚本: train_fourier_gr2.sh
lerobot-train \
    --policy.type=act \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --dataset.repo_id=fourier_gr2_lerobot \
    --dataset.root=/home/phl/workspace/dataset/fourier/lerobot_output \
    --dataset.video_backend=torchcodec \
    --output_dir=outputs/train/fourier_gr2_act \
    --batch_size=64 \
    --steps=50000 \
    --save_freq=5000 \
    --log_freq=100 \
    --num_workers=8
```

### 2. SmolVLA 策略

基于 SmolVLM2-500M 的 VLA 模型，默认冻结视觉编码器，只训练 action expert。

> **注意**: GR2 的 state=45、action=37 超过默认的 `max_state_dim=32` 和 `max_action_dim=32`，必须手动指定。

```bash
# 脚本: train_fourier_gr2_smolvla.sh
lerobot-train \
    --policy.type=smolvla \
    --policy.vlm_model_name=/home/phl/workspace/models/SmolVLM2-500M-Video-Instruct \
    --policy.max_state_dim=45 \
    --policy.max_action_dim=37 \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --dataset.repo_id=fourier_gr2_pick_place \
    --dataset.root=/home/phl/workspace/dataset/fourier/pick_and_place \
    --dataset.video_backend=torchcodec \
    --output_dir=outputs/train/fourier_gr2_smolvla \
    --batch_size=16 \
    --steps=30000 \
    --save_freq=5000 \
    --log_freq=100 \
    --num_workers=4
```

### 3. PI0 策略

基于 PaliGemma-2B 的 Flow Matching 模型，从预训练权重微调。

> **注意**: `--policy.path` 和 `--policy.type` 不能同时使用，用 `--policy.path` 会自动识别类型。

```bash
# 脚本: train_fourier_gr2_pi0.sh
lerobot-train \
    --policy.path=/home/phl/workspace/models/pi0 \
    --policy.use_amp=true \
    --policy.push_to_hub=false \
    --policy.gradient_checkpointing=true \
    --dataset.repo_id=fourier_gr2_pick_place \
    --dataset.root=/home/phl/workspace/dataset/fourier/pick_and_place \
    --dataset.video_backend=torchcodec \
    --output_dir=outputs/train/fourier_gr2_pi0 \
    --batch_size=4 \
    --steps=30000 \
    --save_freq=5000 \
    --log_freq=100 \
    --num_workers=4
```

### 4. 同时训练 SmolVLA + PI0

两个模型共享 4090 48GB 显存（SmolVLA ~15GB + PI0 ~25GB）。

```bash
bash train_both.sh
```

启动后常用命令：

```bash
nvidia-smi                                  # 查看显存
tail -f outputs/train/logs/smolvla_*.log    # SmolVLA 日志
tail -f outputs/train/logs/pi0_*.log        # PI0 日志
```

---

## 恢复训练

如果训练中断，可以从 checkpoint 恢复：

```bash
lerobot-train \
    --resume=true \
    --config_path=outputs/train/fourier_gr2_smolvla/checkpoints/last/pretrained_model/train_config.json
```

---

## 4090 48GB 优化说明

所有脚本已针对魔改 48GB 显存优化，主要调整：

| 策略 | 原 batch_size | 优化后 | 原 num_workers | 优化后 | 额外优化 |
|------|--------------|--------|----------------|--------|---------|
| ACT | 64 | **256** | 8 | **12** | — |
| SmolVLA | 16 | **48** | 4 | **8** | — |
| PI0 | 4 | **12** | 4 | **8** | `dtype=bfloat16`, `compile_model=true` |
| 并行 SmolVLA | 16 | **32** | 4 | 4 | — |
| 并行 PI0 | 4 | **8** | 4 | 4 | `dtype=bfloat16` |

关键优化项：
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**: 减少显存碎片，提高大 batch 训练稳定性
- **`dtype=bfloat16`** (PI0): 模型权重用 BF16 存储，显存减半，4090 原生支持 BF16
- **`compile_model=true`** (PI0 单独训练): torch.compile 加速，首次编译较慢但后续迭代更快
- **更大 batch_size**: 充分利用 48GB 显存，提高 GPU 利用率和训练吞吐
- **更多 num_workers**: 加速数据加载，减少 GPU 等待时间

> **提示**: 如果遇到 OOM，先降低 batch_size（每次减半尝试）。并行训练时两个进程共享显存，已留 ~3GB 余量。

---

## 常见问题

| 问题 | 解决方案 |
|------|---------|
| `ImportError: num2words` | `pip install num2words` |
| `Cannot specify both --policy.path and --policy.type` | 二选一，`path` 会自动识别类型 |
| `max_state_dim / max_action_dim 不够` | SmolVLA/PI0 需手动设置 `--policy.max_state_dim=45 --policy.max_action_dim=37` |
| OOM 显存不足 | 降低 `batch_size`，或开启 `--policy.gradient_checkpointing=true` |
| 训练慢 | 开启 `--policy.use_amp=true`，增加 `--num_workers` |
| torch.compile 报错 | 去掉 `--policy.compile_model=true`，不影响训练结果 |
