# 训练脚本使用说明

## 环境要求

- conda 环境: `lerobot`
- GPU: 建议显存 >= 24GB
- AutoDL 学术加速（脚本内已自动开启）

## 策略列表

### 1. pi0 微调 - 抓取物体放入盒子

- 脚本: `scripts/train_pi0_grab_box.sh`
- 基座模型: `lerobot/pi0`（HuggingFace 自动下载）
- 数据集: `puheliang/lerobot_fmc3_grab_box_v2`（401 episodes, SO-101 机械臂, top + wrist 双相机）
- 任务: Pick up the tape and place it in the box

**首次训练:**
```bash
conda activate lerobot
bash scripts/train_pi0_grab_box.sh
```

**断点续训:**
```bash
conda activate lerobot
bash scripts/train_pi0_grab_box.sh --resume
```

**后台运行（推荐）:**
```bash
conda activate lerobot
screen -S train
bash scripts/train_pi0_grab_box.sh
# 按 Ctrl+A D 脱离 screen
# 回来查看: screen -r train
```

**主要参数说明:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| STEPS | 50000 | 总训练步数 |
| BATCH_SIZE | 8 | 批大小 |
| DTYPE | bfloat16 | 训练精度 |
| SAVE_FREQ | 20000 | checkpoint 保存频率 |
| PUSH_TO_HUB | true | 训练完成后推送到 HuggingFace |
| WANDB_ENABLE | true | 启用 WandB 日志 |

**训练输出:** `/root/output_lerobot_train/grab_box/pi0_fmc3`

**WandB 看板:** 项目名 `Lerobot_Zihao_Project`
