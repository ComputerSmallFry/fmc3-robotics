# 训练脚本说明

所有训练脚本位于 `scripts/train/` 目录下。

## 使用方法

```bash
cd ~/workspace/lerobot-versions/lerobot
bash scripts/train/<脚本名>.sh
```

如需开启 wandb 记录，将脚本中 `--wandb.enable=false` 改为 `true`，并先执行 `wandb login`。

## 脚本列表

| 脚本 | 策略 | 数据集 | 任务 | 备注 |
|------|------|--------|------|------|
| `train_act_grab_box.sh` | ACT | `puheliang/lerobot_fmc3_grab_box_v2` | 抓取胶带放入盒子 | SO-101, 401 episodes, batch_size=64, 100k steps |
