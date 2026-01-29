# RoboOS 启动指南

本文档包含了启动 RoboOS 系统及其相关组件的详细命令。

## 终端 1: 启动傅里叶机器人技能服务器

```bash
conda activate fourier-robot
cd /home/phl/fmc3-robotics/projects/RoboSkill/fmc3-robotics/fourier/gr2
python skill.py
```

## 终端 2: 启动 Redis (如果未运行)

```bash
redis-server
```

或检查是否已运行：
```bash
redis-cli ping
# 返回 PONG 表示正常
```

## 终端 3: 启动 RoboBrain 模型服务

```bash
conda activate robobrain
cd /home/phl/fmc3-robotics/projects/RoboBrain2.0
bash startup.sh
```

## 终端 4: 启动 RoboOS

### 1. 启动 Master
```bash
conda activate roboos
cd /home/phl/fmc3-robotics/projects/RoboOS/master
python run.py
```

### 2. 启动 Slaver
```bash
conda activate roboos
cd /home/phl/fmc3-robotics/projects/RoboOS/slaver
python run.py
```

### 3. 启动网页 (Deploy)
```bash
conda activate roboos
cd /home/phl/fmc3-robotics/projects/RoboOS/deploy
python run.py
```
