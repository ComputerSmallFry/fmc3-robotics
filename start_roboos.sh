#!/bin/bash

# 设置工作空间
WORKSPACE="$HOME/workspace/fmc3-robotics/projects"
SESSION="robo_vscode"

# 检查是否已存在，存在则杀掉重建（防止残留）
tmux kill-session -t $SESSION 2>/dev/null

# 创建新会话 - 窗口 1: Skill
tmux new-session -d -s $SESSION -n "Skill"
tmux send-keys -t $SESSION:Skill "conda activate fourier-robot" C-m
tmux send-keys -t $SESSION:Skill "cd $WORKSPACE/RoboSkill/fmc3-robotics/fourier/gr2 && python skill.py" C-m

# 切分屏幕运行 Brain
tmux split-window -h -t $SESSION:Skill
tmux send-keys -t $SESSION:Skill "conda activate robobrain" C-m
tmux send-keys -t $SESSION:Skill "cd $WORKSPACE/RoboBrain2.0 && bash startup.sh" C-m

# 创建窗口 2: RoboOS
tmux new-window -t $SESSION -n "RoboOS"
# 上半部分: Master
tmux send-keys -t $SESSION:RoboOS "conda activate roboos" C-m
tmux send-keys -t $SESSION:RoboOS "cd $WORKSPACE/RoboOS/master && python run.py" C-m
# 切分下半部分
tmux split-window -v -t $SESSION:RoboOS
# 左下: Slaver
tmux send-keys -t $SESSION:RoboOS "conda activate roboos" C-m
tmux send-keys -t $SESSION:RoboOS "cd $WORKSPACE/RoboOS/slaver && python run.py" C-m
# 切分右下: Deploy
tmux split-window -h -t $SESSION:RoboOS
tmux send-keys -t $SESSION:RoboOS "conda activate roboos" C-m
tmux send-keys -t $SESSION:RoboOS "cd $WORKSPACE/RoboOS/deploy && python run.py" C-m

# 挂载到当前终端
tmux attach-session -t $SESSION