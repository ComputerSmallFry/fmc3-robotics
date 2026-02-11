#!/bin/bash

# 保存当前目录
current_dir=$(pwd)

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 执行原始命令
docker compose down

# 返回原目录
cd "$current_dir"

# 列出正在运行的容器
sleep 1s
docker ps
