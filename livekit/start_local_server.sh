#!/bin/bash

# 保存当前目录
current_dir=$(pwd)

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 登录镜像仓库
docker login -u devops -p fftai@2025 docker.fftaicorp.com

# 执行原始命令
docker compose up -d --remove-orphans

# 返回原目录
cd "$current_dir"

# 列出正在运行的容器
sleep 1s
docker ps
