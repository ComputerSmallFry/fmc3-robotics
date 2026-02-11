#!/bin/bash

# 检查adb命令是否存在
if ! command -v adb &> /dev/null; then
    echo "错误: adb命令未找到，请先安装adb。"
    echo "Ubuntu/Debian系统请执行: sudo apt install adb -y"
    echo "CentOS/RHEL系统请执行: sudo dnf install adb -y 或 sudo yum install adb -y"
    exit 1
fi

# 执行adb反向映射命令
adb reverse tcp:8080  tcp:8080 # web server
adb reverse tcp:7880  tcp:7880 # livekit server
