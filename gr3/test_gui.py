#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本，用于验证gr3_gui.py的基本结构是否正确
"""

import sys
import os

# 检查gr3_gui.py文件是否存在
gui_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gr3_gui.py")
if not os.path.exists(gui_script_path):
    print("错误: gr3_gui.py 文件不存在")
    sys.exit(1)

# 检查Python版本
print(f"Python版本: {sys.version}")

# 检查必要的依赖是否已安装
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, filedialog, messagebox
    print("✓ tkinter 模块已安装")
except ImportError:
    print("✗ 错误: 未安装tkinter模块，请使用以下命令安装:")
    print("  sudo apt-get install python3-tk")

try:
    import subprocess
    print("✓ subprocess 模块已安装")
except ImportError:
    print("✗ 错误: 未安装subprocess模块")

try:
    import threading
    print("✓ threading 模块已安装")
except ImportError:
    print("✗ 错误: 未安装threading模块")

try:
    import queue
    print("✓ queue 模块已安装")
except ImportError:
    print("✗ 错误: 未安装queue模块")

# 检查脚本是否有执行权限
if os.access(gui_script_path, os.X_OK):
    print("✓ gr3_gui.py 具有执行权限")
else:
    print("✗ gr3_gui.py 没有执行权限，建议运行:")
    print(f"  chmod +x {gui_script_path}")

# 检查run_gr3.sh脚本是否存在
run_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_gr3.sh")
if os.path.exists(run_script_path):
    print("✓ run_gr3.sh 脚本已找到")
else:
    print(f"✗ 警告: 未找到run_gr3.sh脚本，路径: {run_script_path}")

print("\n测试完成。要启动GUI程序，请运行:")
print(f"  python3 {gui_script_path}")
print("或")
print(f"  ./{os.path.basename(gui_script_path)}")
print("\n更新内容:")
print("  - 添加了'显示帮助'按钮，可以查看run_gr3.sh脚本的帮助信息")
print("  - 帮助信息会显示在输出日志区域")
print("  - 当容器运行时，会提示用户先停止容器再查看帮助")
print("  - 环境变量文件浏览时现在始终存储绝对路径")
print("  - 添加了对ANSI颜色转义序列的支持，可以正确显示终端颜色输出（如\x1b[2m等）")
print("  - 增加了日志页面背景颜色选择选项（白色/黑色）")
print("  - 修复了停止容器功能，现在即使GUI状态不正确也能通过Docker命令正确停止容器")
print("  - 增加了发送Ctrl+C信号的功能，更有效地停止容器进程")
print("\n注意: GUI程序需要图形界面环境才能运行")