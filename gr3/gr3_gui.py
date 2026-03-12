#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GR3 GUI 程序
用于调用 run_gr3.sh 脚本并展示其输出

uv venv -p 3.13 --seed
uv run python gr3/gr3_gui.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import signal
import subprocess
import threading
import time
import queue
import stat

# 中英文翻译字典
translations = {   # 标题和框架
    "GRx 遥操作启动器": "GRx Teleoperation Launcher",
    "配置参数": "Configuration Parameters",
    "输出日志": "Output Log",
    
    # 标签文本
    "容器镜像:": "Docker Image:",
    "容器标签:": "Or Use Tag:",
    "遥操设备:": "Graph File:",
    "环境变量:": "Environment File:",
    "DAQ_NOTES:": "DAQ_NOTES:",
    "DAQ_PILOT:": "DAQ_PILOT:",
    "DAQ_OPERATOR:": "DAQ_OPERATOR:",
    "STATION_ID:": "STATION_ID:",
    "MACHINE_ID:": "MACHINE_ID:",
    "DOMAIN_ID:": "DOMAIN_ID:",
    "日志背景颜色: ": "Log Background Color: ",
    "浏览...": "Browse...",
    
    # 按钮文本
    "启动数采容器": "Start DAQ Container",
    "停止数采容器": "Stop DAQ Container",
    "显示帮助文档": "Show Help",
    "清空日志输出": "Clear Output",
    "退出": "Exit",
    "启动本地服务": "Start Local Server",
    "停止本地服务": "Stop Local Server",
    "连接本地头显": "Connect Local Headset",
    "更新数采镜像": "Update DAQ Image",
    "查看镜像状态": "View Docker Status",

    # 复选框和单选按钮
    "调试模式 (只显示命令不执行)": "Debug Mode (Show command only)",
    "本地模式": "Local Mode",
    "黑色": "Black",
    "白色": "White",
    
    # 消息框文本
    "警告": "Warning",
    "容器已经在运行中": "Container is already running",
    "信息": "Information",
    "错误": "Error",
    "找不到脚本文件: ": "Cannot find script file: ",
    "没有找到正在运行的相关容器": "No running containers found",
    "容器正在运行中, 请先停止容器后再查看帮助": "Container is running, please stop it before viewing help",
    "退出": "Exit",
    "容器正在运行中, 确定要退出吗？": "Container is running, are you sure you want to exit?",
    
    # 输出文本
    "执行命令: ": "Executing command: ",
    "命令执行成功完成": "Command executed successfully",
    "命令执行失败, 退出码: ": "Command execution failed, exit code: ",
    "执行命令时出错: ": "Error executing command: ",
    "正在停止容器进程...": "Stopping container process...",
    "发送Ctrl+C信号...": "Sending Ctrl+C signal...",
    "进程响应Ctrl+C信号并退出": "Process responded to Ctrl+C and exited",
    "进程未响应Ctrl+C, 尝试正常终止...": "Process not responding to Ctrl+C, trying to terminate normally...",
    "进程已正常终止": "Process terminated normally",
    "强制终止进程...": "Forcibly terminating process...",
    "停止进程时出错: ": "Error stopping process: ",
    "正在检查并停止Docker容器...": "Checking and stopping Docker containers...",
    "找到相关容器: ": "Found relevant containers: ",
    "所有相关容器已成功停止": "All relevant containers stopped successfully",
    "停止容器时遇到问题": "Problems encountered while stopping containers",
    "容器停止操作完成": "Container stop operation completed",
    "正在获取帮助信息...": "Getting help information...",
    "帮助信息获取成功": "Help information retrieved successfully",
    "获取帮助信息失败, 退出码: ": "Failed to retrieve help information, exit code: ",
    "获取帮助信息时出错: ": "Error retrieving help information: ",
    "语言 / Language: ": "Language / 语言: ",
    "正在启动本地服务...": "Starting local server...",
        "正在停止本地服务...": "Stopping local server...",
        "正在更新数采镜像...": "Updating DAQ image...",
        "正在查看镜像状态...": "Viewing docker status...",
        "正在连接本地头显...": "Connecting to local headset..."
}


class GR3GUI:
    def __init__(self, root):
        """初始化GUI界面"""
        self.root = root
        # 语言状态变量：'zh' 表示中文, 'en' 表示英文
        self.current_language = 'zh'
        # 设置标题（使用翻译方法）
        self.root.title(self._("GRx 遥操作启动器"))
        self.root.geometry("900x1000")
        self.root.resizable(True, True)
        
        # 设置中文字体支持
        self.style = ttk.Style()
        self.style.configure("TLabel", font=('SimHei', 10))
        self.style.configure("TButton", font=('SimHei', 10))
        self.style.configure("TEntry", font=('SimHei', 10))
        self.style.configure("TCheckbutton", font=('SimHei', 10))
        self.style.configure("TRadiobutton", font=('SimHei', 10))
        
        # 脚本路径
        self.script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_gr3.sh")
        
        # 检查脚本是否存在
        if not os.path.exists(self.script_path):
            messagebox.showerror(self._("错误"), f"{self._('找不到脚本文件: ')}{self.script_path}")
            self.root.destroy()
            return
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建语言切换区域
        self.create_language_frame()
        
        # 创建参数输入区域
        self.create_params_frame()
        
        # 创建按钮区域
        self.create_buttons_frame()
        
        # 创建输出展示区域
        self.create_output_frame()
        
        # 线程安全的队列用于更新UI
        self.output_queue = queue.Queue()
        self.process = None
        self.running = False
        
        # 启动输出更新循环
        self.update_output()
        
    def _(self, text):
        """翻译函数, 根据当前语言返回对应的文本
        
        Args:
            text: 原始文本（中文）
            
        Returns:
            翻译后的文本（英文或保持中文）
        """
        if self.current_language == 'en' and text in translations:
            return translations[text]
        return text
    
    def create_language_frame(self):
        """创建语言切换框架"""
        # 创建语言选择框架
        language_frame = ttk.Frame(self.main_frame, padding="5")
        language_frame.pack(fill=tk.X, pady=5)
        
        # 添加语言选择标签和下拉菜单
        self.language_label = ttk.Label(language_frame, text=self._("语言 / Language: "))
        self.language_label.pack(side=tk.LEFT, padx=5)
        
        # 语言选项变量
        self.language_var = tk.StringVar(value="中文")
        
        # 创建语言选择下拉菜单
        self.language_combo = ttk.Combobox(language_frame, textvariable=self.language_var, 
                                         values=["中文", "English"], width=10)
        self.language_combo.pack(side=tk.LEFT, padx=5)
        
        # 绑定语言选择事件
        self.language_combo.bind("<<ComboboxSelected>>", self.on_language_selected)
    
    def on_language_selected(self, event=None):
        """当语言选择发生变化时的处理函数"""
        selected_language = self.language_var.get()
        if selected_language == "English":
            self.switch_language('en')
        else:
            # 当不是English时, 统一切换到中文
            self.switch_language('zh')
    
    def switch_language(self, language):
        """切换语言
        
        Args:
            language: 目标语言, 'zh' 表示中文, 'en' 表示英文
        """
        self.current_language = language
        # 更新窗口标题
        self.root.title(self._("GRx 遥操作启动器"))
        # 更新语言选项显示
        if hasattr(self, 'language_combo'):
            if language == 'en':
                self.language_combo.config(values=["Chinese", "English"])
                if self.language_var.get() == "中文":
                    self.language_var.set("Chinese")
            else:
                self.language_combo.config(values=["中文", "English"])
                if self.language_var.get() == "Chinese":
                    self.language_var.set("中文")
        # 更新所有UI元素的文本
        self.update_ui_text()
    
    def update_ui_text(self):
        """更新UI所有文本为当前语言"""
        # 更新窗口标题
        self.root.title(self._("GRx 遥操作启动器"))
        
        # 更新参数框架标题
        if hasattr(self, 'params_frame'):
            self.params_frame.config(text=self._("配置参数"))
        
        # 更新标签
        if hasattr(self, 'docker_image_label'):
            self.docker_image_label.config(text=self._("容器镜像:"))
        if hasattr(self, 'or_label'):
            self.or_label.config(text=self._("容器标签:"))
        if hasattr(self, 'graph_file_label'):
            self.graph_file_label.config(text=self._("遥操设备:"))
        if hasattr(self, 'env_file_label'):
            self.env_file_label.config(text=self._("环境变量:"))
        if hasattr(self, 'log_color_label'):
            self.log_color_label.config(text=self._("日志背景颜色: "))
        # 更新环境变量标签
        if hasattr(self, 'notes_label'):
            self.notes_label.config(text=self._("DAQ_NOTES:"))
        if hasattr(self, 'pilot_label'):
            self.pilot_label.config(text=self._("DAQ_PILOT:"))
        if hasattr(self, 'operator_label'):
            self.operator_label.config(text=self._("DAQ_OPERATOR:"))
        if hasattr(self, 'station_id_label'):
            self.station_id_label.config(text=self._("STATION_ID:"))
        if hasattr(self, 'machine_id_label'):
            self.machine_id_label.config(text=self._("MACHINE_ID:"))
        if hasattr(self, 'domain_id_label'):
            self.domain_id_label.config(text=self._("DOMAIN_ID:"))
        
        # 更新语言框架文本
        if hasattr(self, 'language_label'):
            self.language_label.config(text=self._("语言 / Language: "))
        
        # 更新按钮
        if hasattr(self, 'browse_graph_button'):
            self.browse_graph_button.config(text=self._("浏览..."))
        if hasattr(self, 'browse_env_button'):
            self.browse_env_button.config(text=self._("浏览..."))
        if hasattr(self, 'start_button'):
            self.start_button.config(text=self._("启动数采容器"))
        if hasattr(self, 'stop_button'):
            self.stop_button.config(text=self._("停止数采容器"))
        if hasattr(self, 'help_button'):
            self.help_button.config(text=self._("显示帮助文档"))
        if hasattr(self, 'clear_button'):
            self.clear_button.config(text=self._("清空日志输出"))
        if hasattr(self, 'exit_button'):
            self.exit_button.config(text=self._("退出"))
        if hasattr(self, 'start_local_server_button'):
            self.start_local_server_button.config(text=self._("启动本地服务"))
        if hasattr(self, 'stop_local_server_button'):
            self.stop_local_server_button.config(text=self._("停止本地服务"))
        if hasattr(self, 'connect_headset_button'):
            self.connect_headset_button.config(text=self._("连接本地头显"))
        if hasattr(self, 'update_docker_button'):
            self.update_docker_button.config(text=self._("更新数采镜像"))
        if hasattr(self, 'show_docker_button'):
            self.show_docker_button.config(text=self._("查看镜像状态"))

        # 更新复选框和单选按钮
        if hasattr(self, 'debug_checkbox'):
            self.debug_checkbox.config(text=self._("调试模式 (只显示命令不执行)"))
        if hasattr(self, 'local_mode_checkbox'):
            self.local_mode_checkbox.config(text=self._("本地模式"))
        if hasattr(self, 'black_radio'):
            self.black_radio.config(text=self._("黑色"))
        if hasattr(self, 'white_radio'):
            self.white_radio.config(text=self._("白色"))
        
        # 更新输出框架标题
        if hasattr(self, 'output_frame'):
            self.output_frame.config(text=self._("输出日志"))
    
    def create_params_frame(self):
        """创建参数输入框架"""
        params_frame = ttk.LabelFrame(self.main_frame, text=self._("配置参数"), padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        
        # 创建一个画布和滚动条, 以支持参数区域的滚动
        canvas = tk.Canvas(params_frame, height=350)  # 设置默认高度为300像素
        scrollbar = ttk.Scrollbar(params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 参数变量
        self.docker_var = tk.StringVar(value="docker.fftaicorp.com/farts/farther:onekey-260310114304-c55c76c9")
        self.tag_var = tk.StringVar(value="")
        
        # 预设的graph选项
        self.graph_options = [
            "agv.yml",
            "agv_gr2.yml",
            "agv_opencv.yml",
            "agv_gr2_opencv.yml",
            "daq_t5d.yml",
            "daq_t5d_opencv.yml",
            "exo-debug.yml",
            "daq_t5d_orbbec.yml",
            "quest_t5d_orbbec.yml",
            "quest_t5d_multicam.yml",
            "daq_t5d_multicam.yml",
        ]
        self.graph_var = tk.StringVar(value=self.graph_options[0])
        
        self.notes_var = tk.StringVar(value="grxtest")
        self.pilot_var = tk.StringVar(value="-1")
        self.operator_var = tk.StringVar(value="-1")
        self.station_id_var = tk.StringVar(value=os.popen("hostname | sed 's/[^0-9]//g'").read().strip() or "-1")
        self.machine_id_var = tk.StringVar(value="GRx")
        self.domain_id_var = tk.StringVar(value="123")
        self.env_file_var = tk.StringVar(value="")
        self.debug_var = tk.BooleanVar(value=False)
        self.local_mode_var = tk.BooleanVar(value=False)
        # 日志背景颜色选项, 默认为黑色
        self.log_bg_var = tk.StringVar(value="black")
        
        # 创建网格布局的参数输入
        row = 0
        
        # Docker镜像或标签
        self.docker_image_label = ttk.Label(scrollable_frame, text=self._("容器镜像:"))
        self.docker_image_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.docker_var, width=60).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        self.or_label = ttk.Label(scrollable_frame, text=self._("容器标签:"))
        self.or_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.tag_var, width=60).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # Graph选择
        self.graph_file_label = ttk.Label(scrollable_frame, text=self._("遥操设备:"))
        self.graph_file_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        graph_combo = ttk.Combobox(scrollable_frame, textvariable=self.graph_var, values=self.graph_options, width=57)
        graph_combo.grid(row=row, column=1, sticky=tk.W, pady=2)
        graph_combo.bind("<<ComboboxSelected>>", self.on_graph_selected)
        row += 1
        
        # 环境变量文件
        self.env_file_label = ttk.Label(scrollable_frame, text=self._("环境变量:"))
        self.env_file_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        env_frame = ttk.Frame(scrollable_frame)
        env_frame.grid(row=row, column=1, sticky=tk.W, pady=2)
        ttk.Entry(env_frame, textvariable=self.env_file_var, width=50).pack(side=tk.LEFT)
        self.browse_env_button = ttk.Button(env_frame, text=self._("浏览..."), command=self.browse_env_file)
        self.browse_env_button.pack(side=tk.LEFT, padx=2)
        row += 1
        
        # DAQ_NOTES
        self.notes_label = ttk.Label(scrollable_frame, text=self._("DAQ_NOTES:"))
        self.notes_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.notes_var, width=60).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # DAQ_PILOT
        self.pilot_label = ttk.Label(scrollable_frame, text=self._("DAQ_PILOT:"))
        self.pilot_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.pilot_var, width=60).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # DAQ_OPERATOR
        self.operator_label = ttk.Label(scrollable_frame, text=self._("DAQ_OPERATOR:"))
        self.operator_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.operator_var, width=60).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # STATION_ID
        self.station_id_label = ttk.Label(scrollable_frame, text=self._("STATION_ID:"))
        self.station_id_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.station_id_var, width=60).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # MACHINE_ID
        self.machine_id_label = ttk.Label(scrollable_frame, text=self._("MACHINE_ID:"))
        self.machine_id_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.machine_id_var, width=60).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # DOMAIN_ID
        self.domain_id_label = ttk.Label(scrollable_frame, text=self._("DOMAIN_ID:"))
        self.domain_id_label.grid(row=row, column=0, sticky=tk.W, pady=2)
        ttk.Entry(scrollable_frame, textvariable=self.domain_id_var, width=60).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1
        
        # 复选框选项
        checkbox_frame = ttk.Frame(scrollable_frame)
        checkbox_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        self.debug_checkbox = ttk.Checkbutton(checkbox_frame, text=self._("调试模式 (只显示命令不执行)"), variable=self.debug_var)
        self.debug_checkbox.pack(side=tk.LEFT, padx=10)
        self.local_mode_checkbox = ttk.Checkbutton(checkbox_frame, text=self._("本地模式"), variable=self.local_mode_var)
        self.local_mode_checkbox.pack(side=tk.LEFT, padx=10)
        row += 1
        
        # 日志背景颜色选择
        bg_frame = ttk.Frame(scrollable_frame)
        bg_frame.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        self.log_color_label = ttk.Label(bg_frame, text=self._("日志背景颜色: "))
        self.log_color_label.pack(side=tk.LEFT, padx=5)
        self.black_radio = ttk.Radiobutton(bg_frame, text=self._("黑色"), variable=self.log_bg_var, value="black", command=self.change_log_background)
        self.black_radio.pack(side=tk.LEFT, padx=5)
        self.white_radio = ttk.Radiobutton(bg_frame, text=self._("白色"), variable=self.log_bg_var, value="white", command=self.change_log_background)
        self.white_radio.pack(side=tk.LEFT, padx=5)
        row += 1
        
        # 保存引用以便后续更新
        self.params_frame = params_frame
    
    def on_graph_selected(self, event=None):
        """当选择不同的graph文件时, 可能需要更新环境变量文件"""
        selected_graph = self.graph_var.get()
        if "t5d" in selected_graph and os.path.exists("t5d.env"):
            if not self.env_file_var.get():
                self.env_file_var.set("t5d.env")
        elif "agv" in selected_graph and os.path.exists("agv.env"):
            if not self.env_file_var.get():
                self.env_file_var.set("agv.env")
    
    def browse_env_file(self):
        """浏览环境变量文件"""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="选择环境变量文件",
            filetypes=[("环境文件", "*.env"), ("所有文件", "*")],
            initialdir=os.path.dirname(os.path.abspath(__file__))
        )
        if file_path:
            # 始终存储绝对路径
            self.env_file_var.set(os.path.abspath(file_path))
    
    def create_buttons_frame(self):
        """创建按钮框架"""
        buttons_frame = ttk.Frame(self.main_frame, padding="5")
        buttons_frame.pack(fill=tk.X, pady=5)
        
        # 第一行按钮 - 核心功能
        row1_frame = ttk.Frame(buttons_frame)
        row1_frame.pack(fill=tk.X, pady=5)
        
        # 左对齐的按钮
        left_buttons = ttk.Frame(row1_frame)
        left_buttons.pack(side=tk.LEFT)
        
        # 创建核心功能按钮并保存引用
        self.start_button = ttk.Button(left_buttons, text=self._("启动数采容器"), command=self.start_container)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(left_buttons, text=self._("停止数采容器"), command=self.stop_container)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.help_button = ttk.Button(left_buttons, text=self._("显示帮助文档"), command=self.show_help)
        self.help_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(left_buttons, text=self._("清空日志输出"), command=self.clear_output)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # 右对齐的按钮
        right_buttons = ttk.Frame(row1_frame)
        right_buttons.pack(side=tk.RIGHT)
        
        self.exit_button = ttk.Button(right_buttons, text=self._("退出"), command=self.root.destroy)
        self.exit_button.pack(padx=5)
        
        # 第二行按钮 - 新增功能
        row2_frame = ttk.Frame(buttons_frame)
        row2_frame.pack(fill=tk.X, pady=5)
        
        # 创建新增功能按钮并保存引用
        self.start_local_server_button = ttk.Button(row2_frame, text=self._("启动本地服务"), command=self.start_local_server)
        self.start_local_server_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_local_server_button = ttk.Button(row2_frame, text=self._("停止本地服务"), command=self.stop_local_server)
        self.stop_local_server_button.pack(side=tk.LEFT, padx=5)

        self.connect_headset_button = ttk.Button(row2_frame, text=self._("连接本地头显"), command=self.connect_local_headset)
        self.connect_headset_button.pack(side=tk.LEFT, padx=5)

        self.update_docker_button = ttk.Button(row2_frame, text=self._("更新数采镜像"), command=self.update_docker_image)
        self.update_docker_button.pack(side=tk.LEFT, padx=5)
        
        self.show_docker_button = ttk.Button(row2_frame, text=self._("查看镜像状态"), command=self.show_docker_status)
        self.show_docker_button.pack(side=tk.LEFT, padx=5)
    
    def create_output_frame(self):
        """创建输出展示框架"""
        output_frame = ttk.LabelFrame(self.main_frame, text=self._("输出日志"), padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建带滚动条的文本区域
        # 根据选择设置初始背景和前景色
        bg_color = "black" if self.log_bg_var.get() == "black" else "white"
        fg_color = "white" if self.log_bg_var.get() == "black" else "black"
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=('SimHei', 10),
                                                   bg=bg_color, fg=fg_color)
        self.output_text.pack(fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)
        
        # 保存引用以便后续更新
        self.output_frame = output_frame
    
    def append_output(self, text):
        """向输出区域添加文本（线程安全）"""
        self.output_queue.put(text)
    
    def parse_ansi_color(self, text):
        """解析ANSI颜色转义序列并应用到文本组件"""
        # 初始样式
        current_style = {}
        i = 0
        tag_count = 0
        
        # 获取当前背景色, 用于调整颜色显示
        is_dark_bg = self.log_bg_var.get() == "black"
        
        while i < len(text):
            # 查找ANSI转义序列的开始
            if text[i] == '\x1b' and i + 1 < len(text) and text[i + 1] == '[':
                i += 2  # 跳过 '\x1b['
                codes = []
                code_str = ''
                
                # 收集所有数字代码
                while i < len(text) and text[i].isdigit() or text[i] == ';':
                    if text[i] == ';':
                        if code_str:
                            codes.append(int(code_str))
                            code_str = ''
                    else:
                        code_str += text[i]
                    i += 1
                
                if code_str:
                    codes.append(int(code_str))
                
                # 确保有终止字符
                if i < len(text) and text[i] == 'm':
                    i += 1  # 跳过 'm'
                    
                    # 处理代码
                    for code in codes:
                        if code == 0:  # 重置所有样式
                            current_style = {}
                        elif code == 1:  # 粗体
                            current_style['font'] = ('SimHei', 10, 'bold')
                        elif code == 2:  # 变暗
                            # 根据背景色调整灰色深浅
                            current_style['foreground'] = '#BBBBBB' if is_dark_bg else '#808080'
                        elif code == 30:  # 黑色
                            # 在黑色背景下显示深灰色
                            current_style['foreground'] = '#444444' if is_dark_bg else '#000000'
                        elif code == 31:  # 红色
                            current_style['foreground'] = '#FF6666' if is_dark_bg else '#FF0000'
                        elif code == 32:  # 绿色
                            current_style['foreground'] = '#66FF66' if is_dark_bg else '#00FF00'
                        elif code == 33:  # 黄色
                            current_style['foreground'] = '#FFFF66' if is_dark_bg else '#FFFF00'
                        elif code == 34:  # 蓝色
                            current_style['foreground'] = '#6666FF' if is_dark_bg else '#0000FF'
                        elif code == 35:  # 洋红色
                            current_style['foreground'] = '#FF66FF' if is_dark_bg else '#FF00FF'
                        elif code == 36:  # 青色
                            current_style['foreground'] = '#66FFFF' if is_dark_bg else '#00FFFF'
                        elif code == 37:  # 白色
                            # 在白色背景下显示黑色
                            current_style['foreground'] = '#FFFFFF' if is_dark_bg else '#000000'
            else:
                # 查找下一个转义序列或文本结束
                j = i
                while j < len(text) and not (text[j] == '\x1b' and j + 1 < len(text) and text[j + 1] == '['):
                    j += 1
                
                # 获取纯文本部分
                plain_text = text[i:j]
                if plain_text:
                    # 为当前文本创建唯一的tag
                    tag_name = f"style_{tag_count}"
                    tag_count += 1
                    
                    # 插入文本
                    self.output_text.insert(tk.END, plain_text, tag_name)
                    
                    # 应用样式
                    for attr, value in current_style.items():
                        self.output_text.tag_config(tag_name, **{attr: value})
                
                i = j
                
    def change_log_background(self):
        """切换日志区域的背景颜色"""
        # 获取选择的背景颜色
        bg_color = "black" if self.log_bg_var.get() == "black" else "white"
        fg_color = "white" if self.log_bg_var.get() == "black" else "black"
        
        # 保存当前文本内容
        self.output_text.config(state=tk.NORMAL)
        current_content = self.output_text.get(1.0, tk.END)
        
        # 清空文本并设置新的背景和前景色
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(bg=bg_color, fg=fg_color)
        
        # 重新插入文本（这会丢失之前的颜色格式, 但确保基本可读性）
        # 注意：完全重建颜色格式需要重新解析ANSI序列, 这里采用简化处理
        self.output_text.insert(tk.END, current_content)
        self.output_text.config(state=tk.DISABLED)
    
    def update_output(self):
        """更新输出区域的文本"""
        while not self.output_queue.empty():
            text = self.output_queue.get()
            self.output_text.config(state=tk.NORMAL)
            # 使用ANSI颜色解析函数
            self.parse_ansi_color(text)
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
        
        # 继续检查队列
        self.root.after(100, self.update_output)
    
    def build_command(self):
        """构建执行脚本的命令"""
        command = [self.script_path]
        
        # 添加参数
        if self.tag_var.get():
            command.extend(["--tag", self.tag_var.get()])
        elif self.docker_var.get() != "docker.fftaicorp.com/farts/farther:onekey-daqdeploy":
            command.extend(["--docker", self.docker_var.get()])
        
        command.extend(["--graph", self.graph_var.get()])
        command.extend(["--notes", self.notes_var.get()])
        command.extend(["--pilot", self.pilot_var.get()])
        command.extend(["--operator", self.operator_var.get()])
        command.extend(["--station-id", self.station_id_var.get()])
        command.extend(["--machine-id", self.machine_id_var.get()])
        command.extend(["--domain-id", self.domain_id_var.get()])
        
        if self.env_file_var.get():
            command.extend(["--env-file", self.env_file_var.get()])
        
        if self.debug_var.get():
            command.append("--debug")
        
        if self.local_mode_var.get():
            command.append("--local-mode")
        
        return command
    
    def start_container(self):
        """启动Docker容器"""
        if self.running:
            messagebox.showwarning(self._("警告"), self._("容器已经在运行中"))
            return
        
        # 清空日志输出
        self.clear_output()
        
        # 构建命令
        command = self.build_command()
        
        # 显示命令
        self.append_output(f"{self._('执行命令: ')}{' '.join(command)}\n\n")
        
        # 在新线程中执行命令
        self.running = True
        thread = threading.Thread(target=self.execute_command, args=(command,))
        thread.daemon = True
        thread.start()
    
    def execute_command(self, command):
        """执行命令并捕获输出"""
        try:
            # 执行命令
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 读取输出
            for line in self.process.stdout:
                if not self.running:
                    break
                self.append_output(line)
            
            # 等待进程结束
            if self.process:
                self.process.wait()
            
            # 检查退出码
            if self.process is None or self.process.returncode == 0:
                self.append_output(f"\n{self._('命令执行成功完成')}\n")
            else:
                self.append_output(f"\n{self._('命令执行失败, 退出码: ')}{self.process.returncode}\n")
                
        except Exception as e:
            self.append_output(f"\n{self._('执行命令时出错: ')}{str(e)}\n")
        finally:
            self.process = None
            self.running = False
    
    def stop_container(self):
        """停止容器执行"""
        # 先尝试通过进程对象停止
        process_stopped = False
        
        if self.running and self.process:
            self.append_output(f"\n{self._('正在停止容器进程...')}\n")
            self.running = False
            
            try:
                # 首先发送SIGINT信号（相当于Ctrl+C）
                self.append_output(f"{self._('发送Ctrl+C信号...')}\n")
                os.kill(self.process.pid, signal.SIGINT)
                
                # 等待进程响应Ctrl+C并优雅退出
                try:
                    self.process.wait(timeout=3)
                    process_stopped = True
                    self.append_output(f"{self._('进程响应Ctrl+C信号并退出')}\n")
                except subprocess.TimeoutExpired:
                    # 如果进程没有响应Ctrl+C, 尝试terminate
                    self.append_output(f"{self._('进程未响应Ctrl+C, 尝试正常终止...')}\n")
                    self.process.terminate()
                    self.process.wait(timeout=3)
                    process_stopped = True
                    self.append_output(f"{self._('进程已正常终止')}\n")
            except subprocess.TimeoutExpired:
                # 如果超时, 强制终止
                self.append_output(f"{self._('强制终止进程...')}\n")
                self.process.kill()
                process_stopped = True
            except Exception as e:
                self.append_output(f"{self._('停止进程时出错: ')}{str(e)}\n")
            
            self.process = None
        
        # 无论进程是否成功停止, 都尝试使用docker命令停止所有相关容器
        self.append_output(f"\n{self._('正在检查并停止Docker容器...')}\n")
        try:
            # 使用docker ps命令查找包含gr3或daq相关的容器
            ps_command = ["docker", "ps", "--format", "{{.Names}}"]
            process = subprocess.Popen(
                ps_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            container_names = process.stdout.read().splitlines()
            if process:
                process.wait()
            
            # 筛选可能相关的容器
            relevant_containers = []
            for name in container_names:
                if any(keyword in name.lower() for keyword in ["gr3", "daq", "farther"]):
                    relevant_containers.append(name)
            
            if relevant_containers:
                self.append_output(f"{self._('找到相关容器: ')}{', '.join(relevant_containers)}\n")
                # 停止找到的容器
                stop_command = ["docker", "stop"] + relevant_containers
                process = subprocess.Popen(
                    stop_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                for line in process.stdout:
                    self.append_output(line)
                if process:
                    process.wait()
                
                if process.returncode == 0:
                    self.append_output(f"{self._('所有相关容器已成功停止')}\n")
                else:
                    self.append_output(f"{self._('停止容器时遇到问题')}\n")
            else:
                if not process_stopped:
                    messagebox.showinfo(self._('信息'), self._('没有找到正在运行的相关容器'))
                    return
        except Exception as e:
            self.append_output(f"{self._('执行Docker命令时出错: ')}{str(e)}\n")
        
        self.append_output(f"{self._('容器停止操作完成')}\n")
    
    def clear_output(self):
        """清空输出区域"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state=tk.DISABLED)

    def start_local_server(self):
        """启动本地服务"""
        self.append_output(f"{self._('正在启动本地服务...')}\n")
        # 使用绝对路径确保找到脚本
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(gui_dir, "../livekit", "start_local_server.sh")
        script_path = os.path.abspath(script_path)  # 转换为绝对路径
        self.append_output(f"{self._('脚本路径:')} {script_path}\n")
        # 在新线程中执行脚本
        thread = threading.Thread(target=self.execute_script, args=(script_path, "启动本地服务"))
        thread.daemon = True
        thread.start()
        
    def stop_local_server(self):
        """停止本地服务"""
        self.append_output(f"{self._('正在停止本地服务...')}\n")
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(gui_dir, "../livekit", "stop_local_server.sh")
        script_path = os.path.abspath(script_path)  # 转换为绝对路径
        self.append_output(f"{self._('脚本路径:')} {script_path}\n")
        # 在新线程中执行脚本
        thread = threading.Thread(target=self.execute_script, args=(script_path, "停止本地服务"))
        thread.daemon = True
        thread.start()
        
    def update_docker_image(self):
        """更新数采镜像"""
        self.append_output(f"{self._('正在更新数采镜像...')}\n")
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(gui_dir, "update_docker.sh")
        script_path = os.path.abspath(script_path)  # 转换为绝对路径
        self.append_output(f"{self._('脚本路径:')} {script_path}\n")
        # 在新线程中执行脚本
        thread = threading.Thread(target=self.execute_script, args=(script_path, "更新数采镜像"))
        thread.daemon = True
        thread.start()
        
    def show_docker_status(self):
        """查看镜像状态"""
        self.append_output(f"{self._('正在查询镜像状态...')}\n")
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(gui_dir, "show_docker.sh")
        script_path = os.path.abspath(script_path)  # 转换为绝对路径
        self.append_output(f"{self._('脚本路径:')} {script_path}\n")
        # 在新线程中执行脚本
        thread = threading.Thread(target=self.execute_script, args=(script_path, "查看镜像状态"))
        thread.daemon = True
        thread.start()
        
    def connect_local_headset(self):
        """连接本地头显"""
        self.append_output(f"{self._('正在连接本地头显...')}\n")
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(gui_dir, "../livekit", "connect_quest.sh")
        script_path = os.path.abspath(script_path)  # 转换为绝对路径
        self.append_output(f"{self._('脚本路径:')} {script_path}\n")
        # 在新线程中执行脚本
        thread = threading.Thread(target=self.execute_script, args=(script_path, "连接本地头显"))
        thread.daemon = True
        thread.start()
        
    def execute_script(self, script_path, operation_name):
        """执行shell脚本并在输出日志中显示结果
        
        Args:
            script_path: 脚本路径
            operation_name: 操作名称
        """
        import stat
        try:
            # self.append_output(f"{self._('开始执行脚本:')} {script_path}\n")
            
            # 检查脚本是否存在
            # self.append_output(f"{self._('检查脚本是否存在...')}\n")
            if not os.path.exists(script_path):
                self.append_output(f"{self._('错误:')} {self._('脚本不存在')}: {script_path}\n")
                return
            
            # 检查脚本是否有执行权限
            # self.append_output(f"{self._('检查脚本执行权限...')}\n")
            if not os.access(script_path, os.X_OK):
                self.append_output(f"{self._('警告:')} {self._('脚本没有执行权限')}: {script_path}\n")
                # 尝试添加执行权限
                try:
                    self.append_output(f"{self._('尝试添加执行权限...')}\n")
                    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    self.append_output(f"{self._('已添加执行权限')}\n")
                except Exception as e:
                    self.append_output(f"{self._('添加执行权限失败')}: {str(e)}\n")
                    self.append_output(f"{self._('尝试使用bash解释器直接运行脚本...')}\n")
                    # 即使没有执行权限，仍然尝试通过bash直接运行
            
            # 获取脚本目录
            script_dir = os.path.dirname(os.path.abspath(script_path))
            # self.append_output(f"{self._('脚本目录:')} {script_dir}\n")
            
            # 使用子进程执行脚本并捕获输出
            # self.append_output(f"{self._('启动子进程执行脚本...')}\n")
            process = subprocess.Popen(
                ['bash', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=script_dir
            )
            
            # self.append_output(f"{self._('开始捕获脚本输出...')}\n")
            # 实时获取输出
            while True:
                # 获取标准输出
                stdout_line = process.stdout.readline()
                if stdout_line:
                    self.append_output(stdout_line)
                
                # 获取错误输出
                stderr_line = process.stderr.readline()
                if stderr_line:
                    self.append_output(stderr_line)
                
                # 检查进程是否结束
                if process.poll() is not None:
                    break
                
                # 允许UI更新
                time.sleep(0.1)
            
            # 处理剩余的输出
            # self.append_output(f"{self._('处理剩余输出...')}\n")
            for stdout_line in process.stdout.readlines():
                self.append_output(stdout_line)
            for stderr_line in process.stderr.readlines():
                self.append_output(stderr_line)
            
            # 检查执行结果
            # self.append_output(f"{self._('脚本执行完成，返回码:')} {process.returncode}\n")
            if process.returncode == 0:
                self.append_output(f"{self._('成功:')} {self._(operation_name)} {self._('完成')}\n")
            else:
                self.append_output(f"{self._('失败:')} {self._(operation_name)} {self._('失败，返回码:')} {process.returncode}\n")
                
        except Exception as e:
            self.append_output(f"{self._('执行脚本时发生错误')}: {str(e)}\n")
            import traceback
            error_trace = traceback.format_exc()
            self.append_output(f"{self._('错误详情:')}\n{error_trace}\n")
    
    def show_help(self):
        """显示run_gr3.sh脚本的帮助信息"""
        if self.running:
            messagebox.showwarning(self._("警告"), self._("容器正在运行中, 请先停止容器后再查看帮助"))
            return
        
        # 清空日志输出
        self.clear_output()
        self.append_output(f"{self._('正在获取帮助信息...')}\n\n")
        
        # 构建帮助命令
        help_command = [self.script_path, "--help"]
        
        # 在新线程中执行命令
        thread = threading.Thread(target=self.execute_help_command, args=(help_command,))
        thread.daemon = True
        thread.start()
    
    def execute_help_command(self, command):
        """执行帮助命令并捕获输出"""
        try:
            # 执行命令
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 读取输出
            for line in process.stdout:
                self.append_output(line)
            
            # 等待进程结束
            process.wait()
            
            # 检查退出码
            if process.returncode == 0:
                self.append_output(f"\n{self._('帮助信息获取成功')}\n")
            else:
                self.append_output(f"\n{self._('获取帮助信息失败, 退出码: ')}{process.returncode}\n")
                
        except Exception as e:
            self.append_output(f"\n{self._('获取帮助信息时出错: ')}{str(e)}\n")


def main():
    """主函数"""
    # 检查是否以root权限运行
    # if os.geteuid() != 0:
    #     print("警告: 此程序可能需要以root权限运行以使用Docker")
    
    # 创建GUI窗口
    root = tk.Tk()
    app = GR3GUI(root)
    
    # 处理窗口关闭事件
    def on_closing():
        if app.running and messagebox.askokcancel(app._('退出'), app._('容器正在运行中, 确定要退出吗？')):
            app.stop_container()
            root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # 运行主循环
    root.mainloop()


if __name__ == "__main__":
    main()