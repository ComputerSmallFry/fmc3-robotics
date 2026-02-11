# 检查并安装uv包管理器
if ! command -v uv &> /dev/null; then
    echo "uv 未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 将uv添加到PATH（针对当前会话）
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "uv 安装完成"
else
    echo "uv 已安装，跳过安装步骤"
fi

cd "$(dirname "$0")"
# Get the absolute path of the current script directory
SCRIPT_DIR="$(pwd)"

# Directly create daq-gui.desktop on desktop using $HOME environment variable
cat > "$HOME/Desktop/daq-gui.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DAQ GUI
Comment=DAQ GUI
Exec=$SCRIPT_DIR/launch_gui.sh
Icon=utilities-terminal
Terminal=false
StartupNotify=true
EOF

chmod +x "$HOME/Desktop/daq-gui.desktop"
chmod +x ./launch_gui.sh

cd ..

if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    uv venv -p 3.13 --seed
else
    echo "虚拟环境已存在，跳过创建步骤"
fi
