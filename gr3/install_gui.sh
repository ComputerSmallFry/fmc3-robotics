sudo apt install -y python3-tk

cd "$(dirname "$0")"
# Get the absolute path of the current script directory
SCRIPT_DIR="$(pwd)"

# Dynamically find the desktop path using xdg-user-dir command
DESKTOP_PATH=$(xdg-user-dir DESKTOP 2>/dev/null)

# Fallback to $HOME/Desktop if xdg-user-dir is not available or fails
if [ -z "$DESKTOP_PATH" ]; then
    DESKTOP_PATH="$HOME/Desktop"
fi

# Create daq-gui.desktop on the dynamically found desktop path
cat > "$DESKTOP_PATH/daq-gui.desktop" << EOF
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

chmod +x "$DESKTOP_PATH/daq-gui.desktop"
chmod +x ./launch_gui.sh

cd ..
