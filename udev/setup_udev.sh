#!/bin/bash

# Set target directory
TARGET_DIR="/etc/udev/rules.d"
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if user has permission to write to the target directory
if [ ! -w "$TARGET_DIR" ] && [ "$(id -u)" -ne 0 ]; then
    echo "Error: Root permission is required to copy files to $TARGET_DIR"
    echo "Please run this script with sudo"
    exit 1
fi

echo "Starting to check and copy udev rule files..."
echo "Source directory: $SCRIPT_DIR"
echo "Target directory: $TARGET_DIR"
echo "----------------------------------------"

# Iterate through all .rules files in the current directory
for rules_file in "$SCRIPT_DIR"/*.rules; do
    # Ensure the file exists
    if [ -f "$rules_file" ]; then
        # Get the filename
        filename="$(basename "$rules_file")"
        target_file="$TARGET_DIR/$filename"
        
        # Check if the target file exists
        if [ -f "$target_file" ]; then
            echo "File already exists: $filename"
            
            # Check if contents are different
            if ! cmp -s "$rules_file" "$target_file"; then
                echo "  Note: File contents are different, will update"
                sudo cp "$rules_file" "$target_file"
                echo "  ✓ File updated"
            else
                echo "  ✓ File contents are identical, no update needed"
            fi
        else
            echo "Copying file: $filename"
            sudo cp "$rules_file" "$target_file"
            echo "  ✓ File copied"
        fi
        
        # Set correct permissions
        sudo chmod 644 "$target_file"
    fi
done

echo "----------------------------------------"
echo "Reloading udev rules..."
sudo udevadm control --reload-rules
echo "Triggering udev devices..."
sudo udevadm trigger
echo "----------------------------------------"
echo "Udev rule setup completed!"