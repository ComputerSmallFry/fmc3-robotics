#!/bin/bash

echo "MACHINE_ID=$(echo $HOSTNAME | sed 's/[^0-9]//g')" >> ~/.bashrc

# Step 1: Update the system
echo "Updating the system..."
sudo apt update && sudo apt upgrade -y

# Step 2: Install basic tools
echo "Installing basic tools..."
sudo apt install -y net-tools ssh curl wget gcc build-essential git vim fonts-powerline jq btop xclip xsel zstd pv adb

curl -sS https://starship.rs/install.sh | sh -s -- -y
echo 'eval "$(starship init bash)"' >> ~/.bashrc

curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
  bash get-docker.sh
rm get-docker.sh

sudo groupadd -f docker
sudo usermod -aG docker $USER

# add local registry
echo '{}' | jq '. + {"registry-mirrors": [
    "https://docker.nastool.de",
    "https://docker.1ms.run",
    "https://docker.1panel.live",
    "https://hub1.nat.tf",
    "https://docker.1panel.top",
    "https://dockerpull.org",
    "https://docker.13140521.xyz"
],"insecure-registries": ["docker.fftaicorp.com"]}' | sudo tee /etc/docker/daemon.json >/dev/null 2>&1

# sudo systemctl daemon-reload
# sudo systemctl restart docker

# Install VS Code
sudo snap install --classic zellij
sudo snap install --classic code

wget https://releases.hyper.is/download/deb -O hyper.deb
sudo dpkg -i ./hyper.deb
rm hyper.deb

echo "Setup completed! Restart the computer now."

