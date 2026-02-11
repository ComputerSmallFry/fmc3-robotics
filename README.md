# DAQ-Deploy

本文档为 `GRx` 系列机器人遥操作服务的部署和使用说明。

This document provides deployment and usage instructions for the `GRx` series robot teleoperation services.

---

- [1. SETUP](#1-setup)
  - [1.1 Get the code](#11-get-the-code)
  - [1.2 Setup a new PC](#12-setup-a-new-pc)
  - [1.3 Setup udev rules](#13-setup-udev-rules)
  - [1.4 Setup user group](#14-setup-user-group)
  - [1.5 Option - Prepare GR3 GUI](#15-option---prepare-gr3-gui)
  - [1.6 Option - Prepare GR2 config](#16-option---prepare-gr2-config)
- [2. RUN](#2-run)
  - [2.1 Run GR3 in FFTAI Corp](#21-run-gr3-in-fftai-corp)
  - [2.2 Run GR3 in local mode](#22-run-gr3-in-local-mode)
    - [(1) Start local livekit server](#1-start-local-livekit-server)
    - [(2) Run GR3 in local mode](#2-run-gr3-in-local-mode)
  - [2.3 Run GR2 with oak camera](#23-run-gr2-with-oak-camera)
  - [2.4 Run GR2 with orbbec camera](#24-run-gr2-with-orbbec-camera)

## 1. SETUP

本节为环境配置，通常每台电脑仅需要配置一次。完成全部配置后，建议重启电脑。

This section covers environment configuration, which typically needs to be done only once per computer. After completing all configurations, it is recommended to restart the computer.

### 1.1 Get the code

通过 `token` 拉取代码。非公司内网环境可以提前获取，使用移动设备复制。

Pull the code using a `token`. For environments outside the company intranet, you can obtain it in advance and copy it using a mobile device.

```sh
git clone https://oauth2:glpat-BizpMtjjXmhsc3bipjDi@gitlab.fftaicorp.com/farts/daq-deploy.git
```

公司内网环境下更新代码。

Update code in the company intranet environment:


```sh
cd daq-deploy
git fetch origin && git reset --hard origin/main && git clean -fd
```

### 1.2 Setup a new PC

对于全新安装 `Ubuntu 22.04` 的数采电脑，采用以下脚本安装 `docker` 等运行环境。

For data acquisition computers with a fresh installation of `Ubuntu 22.04`, use the following script to install `docker` and other runtime environments.

```sh
bash setup/provisioning.sh
```

### 1.3 Setup udev rules

配置 `udev` 规则，以便访问相机、外骨骼等 `USB` 设备。

Configure `udev` rules to access cameras, exoskeletons, and other `USB` devices.

```sh
sudo bash udev/setup_udev.sh
```

### 1.4 Setup user group

添加当前用户权限，以便代码正常访问。

Add current user permissions for proper code access.

```sh
sudo bash setup/setup_user_group.sh
```

### 1.5 Option - Prepare GR3 GUI

安装 `GR3` 遥操作 GUI 桌面图标。

Install the `GR3` teleoperation GUI desktop icon.

```sh
bash gr3/install_gui.sh
```

在桌面上找到 `daq-gui.desktop` 图标，右键点击后选择 `Allow Launching` 以允许启动。

Find `daq-gui.desktop` on desktop and right-click to select `Allow Launching` to enable launching.

### 1.6 Option - Prepare GR2 config

仅针对 `GR2` 遥操作配置。

Teleoperation config for `GR2`.

下载 https://miniserver.fftaicorp.com/daq/gr2/teleoperation.zip 到 `$HOME` 并解压，如下所示：

Download https://miniserver.fftaicorp.com/daq/gr2/teleoperation.zip to `$HOME` and unzip it, like:
```
❯ cd ~ && tree ./teleoperation
./teleoperation
├── configs
│   ├── camera
│   │   ├── oak_97_high_res.yaml
│   │   ├── oak_97_legacy.yaml
│   │   ├── oak_97_multi.yaml
│   │   ├── oak_97.yaml
│   │   ├── oak.yaml
```


## 2. RUN

本节为运行教程，各部分为不同模式、独立运行。

This section contains running tutorials. Each part represents a different mode and can be run independently.

对于各个模式，均需要 `ssh` 远程连接机器人 `NUC` 并启动 `AuroraCore`：

For each mode, you need to `ssh` into the robot’s `NUC` and start `AuroraCore`:

```sh
# If robot id is gr301aa0017
ssh gr301aa0017@192.168.137.220
# password: fftai2015

# Select a tag and enter docker
deploy_aurora tag Algorithm-aurora_unified-305-GR3

# Launch aurora server
AuroraCore
```

### 2.1 Run GR3 in FFTAI Corp

使用公司内网服务运行 `GR3` 机器人遥操作。

Run `GR3` robot teleoperation using the company intranet service.

```sh
bash gr3/run_gr3.sh --help
```
* Viewer: https://teleop-viewer.fftaicorp.com/
* VR client: https://teleop-client.fftaicorp.com/

### 2.2 Run GR3 in local mode

使用本地网络模式运行 `GR3` 机器人遥操作。

Run `GR3` robot teleoperation in local network mode.

#### (1) Start local livekit server

启动本地 LiveKit 服务器，用于 `GR3` 机器人遥操作。

Start the local LiveKit server for `GR3` robot teleoperation.

```sh
bash livekit/start_local_server.sh
```

#### (2) Run GR3 in local mode

在本地模式下运行 `GR3` 机器人遥操作。

Run `GR3` robot teleoperation in local network mode.

```sh
bash gr3/run_gr3.sh --local-mode --help
```

If use `--graph daq-quest-depthai.yml`, then connect quest to PC via USB and:
```sh
bash livekit/connect_quest.sh
```

* Viewer: http://localhost:8081/
* VR client: http://localhost:8080/


### 2.3 Run GR2 with oak camera

使用 `OAK` 相机运行 `GR2` 机器人遥操作。

Run `GR2` robot teleoperation using `OAK` camera.

```sh
bash gr2/run_t5d_depthai.sh gr2t2d oak_97 t5dtest use_depth=true robot.visualize=true recording.pilot=-1 recording.operator=-1 hydra.job.chdir=false hand=fourier_dexpilot_dhx use_head=true
```

### 2.4 Run GR2 with orbbec camera

使用 `Orbbec` 相机运行 `GR2` 机器人遥操作。

Run `GR2` robot teleoperation using `Orbbec` camera.

```sh
bash gr2/run_t5d_orbbec.sh gr2t2d orbbec t5dtestorbbec use_depth=false robot.visualize=true recording.pilot=-1 recording.operator=-1 hydra.job.chdir=false hand=fourier_dexpilot_dhx use_head=true camera.instance.display_config.mode=mono
```
