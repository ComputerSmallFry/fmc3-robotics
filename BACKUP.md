# 遥操环境备份与恢复

## 1. Docker 镜像备份

### 1.1 查看当前镜像

```bash
docker images | grep -E "farther|teleop|livekit"
```

### 1.2 逐个保存镜像

```bash
# GR3 遥操主镜像（最新）
docker save -o farther-onekey-260123060517.tar \
  docker.fftaicorp.com/farts/farther:onekey-260123060517-462cac7b

# GR3 遥操主镜像
docker save -o farther-onekey-260123030919.tar \
  docker.fftaicorp.com/farts/farther:onekey-260123030919-2a02f481

# GR3 遥操主镜像（旧版）
docker save -o farther-onekey-daqdeploy.tar \
  docker.fftaicorp.com/farts/farther:onekey-daqdeploy

# LiveKit teleop 客户端（agv）
docker save -o teleop-client-agv.tar \
  docker.fftaicorp.com/farts/teleop-client:agv-0.1.0

# LiveKit teleop 客户端
docker save -o teleop-client-latest.tar \
  docker.fftaicorp.com/farts/teleop-client:latest

# LiveKit 视频查看器
docker save -o teleop-viewer-latest.tar \
  docker.fftaicorp.com/farts/teleop-viewer:latest

# LiveKit 服务端
docker save -o livekit-server-v1.9.0.tar \
  docker.fftaicorp.com/devops/livekit/livekit-server:v1.9.0
```

### 1.3 一次性保存全部镜像（推荐，共享层会去重，体积更小）

```bash
docker save -o teleop-images-all.tar \
  docker.fftaicorp.com/farts/farther:onekey-260123060517-462cac7b \
  docker.fftaicorp.com/farts/farther:onekey-260123030919-2a02f481 \
  docker.fftaicorp.com/farts/farther:onekey-daqdeploy \
  docker.fftaicorp.com/farts/teleop-client:agv-0.1.0 \
  docker.fftaicorp.com/farts/teleop-client:latest \
  docker.fftaicorp.com/farts/teleop-viewer:latest \
  docker.fftaicorp.com/devops/livekit/livekit-server:v1.9.0
```

### 1.4 压缩备份（节省磁盘空间）

```bash
# 保存并压缩
docker save \
  docker.fftaicorp.com/farts/farther:onekey-260123060517-462cac7b \
  docker.fftaicorp.com/farts/farther:onekey-260123030919-2a02f481 \
  docker.fftaicorp.com/farts/farther:onekey-daqdeploy \
  docker.fftaicorp.com/farts/teleop-client:agv-0.1.0 \
  docker.fftaicorp.com/farts/teleop-client:latest \
  docker.fftaicorp.com/farts/teleop-viewer:latest \
  docker.fftaicorp.com/devops/livekit/livekit-server:v1.9.0 \
  | gzip > teleop-images-all.tar.gz
```

## 2. Docker 镜像恢复

### 2.1 从 tar 文件恢复

```bash
docker load -i teleop-images-all.tar
```

### 2.2 从压缩文件恢复

```bash
gunzip -c teleop-images-all.tar.gz | docker load
```

### 2.3 验证恢复结果

```bash
docker images | grep -E "farther|teleop|livekit"
```

## 3. daq-deploy 文件夹备份

### 3.1 备份 daq-deploy 部署代码

```bash
tar -czf daq-deploy-backup.tar.gz -C ~/  daq-deploy/
```

### 3.2 备份 daq-deploy 及录制数据

```bash
# 仅部署代码
tar -czf daq-deploy-backup.tar.gz -C ~/ daq-deploy/

# 录制数据（GR3 数据目录）
tar -czf daq-data-backup.tar.gz -C ~/ data/

# 合并备份（部署代码 + 数据）
tar -czf daq-full-backup.tar.gz -C ~/ daq-deploy/ data/
```

### 3.3 备份配置文件（轻量备份）

```bash
# 仅备份关键配置
tar -czf daq-config-backup.tar.gz \
  ~/daq-deploy/gr3/config.env \
  ~/daq-deploy/gr3/run_gr3.sh \
  ~/daq-deploy/livekit/docker-compose.yaml \
  ~/daq-deploy/daemon/
```

## 4. daq-deploy 文件夹恢复

### 4.1 恢复部署代码

```bash
tar -xzf daq-deploy-backup.tar.gz -C ~/
```

### 4.2 恢复录制数据

```bash
tar -xzf daq-data-backup.tar.gz -C ~/
```

### 4.3 恢复完整备份

```bash
tar -xzf daq-full-backup.tar.gz -C ~/
```

## 5. 完整备份脚本（一键执行）

```bash
#!/bin/bash
BACKUP_DIR=~/backup/$(date +%Y%m%d)
mkdir -p "$BACKUP_DIR"

echo "=== 备份 daq-deploy 文件夹 ==="
tar -czf "$BACKUP_DIR/daq-deploy.tar.gz" -C ~/ daq-deploy/

echo "=== 备份录制数据 ==="
tar -czf "$BACKUP_DIR/daq-data.tar.gz" -C ~/ data/

echo "=== 备份 Docker 镜像（耗时较长）==="
docker save \
  docker.fftaicorp.com/farts/farther:onekey-260123060517-462cac7b \
  docker.fftaicorp.com/farts/farther:onekey-260123030919-2a02f481 \
  docker.fftaicorp.com/farts/farther:onekey-daqdeploy \
  docker.fftaicorp.com/farts/teleop-client:agv-0.1.0 \
  docker.fftaicorp.com/farts/teleop-client:latest \
  docker.fftaicorp.com/farts/teleop-viewer:latest \
  docker.fftaicorp.com/devops/livekit/livekit-server:v1.9.0 \
  | gzip > "$BACKUP_DIR/teleop-images-all.tar.gz"

echo "=== 备份完成 ==="
ls -lh "$BACKUP_DIR/"
```

## 6. 完整恢复脚本（一键执行）

```bash
#!/bin/bash
BACKUP_DIR=~/backup/20260211  # 修改为实际备份日期

echo "=== 恢复 daq-deploy 文件夹 ==="
tar -xzf "$BACKUP_DIR/daq-deploy.tar.gz" -C ~/

echo "=== 恢复录制数据 ==="
tar -xzf "$BACKUP_DIR/daq-data.tar.gz" -C ~/

echo "=== 恢复 Docker 镜像（耗时较长）==="
gunzip -c "$BACKUP_DIR/teleop-images-all.tar.gz" | docker load

echo "=== 恢复完成，验证镜像 ==="
docker images | grep -E "farther|teleop|livekit"
```
