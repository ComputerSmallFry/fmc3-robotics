# 创建文件，请将尖括号内的内容替换为你的实际密钥
```sh
sudo echo "${S3_FAST_MINIO_ACCESS_KEY}:${S3_FAST_MINIO_SECRET_KEY}" > /etc/passwd-s3fs
```

# 设置严格的权限，这是s3fs的安全要求
```sh
sudo chmod 600 /etc/passwd-s3fs
```

# 在 /etc/fstab 末尾添加如下一行
```sh
s3fs#farther-data /home/your_username/s3/farther-data fuse _netdev,allow_other,use_path_request_style,url=https://s3.fftaicorp.com:443,passwd_file=/etc/passwd-s3fs,uid=1000,gid=1000,umask=022 0 0
```

# 添加配置后，使用以下命令测试是否正确，它会对 /etc/fstab 中所有配置进行挂载尝试：
```sh
sudo mount -a
```