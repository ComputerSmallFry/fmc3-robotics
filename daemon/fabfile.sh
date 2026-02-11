# --hosts=11,12,13,14,15,16,17,18,21,22,23,24 为当前所有的数采小电脑

# 停止daemon
fab run --hosts=11,12,13,14,15,16,17,18,21,22,23,24 --cmd='cd farts_deploy && docker compose down'

# 重启daemon
fab run --hosts=11,12,13,14,15,16,17,18,21,22,23,24 --cmd='cd farts_deploy && ./start.sh'

# 查看最新5条日志
fab run --hosts=11,12,13,14,15,16,17,18,21,22,23,24 --cmd='cd farts_deploy && docker logs daemon -n 5'

# 更新 .env
fab run --hosts=11,12,13,14,15,16,17,18,21,22,23,24 --cmd='cd farts_deploy && wget -O .env http://srv002.farts.com:8099/.env'
# 或者用 fab rput 直接传输本地文件

# 查看本地数据空间占用
fab run --hosts=11,12,13,14,15,16,17,18,21,22,23,24 --cmd='df -lh | grep /mnt/Data || echo $HOSTNAME'
