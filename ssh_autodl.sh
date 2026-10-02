#!/usr/bin/env bash
# 用项目内的专用密钥连接 AutoDL（免密，非交互）。
# 需要先让远程把 .remote-key/id_ed25519.pub 装进 ~/.ssh/authorized_keys。
exec ssh -i "$(dirname "$0")/.remote-key/id_ed25519" \
     -o IdentitiesOnly=yes \
     -o BatchMode=yes \
     -o StrictHostKeyChecking=accept-new \
     -o ServerAliveInterval=30 \
     -o ConnectTimeout=15 \
     -p 16402 root@connect.nmb2.seetacloud.com "$@"
