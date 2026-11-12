#!/usr/bin/env bash
# 虚拟 zsibot_control 启动包装（Linux）：把启动参数透传给 Python 脚本。
# SdkDaemonClient 会以 [本文件, --daemon, --dog-ip, ...] 形式拉起，脚本忽略这些参数。
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$DIR/virtual_zsibot_control.py" "$@"
