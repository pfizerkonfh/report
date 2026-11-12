@echo off
REM 虚拟 zsibot_control 启动包装（Windows）：把启动参数透传给 Python 脚本。
REM SdkDaemonClient 会以 [本文件, --daemon, --dog-ip, ...] 形式拉起，脚本忽略这些参数。
setlocal
set "SCRIPT=%~dp0virtual_zsibot_control.py"
python "%SCRIPT%" %*
