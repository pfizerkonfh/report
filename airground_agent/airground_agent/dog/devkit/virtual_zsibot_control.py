#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""虚拟 zsibot_control：模仿 main_sdk.cpp 的 `--daemon` 模式（本地测试用，非真机）。

协议：stdin 读一行命令文本，stdout 回一行 JSON。响应形状严格对齐
`main_sdk.cpp` 的 `send_json_response(bool, command, message, data_json)`：
    {"ok":true,"command":"stand_up","message":"stand_up"}
    {"ok":true,"command":"get_status","message":"get_status","data":{...}}

忽略启动参数 `--daemon / --dog-ip / --send-port / --recv-port`。
真实二进制是 Ubuntu 上 `make` 出的 C++ 程序；本脚本是它在开发机上的"替身"，
让 `SdkDaemonClient` 能在没有真机的机器上用真实子进程 stdin/stdout 跑通整条链路。
"""
from __future__ import annotations

import json
import sys
import time


def make_status() -> dict:
    """模仿 make_status_json 的字段（电量等给固定演示值）。"""
    return {
        "power": 87.5,
        "battery_power": 87.5,
        "mode": "stand",
        "online": True,
        "uptime_s": int(time.time()) % 100000,
    }


def respond(command: str):
    """返回要写回 stdout 的 JSON 字符串（或 None 表示退出）。"""
    if command == "quit":
        return json.dumps({"ok": True, "command": "quit", "message": "quit"},
                          ensure_ascii=False)
    if command == "get_status":
        return json.dumps({"ok": True, "command": "get_status",
                           "message": "get_status", "data": make_status()},
                          ensure_ascii=False)
    # 其余已知/未知动作统一返回 ok（真机会对未知命令返回 ok=false，
    # 这里对测试友好的放行；本测试只发已知命令）
    return json.dumps({"ok": True, "command": command, "message": command},
                      ensure_ascii=False)


def main() -> None:
    # 逐行读取 stdin（与 main_sdk.cpp 的 daemon 循环一致：一行命令一行响应）
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        out = respond(line)
        sys.stdout.write(out + "\n")
        sys.stdout.flush()
        if line == "quit":
            break


if __name__ == "__main__":
    main()
