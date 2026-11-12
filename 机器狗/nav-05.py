#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import time
import os
from typing import Dict, Optional, Any

import websockets

# ========== 调试开关：通过环境变量 ROBOT_DEBUG=true 开启 ==========
DEBUG = os.getenv("ROBOT_DEBUG", "False").lower() == "true"

class RobotWebSocketClient:
    def __init__(self, uri: str = "ws://192.168.234.234:10010", max_msg_size: int = 50 * 1024 * 1024):
        self.uri = uri
        self.max_msg_size = max_msg_size
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._counter = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def connect(self):
        self.websocket = await websockets.connect(self.uri, max_size=self.max_msg_size)
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
        if self.websocket:
            await self.websocket.close()

    async def _receive_loop(self):
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                head = data.get("head", {})
                if head.get("type") != "app_resp":
                    continue
                frame_count = head.get("frame_count")
                if frame_count is None:
                    continue
                future = self._pending.pop(frame_count, None)
                if future and not future.done():
                    future.set_result(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"接收循环异常: {e}")

    async def _send_request(self, req_func: Any, timeout: float = 30.0) -> Dict:
        async with self._lock:
            self._counter += 1
            frame_count = self._counter
        msg = {
            "head": {
                "type": "app_req",
                "time_stamp": int(time.time() * 1000),
                "source": "app",
                "frame_count": frame_count,
            },
            "data": {"req_func": req_func},
        }
        future = asyncio.get_running_loop().create_future()
        self._pending[frame_count] = future
        await self.websocket.send(json.dumps(msg, ensure_ascii=False))
        try:
            resp = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            self._pending.pop(frame_count, None)
            raise TimeoutError(f"请求 {req_func} 超时")

        req_result = resp.get("data", {}).get("req_result", {})
        if "AppResponse" in req_result:
            req_result = req_result["AppResponse"]
            resp["data"]["req_result"] = req_result

        if req_result.get("status") == "error":
            raise Exception(f"请求失败: {req_result.get('msg', '未知错误')}")
        return resp

    # ---------------- 地图 ----------------
    async def load_localization_map(self, map_id: str):
        await self._send_request({"loc_load_map": map_id})

    async def get_all_pgm_maps(self) -> Dict:
        resp = await self._send_request({"get_all_pgm_map": None})
        return resp.get("data", {}).get("req_result", {}).get("data", {})

    # ---------------- 导航控制 ----------------
    async def start_multi_nav(self, map_id: str, path_id: str):
        await self._send_request({"start_multi_nav": [map_id, path_id]})

    async def stop_nav(self):
        await self._send_request({"stop_nav": None})

    async def start_nav_return_home(self):
        await self._send_request({"start_nav_return_home": None})

    async def pause_nav(self):
        """暂停导航，并返回响应（可能实际效果与 stop 相同）"""
        return await self._send_request({"pause_nav": None})

    async def continue_nav(self):
        return await self._send_request({"continue_nav": None})

    async def get_nav_status(self) -> str:
        resp = await self._send_request({"get_nav_status": None})
        if DEBUG:
            print(f"[DEBUG] get_nav_status 响应:\n{json.dumps(resp, indent=2, ensure_ascii=False)}")
        req_result = resp.get("data", {}).get("req_result", {})
        status = req_result.get("data")
        return status if status is not None else req_result.get("msg", "Unknown")

    async def get_loc_status(self) -> str:
        resp = await self._send_request({"get_loc_status": None})
        if DEBUG:
            print(f"[DEBUG] get_loc_status 响应:\n{json.dumps(resp, indent=2, ensure_ascii=False)}")
        req_result = resp.get("data", {}).get("req_result", {})
        status = req_result.get("data")
        return status if status is not None else req_result.get("msg", "Unknown")

    async def get_mapping_status(self) -> str:
        resp = await self._send_request({"get_mapping_status": None})
        if DEBUG:
            print(f"[DEBUG] get_mapping_status 响应:\n{json.dumps(resp, indent=2, ensure_ascii=False)}")
        req_result = resp.get("data", {}).get("req_result", {})
        status = req_result.get("data")
        return status if status is not None else req_result.get("msg", "Unknown")

    async def start_nav(self, position: Dict, orientation: Optional[Dict] = None):
        req = {"position": position}
        if orientation:
            req["orientation"] = orientation
        await self._send_request({"start_nav": req})

    async def send_raw_request(self, req_func: Any):
        resp = await self._send_request(req_func)
        print("原始响应:")
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return resp


async def print_nav_status(client: RobotWebSocketClient):
    """查询并打印导航状态"""
    try:
        status = await client.get_nav_status()
        print(f"📊 导航状态: {status}")
    except Exception as e:
        print(f"⚠️ 获取导航状态失败: {e}")

async def print_loc_status(client: RobotWebSocketClient):
    try:
        status = await client.get_loc_status()
        print(f"📍 定位状态: {status}")
    except Exception as e:
        print(f"⚠️ 获取定位状态失败: {e}")

async def interactive_menu(client: RobotWebSocketClient):
    print("\n" + "=" * 50)
    print("四足机器人控制台 (支持中文ID)")
    if DEBUG:
        print("【调试模式已开启】")
    print("=" * 50)

    # 初始设置地图ID和路径ID
    map_id = input("\n请输入地图ID: ").strip()
    if not map_id:
        print("地图ID不能为空，退出。")
        return

    path_id = input("请输入路径ID (可选，可直接回车跳过): ").strip()
    print(f"\n当前: 地图={map_id}, 路径={path_id if path_id else '(未提供)'}")

    while True:
        print("\n可选操作：")
        print("  1. 加载定位地图")
        print("  2. 通过路径ID开始多点导航")
        print("  3. 单点导航（手动输入坐标）")
        print("  4. 停止导航")
        print("  5. 开始返航")
        print("  6. 查看导航状态")
        print("  7. 查看定位状态")
        print("  8. 查看建图状态")
        print("  9. 获取所有地图列表")
        print(" 10. 暂停导航")
        print(" 11. 继续导航")
        print(" 12. 发送自定义请求（调试）")
        # ========== 新增功能 ==========
        print(" 13. 修改地图ID/路径ID")
        print(" 14. 查看当前地图ID和路径ID")
        # ==============================
        print("  0. 退出")

        op = input("\n请选择 (0-14): ").strip()

        if op == "1":
            try:
                print(f"加载地图 {map_id} ...")
                await client.load_localization_map(map_id)
                print("✅ 地图加载成功")
            except Exception as e:
                print(f"❌ 加载失败: {e}")

        elif op == "2":
            if not path_id:
                print("❌ 未提供路径ID")
                continue
            try:
                print(f"启动多点导航: 地图={map_id}, 路径={path_id}")
                await client.start_multi_nav(map_id, path_id)
                print("✅ 导航已启动")
                await print_nav_status(client)
            except Exception as e:
                print(f"❌ 导航启动失败: {e}")

        elif op == "3":
            try:
                x = float(input("X (米): "))
                y = float(input("Y (米): "))
                z = float(input("Z (通常0): ") or "0")
                ori = input("四元数 x,y,z,w (回车跳过): ").strip()
                orientation = None
                if ori:
                    parts = ori.split(",")
                    if len(parts) == 4:
                        orientation = {
                            "x": float(parts[0]),
                            "y": float(parts[1]),
                            "z": float(parts[2]),
                            "w": float(parts[3]),
                        }
                    else:
                        print("格式错误，使用默认方向")
                await client.start_nav({"x": x, "y": y, "z": z}, orientation)
                print("✅ 单点导航已启动")
                await print_nav_status(client)
            except ValueError:
                print("❌ 坐标必须是数字")
            except Exception as e:
                print(f"❌ 单点导航失败: {e}")

        elif op == "4":
            try:
                await client.stop_nav()
                print("✅ 已发送停止导航指令")
                await print_nav_status(client)
            except Exception as e:
                print(f"❌ 停止失败: {e}")

        elif op == "5":
            try:
                await client.start_nav_return_home()
                print("✅ 返航指令已发送")
                await print_nav_status(client)
            except Exception as e:
                print(f"❌ 返航失败: {e}")

        elif op == "6":
            await print_nav_status(client)

        elif op == "7":
            await print_loc_status(client)

        elif op == "8":
            try:
                status = await client.get_mapping_status()
                print(f"🗺️ 建图状态: {status}")
            except Exception as e:
                print(f"❌ 获取建图状态失败: {e}")

        elif op == "9":
            try:
                maps = await client.get_all_pgm_maps()
                if maps:
                    print("📂 已有地图列表:")
                    for m in maps.keys():
                        print(f"  - {m}")
                else:
                    print("未找到任何地图")
            except Exception as e:
                print(f"❌ 获取地图列表失败: {e}")

        elif op == "10":
            try:
                await client.pause_nav()
                print("⏸️ 已发送暂停导航指令")
                await print_nav_status(client)
            except Exception as e:
                print(f"❌ 暂停失败: {e}")

        elif op == "11":
            try:
                await client.continue_nav()
                print("▶️ 已发送继续导航指令")
                await print_nav_status(client)
            except Exception as e:
                print(f"❌ 继续失败: {e}")

        elif op == "12":
            req_str = input("输入 req_func JSON (例如 {\"get_nav_status\": null}): ").strip()
            try:
                req_func = json.loads(req_str)
                await client.send_raw_request(req_func)
            except Exception as e:
                print(f"❌ 请求失败: {e}")

        # ========== 新增功能实现 ==========
        elif op == "13":
            new_map = input("请输入新的地图ID (留空则不修改): ").strip()
            if new_map:
                map_id = new_map
            new_path = input("请输入新的路径ID (留空则不修改，输入空字符串可清空): ").strip()
            # 允许用户输入空字符串来清空 path_id (输入空字符串表示清空)
            if new_path == "":
                path_id = ""
            elif new_path:
                path_id = new_path
            # 如果用户直接回车（空字符串）且新路径ID为空字符串，则 path_id 变为空
            # 如果用户只修改了地图，路径保持不变
            print(f"\n✅ 已更新：地图={map_id}, 路径={path_id if path_id else '(未提供)'}")

        elif op == "14":
            print(f"\n📌 当前地图ID: {map_id}")
            print(f"📌 当前路径ID: {path_id if path_id else '(未提供)'}")
        # =================================

        elif op == "0":
            print("退出程序")
            break
        else:
            print("无效选项")


async def main():
    # 根据实际机器人修改 IP
    #robot_uri = "ws://192.168.144.100:10010"  # 铜锤
    robot_uri = "ws://192.168.234.234:10010"  # 钢蹦
    client = RobotWebSocketClient(robot_uri)

    try:
        print(f"正在连接 {robot_uri} ...")
        await client.connect()
        print("✅ 连接成功！")
        await interactive_menu(client)
    except Exception as e:
        print(f"❌ 连接或运行出错: {e}")
    finally:
        await client.disconnect()
        print("连接已关闭。")

if __name__ == "__main__":
    asyncio.run(main())