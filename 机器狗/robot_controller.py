#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import contextlib
import subprocess
import threading
import queue
import logging
import json
import asyncio
import time
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from logging.handlers import RotatingFileHandler

import websockets

try:
    import cv2
    from PIL import Image, ImageOps, ImageTk
except ImportError:
    cv2 = None
    Image = None
    ImageOps = None
    ImageTk = None

# ============================================================
# 常量
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOG_IP = "192.168.234.1"
DEFAULT_SEND_PORT = "8081"
DEFAULT_RECV_PORT = "8080"
DEFAULT_WS_URI = "ws://192.168.234.234:10010"
DEFAULT_RTSP_URL = "rtsp://192.168.234.1:8554/test"
DEFAULT_EXE_PATH = str(BASE_DIR / "zsibot_control")
LOG_FILE = BASE_DIR / "robot_controller.log"
STATUS_REFRESH_INTERVAL = 5000  # ms
VIDEO_REFRESH_INTERVAL = 33  # ms

CMD_LABELS = {
    "sdk_mode": "SDK模式",
    "remote_mode": "Remote模式",
    "roamerx_mode": "RoamerX模式",
    "forward": "向前",
    "backward": "向后",
    "turn_left": "左转",
    "turn_right": "右转",
    "move_left": "左移",
    "move_right": "右移",
    "stop_move": "停止",
    "emergency_stop": "急停",
    "sit_down": "匍匐",
    "stand_up": "站立",
    "get_status": "查询状态",
}

SPEED_LEVEL_LABELS = {
    0: "无",
    1: "慢",
    2: "中",
    3: "快",
}

FUNCTION_MODE_LABELS = {
    0: "无",
    1: "遥控",
    2: "跟踪",
    3: "SDK",
    4: "通用SDK",
    5: "Roamer",
}

CONTROL_MODE_LABELS = {
    0: "无",
    1: "站立",
    2: "匍匐",
    3: "锁定",
    4: "急停",
    5: "运动",
    6: "平衡站立",
    7: "初始化",
}

MOTION_MODE_LABELS = {
    0: "无",
    1: "运行",
    2: "休息",
    3: "实验室限制",
}

MOTION_TYPE_LABELS = {
    0: "无",
    1: "跳跃",
    2: "前跳",
    3: "后空翻",
    4: "打招呼",
    5: "双腿站立",
    6: "禁止",
    7: "未知",
    8: "卸货下蹲",
}

MODEL_LABELS = {
    0: "XG",
    1: "XGW",
    2: "XGWHSPD",
}


def _compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_status_payload(payload):
    if not isinstance(payload, dict):
        return {}

    status = dict(payload)
    battery = status.pop("battery", None)
    if isinstance(battery, dict):
        status["battery_current"] = battery.get("current")
        status["battery_error"] = battery.get("error")
        status["battery_power"] = battery.get("power")
        status["battery_temp"] = battery.get("temp")
        status["battery_volt"] = battery.get("volt")

    version = status.pop("version", None)
    if isinstance(version, dict):
        status["version_mc"] = version.get("mc_version")
        status["version_task"] = version.get("dog_task_version")

    return status


def _enum_text(value, labels):
    try:
        code = int(value)
    except (TypeError, ValueError):
        return str(value) if value is not None else "--"
    label = labels.get(code)
    if label is None:
        return str(code)
    return f"{code} ({label})"


def _format_status_value(key, value):
    if value is None:
        return "--"
    if key == "connected":
        return "已连接" if bool(value) else "未连接"
    if key in {"power", "battery_power"}:
        return f"{value}%"
    if key in {"temperature", "battery_temp"}:
        return f"{value}°C"
    if key == "battery_volt":
        return f"{value}V"
    if key == "battery_current":
        return f"{value}A"
    if key == "speed":
        return f"{value} m/s"
    if key in {"angle_speed", "shift_speed"}:
        return f"{value} rad/s"
    if key == "angle":
        return f"{value} rad"
    if key == "speed_level":
        return _enum_text(value, SPEED_LEVEL_LABELS)
    if key == "function_mode":
        return _enum_text(value, FUNCTION_MODE_LABELS)
    if key == "control_mode":
        return _enum_text(value, CONTROL_MODE_LABELS)
    if key == "motion_mode":
        return _enum_text(value, MOTION_MODE_LABELS)
    if key == "motion_type":
        return _enum_text(value, MOTION_TYPE_LABELS)
    if key == "model":
        return _enum_text(value, MODEL_LABELS)
    return str(value)


# ============================================================
# 日志: 输出到 tkinter Text 控件
# ============================================================
class TextWidgetHandler(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        try:
            msg = self.format(record)
            self.widget.after(0, self._append, msg)
        except Exception:
            pass

    def _append(self, msg):
        if not self.widget.winfo_exists():
            return
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, msg + "\n")
        self.widget.see(tk.END)
        self.widget.configure(state="disabled")


# ============================================================
# WebSocket 导航客户端 (集成自 nav-05.py)
# ============================================================
class RobotWebSocketClient:
    def __init__(self, uri, max_msg_size=50 * 1024 * 1024):
        self.uri = uri
        self.max_msg_size = max_msg_size
        self.websocket = None
        self._receive_task = None
        self._counter = 0
        self._pending = {}
        self._lock = asyncio.Lock()

    async def connect(self):
        self.websocket = await websockets.connect(self.uri, max_size=self.max_msg_size)
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._receive_task
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self._receive_task = None

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
        except Exception as exc:
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(exc)
        finally:
            if self._pending:
                exc = ConnectionError("WebSocket连接已关闭")
                for future in list(self._pending.values()):
                    if not future.done():
                        future.set_exception(exc)
            self._pending.clear()

    async def _send_request(self, req_func, timeout=30.0):
        if not self.websocket:
            raise RuntimeError("WebSocket未连接")
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
            raise TimeoutError(f"请求超时")

        req_result = resp.get("data", {}).get("req_result", {})
        if "AppResponse" in req_result:
            req_result = req_result["AppResponse"]
            resp["data"]["req_result"] = req_result

        if req_result.get("status") == "error":
            raise Exception(f"请求失败: {req_result.get('msg', '未知错误')}")
        return resp

    async def load_localization_map(self, map_id):
        await self._send_request({"loc_load_map": map_id})

    async def get_all_pgm_maps(self):
        resp = await self._send_request({"get_all_pgm_map": None})
        return resp.get("data", {}).get("req_result", {}).get("data", {})

    async def start_multi_nav(self, map_id, path_id):
        await self._send_request({"start_multi_nav": [map_id, path_id]})

    async def stop_nav(self):
        await self._send_request({"stop_nav": None})

    async def start_nav_return_home(self):
        await self._send_request({"start_nav_return_home": None})

    async def pause_nav(self):
        return await self._send_request({"pause_nav": None})

    async def continue_nav(self):
        return await self._send_request({"continue_nav": None})

    async def get_nav_status(self):
        resp = await self._send_request({"get_nav_status": None})
        req_result = resp.get("data", {}).get("req_result", {})
        status = req_result.get("data")
        return status if status is not None else req_result.get("msg", "Unknown")

    async def get_loc_status(self):
        resp = await self._send_request({"get_loc_status": None})
        req_result = resp.get("data", {}).get("req_result", {})
        status = req_result.get("data")
        return status if status is not None else req_result.get("msg", "Unknown")

    async def get_mapping_status(self):
        resp = await self._send_request({"get_mapping_status": None})
        req_result = resp.get("data", {}).get("req_result", {})
        status = req_result.get("data")
        return status if status is not None else req_result.get("msg", "Unknown")

    async def start_nav(self, position, orientation=None):
        req = {"position": position}
        if orientation:
            req["orientation"] = orientation
        await self._send_request({"start_nav": req})

    async def send_raw_request(self, req_func):
        return await self._send_request(req_func)


# ============================================================
# 机器狗控制: 管理 C++ 守护进程
# ============================================================
@dataclass(frozen=True)
class DogConfig:
    exe_path: str
    dog_ip: str
    send_port: int
    recv_port: int


class DogController:
    def __init__(self, config, logger, on_response):
        self.config = config
        self.logger = logger
        self.on_response = on_response
        self.process = None
        self.cmd_queue = queue.Queue()
        self.worker_thread = None
        self.stderr_thread = None
        self.running = False

    def start(self):
        try:
            args = [
                self.config.exe_path,
                "--daemon",
                "--dog-ip",
                self.config.dog_ip,
                "--send-port",
                str(self.config.send_port),
                "--recv-port",
                str(self.config.recv_port),
            ]
            self.process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            self.stderr_thread = threading.Thread(target=self._stderr_worker, daemon=True)
            self.stderr_thread.start()
            self.logger.info(
                "控制程序已启动 (PID: %d, dog_ip=%s, send_port=%s, recv_port=%s)",
                self.process.pid,
                self.config.dog_ip,
                self.config.send_port,
                self.config.recv_port,
            )
            return True
        except FileNotFoundError:
            self.logger.error("找不到可执行文件: %s", self.config.exe_path)
            return False
        except Exception as e:
            self.logger.error("启动控制程序失败: %s", e)
            return False

    def stop(self):
        self.running = False
        process = self.process
        if not process:
            return

        try:
            if process.stdin:
                process.stdin.write("quit\n")
                process.stdin.flush()
        except Exception:
            pass

        try:
            process.wait(timeout=3)
        except Exception:
            try:
                process.terminate()
                process.wait(timeout=2)
            except Exception:
                process.kill()

        if self.worker_thread and self.worker_thread.is_alive() and threading.current_thread() is not self.worker_thread:
            self.worker_thread.join(timeout=1)
        if self.stderr_thread and self.stderr_thread.is_alive() and threading.current_thread() is not self.stderr_thread:
            self.stderr_thread.join(timeout=1)

        self.process = None
        self.logger.info("控制程序已停止")

    def send_command(self, cmd):
        if not self.running or not self.process:
            self.on_response(cmd, False, "控制程序未启动")
            return
        self.cmd_queue.put(cmd)

    def _parse_response(self, response):
        response = response.strip()
        if not response:
            return {"ok": False, "message": "无响应"}
        if response.startswith("{"):
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
        if response.startswith("OK"):
            message = response[2:].strip()
            return {"ok": True, "message": message or "OK"}
        if response.startswith("ERROR:"):
            return {"ok": False, "message": response[6:].strip()}
        if response.startswith("DATA:"):
            data = {}
            for pair in response[5:].split(","):
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    data[key.strip()] = value.strip()
            return {"ok": True, "data": data}
        return {"ok": False, "message": f"未知响应: {response}"}

    def _worker(self):
        while self.running:
            try:
                cmd = self.cmd_queue.get(timeout=0.5)
            except queue.Empty:
                if self.process and self.process.poll() is not None:
                    self.logger.error("控制进程意外退出 (code=%s)", self.process.returncode)
                    self.running = False
                    self.on_response("__process_died__", False, "进程意外退出")
                continue

            if cmd is None:
                continue

            try:
                process = self.process
                if not process or not process.stdin or not process.stdout:
                    raise RuntimeError("控制进程不可用")
                process.stdin.write(cmd + "\n")
                process.stdin.flush()
                response = process.stdout.readline()
                if response == "":
                    if process.poll() is not None:
                        self.logger.error("控制进程意外退出 (code=%s)", process.returncode)
                        self.running = False
                        self.on_response("__process_died__", False, "进程意外退出")
                        break
                    self.on_response(cmd, False, "无响应")
                    continue
                parsed = self._parse_response(response)
                if parsed.get("ok"):
                    data = parsed.get("data")
                    if cmd == "get_status" and isinstance(data, dict):
                        self.on_response(cmd, True, data)
                    else:
                        self.on_response(cmd, True, parsed.get("message") or data or "")
                else:
                    self.on_response(cmd, False, parsed.get("message", "执行失败"))
            except BrokenPipeError:
                self.logger.error("与控制程序的管道已断开")
                self.running = False
                self.on_response(cmd, False, "管道断开")
                break
            except Exception as e:
                self.logger.error("命令执行异常: %s", e)
                self.on_response(cmd, False, str(e))

    def _stderr_worker(self):
        process = self.process
        if not process or not process.stderr:
            return
        try:
            for line in process.stderr:
                text = line.rstrip()
                if text:
                    self.logger.error("[控制程序stderr] %s", text)
        except Exception as exc:
            self.logger.error("读取控制程序stderr失败: %s", exc)


# ============================================================
# 导航控制: 管理 WebSocket 连接
# ============================================================
class NavController:
    def __init__(self, uri, logger, on_result):
        self.uri = uri
        self.logger = logger
        self.on_result = on_result
        self.loop = None
        self.thread = None
        self.client = None
        self.connected = False
        self._ready = threading.Event()

    def start_loop(self):
        if self.thread and self.thread.is_alive():
            return
        self._ready.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("导航事件循环启动超时")

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._ready.set()
        self.loop.run_forever()
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self.loop.close()

    def connect(self):
        async def _connect():
            self.client = RobotWebSocketClient(self.uri)
            await self.client.connect()
            self.connected = True

        if not self.loop:
            raise RuntimeError("导航事件循环未启动")
        try:
            future = asyncio.run_coroutine_threadsafe(_connect(), self.loop)
            future.result(timeout=10)
            self.logger.info("导航WebSocket已连接: %s", self.uri)
            return True
        except Exception as e:
            self.logger.error("导航连接失败: %s", e)
            self.client = None
            self.connected = False
            return False

    def disconnect(self):
        async def _disconnect():
            if self.client:
                await self.client.disconnect()
            self.connected = False
            self.client = None

        if self.loop and self.connected:
            try:
                future = asyncio.run_coroutine_threadsafe(_disconnect(), self.loop)
                future.result(timeout=5)
            except Exception as exc:
                self.logger.warning("断开导航连接时出现异常: %s", exc)
            finally:
                self.connected = False
                self.client = None
                self.logger.info("导航连接已断开")

    def shutdown(self):
        try:
            self.disconnect()
        finally:
            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread and self.thread.is_alive() and threading.current_thread() is not self.thread:
                self.thread.join(timeout=2)
            self.connected = False
            self.client = None
            self.loop = None
            self.thread = None

    def _submit(self, coro, action_name=""):
        if not self.connected or not self.client:
            self.on_result(action_name, False, "导航未连接")
            return
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)

        def _done(done_future):
            try:
                result = done_future.result()
                self.on_result(action_name, True, result if result is not None else "OK")
            except Exception as exc:
                self.on_result(action_name, False, str(exc))

        future.add_done_callback(_done)

    def load_map(self, map_id):
        self._submit(self.client.load_localization_map(map_id), "加载地图")

    def get_maps(self):
        self._submit(self.client.get_all_pgm_maps(), "获取地图列表")

    def start_multi_nav(self, map_id, path_id):
        self._submit(self.client.start_multi_nav(map_id, path_id), "多点导航")

    def start_single_nav(self, x, y, z, orientation=None):
        pos = {"x": x, "y": y, "z": z}
        self._submit(self.client.start_nav(pos, orientation), "单点导航")

    def stop_nav(self):
        self._submit(self.client.stop_nav(), "停止导航")

    def return_home(self):
        self._submit(self.client.start_nav_return_home(), "返航")

    def pause_nav(self):
        self._submit(self.client.pause_nav(), "暂停导航")

    def continue_nav(self):
        self._submit(self.client.continue_nav(), "继续导航")

    def get_nav_status(self):
        self._submit(self.client.get_nav_status(), "导航状态")

    def get_loc_status(self):
        self._submit(self.client.get_loc_status(), "定位状态")

    def get_mapping_status(self):
        self._submit(self.client.get_mapping_status(), "建图状态")

    def send_raw_request(self, req_func):
        self._submit(self.client.send_raw_request(req_func), "自定义请求")


# ============================================================
# RTSP 视频流: 管理机器狗视频画面
# ============================================================
class RtspVideoStream:
    def __init__(self, logger):
        self.logger = logger
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = None
        self._run_id = 0
        self._running = False
        self._url = ""
        self._latest_frame = None
        self._frame_index = 0
        self._status = "视频已停止"

    def start(self, url):
        url = (url or "").strip()
        self.stop()
        if not url:
            self._set_status("请输入RTSP地址")
            return False
        if cv2 is None or Image is None or ImageTk is None:
            self._set_status("视频功能依赖未安装")
            self.logger.warning("视频功能不可用：缺少OpenCV或Pillow")
            return False

        with self._lock:
            self._run_id += 1
            run_id = self._run_id
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._running = True
            self._url = url
            self._status = "正在连接视频流..."
            self._latest_frame = None
            self._frame_index += 1

        self._thread = threading.Thread(
            target=self._worker, args=(run_id, stop_event, url), daemon=True
        )
        self._thread.start()
        self.logger.info("开始连接视频流: %s", url)
        return True

    def stop(self):
        with self._lock:
            self._run_id += 1
            stop_event = self._stop_event
            self._stop_event = None
            self._running = False
            self._latest_frame = None
            self._frame_index += 1
            self._status = "视频已停止"
        if stop_event:
            stop_event.set()
        self._thread = None

    def snapshot(self):
        with self._lock:
            return self._frame_index, self._latest_frame, self._status, self._running, self._url

    def _set_status(self, status):
        with self._lock:
            self._status = status
            self._frame_index += 1

    def _set_status_if_current(self, run_id, status):
        with self._lock:
            if self._run_id != run_id:
                return False
            self._status = status
            self._frame_index += 1
            return True

    def _store_frame(self, run_id, image):
        with self._lock:
            if self._run_id != run_id:
                return False
            self._latest_frame = image
            self._frame_index += 1
            return True

    def _open_capture(self, url):
        if cv2 is None:
            return None
        old_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;3000000"
        cap = None
        try:
            params = []
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000])
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
            if params:
                cap = cv2.VideoCapture(url, getattr(cv2, "CAP_FFMPEG", 0), params)
            else:
                cap = cv2.VideoCapture(url, getattr(cv2, "CAP_FFMPEG", 0))
        except Exception:
            try:
                cap = cv2.VideoCapture(url)
            except Exception as exc:
                self.logger.error("打开视频流失败: %s", exc)
        finally:
            if old_options is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old_options
        if cap is None:
            return None
        try:
            if cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return cap
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        return None

    def _worker(self, run_id, stop_event, url):
        retry_delay = 1.0
        cap = None
        try:
            while not stop_event.is_set():
                cap = self._open_capture(url)
                if cap is None:
                    self._set_status_if_current(run_id, "等待视频流连接...")
                    if stop_event.wait(retry_delay):
                        break
                    continue

                self._set_status_if_current(run_id, "视频流已连接")
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        self._set_status_if_current(run_id, "视频流断开，正在重连...")
                        break
                    try:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(rgb)
                    except Exception as exc:
                        self.logger.error("视频帧转换失败: %s", exc)
                        continue
                    self._store_frame(run_id, image)

                try:
                    cap.release()
                except Exception:
                    pass
                cap = None

                if stop_event.is_set():
                    break
                if stop_event.wait(retry_delay):
                    break
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._set_status_if_current(run_id, "视频已停止")
            with self._lock:
                if self._run_id == run_id:
                    self._running = False
                    self._stop_event = None


# ============================================================
# 主界面
# ============================================================
class RobotControllerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("机器狗综合控制面板")
        self.root.geometry("1680x1020")
        self.root.minsize(1500, 860)

        self.logger = logging.getLogger("robot_controller")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        fh = RotatingFileHandler(str(LOG_FILE), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)

        self.dog_ctrl = None
        self.nav_ctrl = None
        self._video_stream = RtspVideoStream(self.logger)
        self._status_data = {}
        self._auto_refresh_var = tk.BooleanVar(value=False)
        self._video_visible_var = tk.BooleanVar(value=True)
        self._rtsp_url_var = tk.StringVar(value=DEFAULT_RTSP_URL)
        self._nav_connecting = False
        self._dog_starting = False
        self._dog_stopping = False
        self._closing = False
        self._closed = False
        self._video_photo = None
        self._video_rendered_index = -1
        self._video_status_text = ""

        self._build_ui()

        self.text_handler = TextWidgetHandler(self.log_text)
        self.text_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(self.text_handler)
        self.logger.propagate = False

        self.logger.info("=" * 40)
        self.logger.info("系统启动")
        self.logger.info("=" * 40)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------------------------------------
    # UI 构建
    # ----------------------------------------------------------
    def _build_ui(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Microsoft YaHei", 12, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Microsoft YaHei", 10, "bold"))
        style.configure("Big.TButton", font=("Microsoft YaHei", 10), padding=4)

        top_frame = ttk.Frame(self.root, padding=5)
        top_frame.pack(fill=tk.X)
        self._build_settings_bar(top_frame)

        main_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        content_frame = ttk.Frame(main_paned)
        main_paned.add(content_frame, weight=4)

        content_paned = ttk.PanedWindow(content_frame, orient=tk.HORIZONTAL)
        content_paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(content_paned)
        content_paned.add(left_frame, weight=3)
        self._build_left_panel(left_frame)

        right_frame = ttk.Frame(content_paned)
        content_paned.add(right_frame, weight=2)
        self._build_right_panel(right_frame)

        log_frame = ttk.Frame(main_paned)
        main_paned.add(log_frame, weight=2)
        self._build_log_panel(log_frame)

        self._sync_video_panel_visibility()
        self._refresh_video_panel()

    def _build_settings_bar(self, parent):
        row0 = ttk.Frame(parent)
        row0.pack(fill=tk.X, pady=(0, 3))

        ttk.Label(row0, text="可执行文件:").pack(side=tk.LEFT)
        self._exe_var = tk.StringVar(value=DEFAULT_EXE_PATH)
        ttk.Entry(row0, textvariable=self._exe_var, width=35).pack(side=tk.LEFT, padx=3)
        ttk.Button(row0, text="浏览", command=self._browse_exe).pack(side=tk.LEFT)

        ttk.Separator(row0, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(row0, text="机器狗IP:").pack(side=tk.LEFT)
        self._dog_ip_var = tk.StringVar(value=DEFAULT_DOG_IP)
        ttk.Entry(row0, textvariable=self._dog_ip_var, width=16).pack(side=tk.LEFT, padx=3)

        ttk.Label(row0, text="发送端口:").pack(side=tk.LEFT, padx=(8, 0))
        self._send_port_var = tk.StringVar(value=DEFAULT_SEND_PORT)
        ttk.Entry(row0, textvariable=self._send_port_var, width=7).pack(side=tk.LEFT, padx=3)

        ttk.Label(row0, text="接收端口:").pack(side=tk.LEFT, padx=(8, 0))
        self._recv_port_var = tk.StringVar(value=DEFAULT_RECV_PORT)
        ttk.Entry(row0, textvariable=self._recv_port_var, width=7).pack(side=tk.LEFT, padx=3)

        row1 = ttk.Frame(parent)
        row1.pack(fill=tk.X)

        ttk.Label(row1, text="导航WS:").pack(side=tk.LEFT)
        self._ws_uri_var = tk.StringVar(value=DEFAULT_WS_URI)
        ttk.Entry(row1, textvariable=self._ws_uri_var, width=38).pack(side=tk.LEFT, padx=3)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._btn_start_dog = ttk.Button(
            row1, text="启动控制程序", command=self._start_dog
        )
        self._btn_start_dog.pack(side=tk.LEFT, padx=2)
        self._btn_stop_dog = ttk.Button(
            row1, text="停止控制程序", command=self._stop_dog, state=tk.DISABLED
        )
        self._btn_stop_dog.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self._btn_ws_connect = ttk.Button(
            row1, text="连接导航", command=self._ws_connect
        )
        self._btn_ws_connect.pack(side=tk.LEFT, padx=2)
        self._btn_ws_disconnect = ttk.Button(
            row1, text="断开导航", command=self._ws_disconnect, state=tk.DISABLED
        )
        self._btn_ws_disconnect.pack(side=tk.LEFT, padx=2)

        row2 = ttk.Frame(parent)
        row2.pack(fill=tk.X, pady=(3, 0))

        ttk.Label(row2, text="RTSP地址:").pack(side=tk.LEFT)
        rtsp_entry = ttk.Entry(row2, textvariable=self._rtsp_url_var, width=48)
        rtsp_entry.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)
        rtsp_entry.bind("<Return>", lambda _e: self._apply_video_settings())

        ttk.Checkbutton(
            row2,
            text="显示视频流",
            variable=self._video_visible_var,
            command=self._toggle_video_panel,
        ).pack(side=tk.LEFT, padx=(8, 2))

        ttk.Button(row2, text="应用", command=self._apply_video_settings).pack(side=tk.LEFT, padx=2)

    def _build_left_panel(self, parent):
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_dog = ttk.Frame(notebook, padding=5)
        notebook.add(tab_dog, text="  机器狗控制  ")
        self._build_dog_tab(tab_dog)

        tab_nav = ttk.Frame(notebook, padding=5)
        notebook.add(tab_nav, text="  导航控制  ")
        self._build_nav_tab(tab_nav)

    def _build_dog_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        self._build_dog_controls(frame)

    def _build_dog_controls(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X)

        mode_frame = ttk.LabelFrame(row, text="模式切换", style="Section.TLabelframe", padding=5)
        mode_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        for cmd, label in [
            ("sdk_mode", "SDK模式"),
            ("remote_mode", "Remote模式"),
            ("roamerx_mode", "RoamerX模式"),
        ]:
            btn = ttk.Button(
                mode_frame, text=label, style="Big.TButton",
                command=lambda c=cmd, l=label: self._send_dog_cmd(c, l),
            )
            btn.pack(fill=tk.X, pady=2)

        move_frame = ttk.LabelFrame(row, text="运动控制", style="Section.TLabelframe", padding=5)
        move_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        grid = ttk.Frame(move_frame)
        grid.pack()

        btn_cfg = {"style": "Big.TButton", "width": 8}
        ttk.Button(grid, text="↑ 前", command=lambda: self._send_dog_cmd("forward", "向前"), **btn_cfg).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(grid, text="← 左转", command=lambda: self._send_dog_cmd("turn_left", "左转"), **btn_cfg).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(grid, text="■ 停止", command=lambda: self._send_dog_cmd("stop_move", "停止"), **btn_cfg).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(grid, text="右转 →", command=lambda: self._send_dog_cmd("turn_right", "右转"), **btn_cfg).grid(row=1, column=2, padx=2, pady=2)
        ttk.Button(grid, text="↓ 后", command=lambda: self._send_dog_cmd("backward", "向后"), **btn_cfg).grid(row=2, column=1, padx=2, pady=2)

        ttk.Separator(move_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        row2 = ttk.Frame(move_frame)
        row2.pack()
        ttk.Button(row2, text="左移", command=lambda: self._send_dog_cmd("move_left", "左移"), **btn_cfg).pack(side=tk.LEFT, padx=2)
        ttk.Button(row2, text="右移", command=lambda: self._send_dog_cmd("move_right", "右移"), **btn_cfg).pack(side=tk.LEFT, padx=2)

        ttk.Separator(move_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        row3 = ttk.Frame(move_frame)
        row3.pack(fill=tk.X)
        for cmd, label in [
            ("emergency_stop", "急停"),
            ("sit_down", "匍匐"),
            ("stand_up", "站立"),
        ]:
            ttk.Button(
                row3, text=label, style="Big.TButton",
                command=lambda c=cmd, l=label: self._send_dog_cmd(c, l),
            ).pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        status_frame = ttk.LabelFrame(row, text="状态查询", style="Section.TLabelframe", padding=5)
        status_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        ttk.Button(
            status_frame, text="刷新状态", style="Big.TButton",
            command=lambda: self._send_dog_cmd("get_status", "查询状态"),
        ).pack(fill=tk.X, pady=2)

        ttk.Checkbutton(
            status_frame, text=f"自动刷新 ({STATUS_REFRESH_INTERVAL // 1000}s)",
            variable=self._auto_refresh_var, command=self._toggle_auto_refresh,
        ).pack(pady=5)

    def _build_nav_tab(self, parent):
        top = ttk.LabelFrame(parent, text="导航参数", style="Section.TLabelframe", padding=5)
        top.pack(fill=tk.X, pady=(0, 5))

        param_row = ttk.Frame(top)
        param_row.pack(fill=tk.X)

        ttk.Label(param_row, text="地图ID:").pack(side=tk.LEFT)
        self._map_id_var = tk.StringVar()
        ttk.Entry(param_row, textvariable=self._map_id_var, width=25).pack(side=tk.LEFT, padx=5)

        ttk.Label(param_row, text="路径ID:").pack(side=tk.LEFT, padx=(15, 0))
        self._path_id_var = tk.StringVar()
        ttk.Entry(param_row, textvariable=self._path_id_var, width=25).pack(side=tk.LEFT, padx=5)

        btn_frame = ttk.LabelFrame(parent, text="导航控制", style="Section.TLabelframe", padding=5)
        btn_frame.pack(fill=tk.X, pady=(0, 5))

        nav_buttons = [
            ("加载定位地图", self._nav_load_map),
            ("多点导航", self._nav_start_multi),
            ("单点导航", self._nav_start_single),
            ("停止导航", self._nav_stop),
            ("返航", self._nav_return_home),
            ("暂停导航", self._nav_pause),
            ("继续导航", self._nav_continue),
        ]
        grid = ttk.Frame(btn_frame)
        grid.pack(fill=tk.X)
        for i, (label, cmd) in enumerate(nav_buttons):
            r, c = divmod(i, 4)
            ttk.Button(grid, text=label, style="Big.TButton", command=cmd).grid(
                row=r, column=c, padx=3, pady=3, sticky="ew"
            )
            grid.columnconfigure(c, weight=1)

        coord_frame = ttk.LabelFrame(parent, text="单点导航坐标", style="Section.TLabelframe", padding=5)
        coord_frame.pack(fill=tk.X, pady=(0, 5))

        crow = ttk.Frame(coord_frame)
        crow.pack(fill=tk.X)
        ttk.Label(crow, text="X:").pack(side=tk.LEFT)
        self._nav_x_var = tk.StringVar(value="0")
        ttk.Entry(crow, textvariable=self._nav_x_var, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Label(crow, text="Y:").pack(side=tk.LEFT, padx=(10, 0))
        self._nav_y_var = tk.StringVar(value="0")
        ttk.Entry(crow, textvariable=self._nav_y_var, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Label(crow, text="Z:").pack(side=tk.LEFT, padx=(10, 0))
        self._nav_z_var = tk.StringVar(value="0")
        ttk.Entry(crow, textvariable=self._nav_z_var, width=8).pack(side=tk.LEFT, padx=3)

        orow = ttk.Frame(coord_frame)
        orow.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(orow, text="四元数 (x,y,z,w, 可选):").pack(side=tk.LEFT)
        self._nav_ori_var = tk.StringVar()
        ttk.Entry(orow, textvariable=self._nav_ori_var, width=30).pack(side=tk.LEFT, padx=5)

        query_frame = ttk.LabelFrame(parent, text="状态查询", style="Section.TLabelframe", padding=5)
        query_frame.pack(fill=tk.X, pady=(0, 5))

        qgrid = ttk.Frame(query_frame)
        qgrid.pack(fill=tk.X)
        query_buttons = [
            ("导航状态", self._nav_query_status),
            ("定位状态", self._nav_query_loc),
            ("建图状态", self._nav_query_mapping),
            ("获取地图列表", self._nav_get_maps),
        ]
        for i, (label, cmd) in enumerate(query_buttons):
            r, c = divmod(i, 4)
            ttk.Button(qgrid, text=label, style="Big.TButton", command=cmd).grid(
                row=r, column=c, padx=3, pady=3, sticky="ew"
            )
            qgrid.columnconfigure(c, weight=1)

        raw_frame = ttk.LabelFrame(parent, text="自定义请求", style="Section.TLabelframe", padding=5)
        raw_frame.pack(fill=tk.X, pady=(0, 5))
        raw_row = ttk.Frame(raw_frame)
        raw_row.pack(fill=tk.X)
        self._nav_raw_req_var = tk.StringVar(value='{"get_nav_status":null}')
        ttk.Entry(raw_row, textvariable=self._nav_raw_req_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(raw_row, text="发送请求", style="Big.TButton", command=self._nav_send_raw_request).pack(side=tk.LEFT)

    def _build_right_panel(self, parent):
        self._video_frame = ttk.LabelFrame(parent, text="视频流", style="Section.TLabelframe", padding=5)
        self._video_frame.pack(fill=tk.X, pady=(0, 5))

        self._video_status_var = tk.StringVar(value="视频已停止")
        ttk.Label(self._video_frame, textvariable=self._video_status_var).pack(anchor=tk.W)

        self._video_canvas = tk.Canvas(
            self._video_frame,
            width=640,
            height=360,
            bg="black",
            highlightthickness=0,
        )
        self._video_canvas.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        self._video_canvas.bind("<Configure>", self._on_video_canvas_resize)

        self._status_frame = ttk.LabelFrame(parent, text="设备状态", style="Section.TLabelframe", padding=5)
        self._status_frame.pack(fill=tk.X, pady=(0, 5))

        self._status_labels = {}
        status_items = [
            ("connected", "连接状态"),
            ("sn", "设备序列号"),
            ("dev_name", "设备名称"),
            ("wifi_ssid", "WiFi"),
            ("power", "电量"),
            ("temperature", "温度"),
            ("battery_power", "电池电量"),
            ("battery_current", "电池电流"),
            ("battery_temp", "电池温度"),
            ("battery_volt", "电池电压"),
            ("battery_error", "电池错误"),
            ("speed_level", "速度等级"),
            ("control_mode", "控制模式"),
            ("function_mode", "功能模式"),
            ("motion_mode", "运动模式"),
            ("motion_type", "运动类型"),
            ("model", "机型"),
            ("speed", "速度(m/s)"),
            ("angle_speed", "角速度(rad/s)"),
            ("shift_speed", "平移速度(m/s)"),
            ("angle", "角度(rad)"),
            ("motor_temp0", "电机M0温度"),
            ("motor_temp1", "电机M1温度"),
            ("motor_temp2", "电机M2温度"),
            ("motor_temp3", "电机M3温度"),
            ("fault_count", "故障数"),
            ("version_mc", "主控版本"),
            ("version_task", "任务版本"),
        ]

        grid = ttk.Frame(self._status_frame)
        grid.pack(fill=tk.X)
        for i, (key, label) in enumerate(status_items):
            r, c = divmod(i, 2)
            ttk.Label(grid, text=f"{label}:").grid(row=r, column=c * 2, sticky=tk.W, padx=(5, 0), pady=1)
            val_label = ttk.Label(grid, text="--", foreground="blue")
            val_label.grid(row=r, column=c * 2 + 1, sticky=tk.W, padx=(0, 10), pady=1)
            self._status_labels[key] = val_label

    def _build_log_panel(self, parent):
        self._log_frame = ttk.LabelFrame(parent, text="日志输出", style="Section.TLabelframe", padding=5)
        self._log_frame.pack(fill=tk.BOTH, expand=True)

        btn_row = ttk.Frame(self._log_frame)
        btn_row.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(btn_row, text="清除日志", command=self._clear_log).pack(side=tk.RIGHT)

        self.log_text = scrolledtext.ScrolledText(
            self._log_frame, height=14, state="disabled",
            font=("Consolas", 9), wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _video_placeholder_text(self):
        url = self._rtsp_url_var.get().strip()
        if not url:
            return "请输入RTSP地址"
        return "正在连接视频流..."

    def _clear_video_canvas(self):
        if not hasattr(self, "_video_canvas") or not self._video_canvas.winfo_exists():
            return
        self._video_canvas.delete("all")
        self._video_canvas.configure(bg="black")
        self._video_photo = None
        self._video_rendered_index = -1

    def _draw_video_placeholder(self, text):
        if not hasattr(self, "_video_canvas") or not self._video_canvas.winfo_exists():
            return
        self._video_canvas.delete("all")
        width = max(self._video_canvas.winfo_width(), 640)
        height = max(self._video_canvas.winfo_height(), 360)
        self._video_canvas.configure(bg="black")
        self._video_canvas.create_rectangle(0, 0, width, height, fill="black", outline="black")
        self._video_canvas.create_text(
            width // 2,
            height // 2,
            text=text,
            fill="white",
            font=("Microsoft YaHei", 12),
        )
        self._video_photo = None
        self._video_rendered_index = -1

    def _render_video_frame(self, frame):
        if ImageTk is None:
            return
        if not hasattr(self, "_video_canvas") or not self._video_canvas.winfo_exists():
            return
        width = max(self._video_canvas.winfo_width(), 640)
        height = max(self._video_canvas.winfo_height(), 360)
        if ImageOps is not None:
            image = ImageOps.contain(frame, (width, height), Image.LANCZOS)
        else:
            image = frame.resize((width, height), Image.LANCZOS)
        self._video_photo = ImageTk.PhotoImage(image)
        self._video_canvas.delete("all")
        self._video_canvas.configure(bg="black")
        self._video_canvas.create_image(width // 2, height // 2, image=self._video_photo, anchor=tk.CENTER)

    def _on_video_canvas_resize(self, _event):
        if self._closed or not self._video_visible_var.get():
            return
        frame_index, frame, status, running, _url = self._video_stream.snapshot()
        if running and frame is not None:
            self._render_video_frame(frame)
            self._video_rendered_index = frame_index
        elif not running and status:
            self._draw_video_placeholder(self._video_status_text or status)

    def _sync_video_stream(self):
        if self._closing or self._closed:
            return
        if not self._video_visible_var.get():
            self._stop_video_stream()
            return

        url = self._rtsp_url_var.get().strip()
        if not url:
            self._stop_video_stream()
            self._video_status_text = "请输入RTSP地址"
            self._video_status_var.set(self._video_status_text)
            self._draw_video_placeholder(self._video_status_text)
            return

        _frame_index, _frame, status, running, current_url = self._video_stream.snapshot()
        if running and current_url == url:
            self._video_status_text = status
            self._video_status_var.set(status)
            return

        if self._video_stream.start(url):
            self._video_status_text = "正在连接视频流..."
            self._video_status_var.set(self._video_status_text)
            self._draw_video_placeholder(self._video_status_text)
            self._video_rendered_index = -1
        else:
            status = self._video_stream.snapshot()[2]
            self._video_status_text = status
            self._video_status_var.set(status)
            self._draw_video_placeholder(status)

    def _sync_video_panel_visibility(self):
        if not hasattr(self, "_video_frame"):
            return
        if self._video_visible_var.get():
            if not self._video_frame.winfo_manager():
                self._video_frame.pack(before=self._status_frame, fill=tk.X, pady=(0, 5))
            self._sync_video_stream()
        else:
            self._stop_video_stream()
            if self._video_frame.winfo_manager():
                self._video_frame.pack_forget()

    def _toggle_video_panel(self):
        self._sync_video_panel_visibility()

    def _apply_video_settings(self):
        self._sync_video_panel_visibility()

    def _stop_video_stream(self):
        if hasattr(self, "_video_stream"):
            self._video_stream.stop()
        self._video_status_text = "视频已停止"
        if hasattr(self, "_video_status_var"):
            self._video_status_var.set(self._video_status_text)
        self._clear_video_canvas()

    def _refresh_video_panel(self):
        if self._closed or not hasattr(self, "_video_canvas"):
            return

        if self._video_visible_var.get():
            frame_index, frame, status, running, _url = self._video_stream.snapshot()
            if running:
                if status != self._video_status_text:
                    self._video_status_text = status
                    self._video_status_var.set(status)
                    if frame is None:
                        self._draw_video_placeholder(status)
                if frame is not None and frame_index != self._video_rendered_index:
                    self._render_video_frame(frame)
                    self._video_rendered_index = frame_index

        self.root.after(VIDEO_REFRESH_INTERVAL, self._refresh_video_panel)

    # ----------------------------------------------------------
    # 机器狗控制逻辑
    # ----------------------------------------------------------
    def _browse_exe(self):
        path = filedialog.askopenfilename(title="选择可执行文件")
        if path:
            self._exe_var.set(path)

    def _parse_port(self, value, label):
        try:
            port = int(value)
        except ValueError:
            messagebox.showwarning("提示", f"{label}必须为数字")
            return None
        if not (1 <= port <= 65535):
            messagebox.showwarning("提示", f"{label}必须在1-65535之间")
            return None
        return port

    def _build_dog_config(self):
        exe = self._exe_var.get().strip()
        dog_ip = self._dog_ip_var.get().strip()
        send_port = self._parse_port(self._send_port_var.get().strip(), "发送端口")
        recv_port = self._parse_port(self._recv_port_var.get().strip(), "接收端口")
        if not exe or not dog_ip or send_port is None or recv_port is None:
            return None
        return DogConfig(exe, dog_ip, send_port, recv_port)

    def _set_dog_buttons(self, running, stopping=False, starting=False):
        starting = starting or self._dog_starting
        self._btn_start_dog.configure(state=tk.DISABLED if running or stopping or starting else tk.NORMAL)
        self._btn_stop_dog.configure(state=tk.NORMAL if running and not stopping and not starting else tk.DISABLED)

    def _stop_dog_async(self):
        if self._dog_stopping:
            return

        ctrl = self.dog_ctrl
        if not ctrl:
            self._set_dog_buttons(False)
            self._auto_refresh_var.set(False)
            if self._closing:
                self._finalize_close()
            else:
                self._sync_video_panel_visibility()
            return

        self._dog_stopping = True
        self._set_dog_buttons(True, stopping=True)
        self.logger.info("正在停止控制程序")

        def _worker():
            try:
                ctrl.stop()
            except Exception as exc:
                self.logger.error("停止控制程序失败: %s", exc)
            finally:
                def _finish():
                    if self.dog_ctrl is ctrl:
                        self.dog_ctrl = None
                    self._dog_stopping = False
                    self._set_dog_buttons(False)
                    self._auto_refresh_var.set(False)
                    if self._closing:
                        self._finalize_close()
                    else:
                        self._sync_video_panel_visibility()

                try:
                    self.root.after(0, _finish)
                except Exception:
                    _finish()

        threading.Thread(target=_worker, daemon=True).start()

    def _start_dog(self):
        if self._dog_starting:
            messagebox.showinfo("提示", "控制程序正在启动，请稍候")
            return
        if self._dog_stopping:
            messagebox.showinfo("提示", "控制程序正在停止，请稍候")
            return
        if self.dog_ctrl and self.dog_ctrl.running:
            messagebox.showinfo("提示", "控制程序已在运行")
            return
        config = self._build_dog_config()
        if not config:
            return
        if not os.path.isfile(config.exe_path):
            if not messagebox.askyesno(
                "文件不存在",
                f"文件 {config.exe_path} 不存在，是否仍然尝试启动？\n（请确保已编译: make）",
            ):
                return

        self._dog_starting = True
        self._set_dog_buttons(False, starting=True)
        self.logger.info("正在启动控制程序")

        def _worker():
            ctrl = DogController(config, self.logger, self._on_dog_response)
            ok = False
            err = None
            try:
                ok = ctrl.start()
            except Exception as exc:
                err = str(exc)
                self.logger.error("启动控制程序异常: %s", exc)

            def _finish():
                if self._closed:
                    if ok:
                        try:
                            ctrl.stop()
                        except Exception:
                            pass
                    return

                self._dog_starting = False

                if ok:
                    self.dog_ctrl = ctrl
                    self._set_dog_buttons(True)
                    self._sync_video_panel_visibility()
                    self.logger.info(
                        "机器狗连接参数: exe=%s dog_ip=%s send_port=%s recv_port=%s",
                        config.exe_path,
                        config.dog_ip,
                        config.send_port,
                        config.recv_port,
                    )
                    if self._closing:
                        self._stop_dog_async()
                else:
                    self.dog_ctrl = None
                    self._set_dog_buttons(False)
                    if err:
                        self.logger.error("启动控制程序失败: %s", err)
                    if self._closing:
                        self._finalize_close()

            try:
                self.root.after(0, _finish)
            except Exception:
                _finish()

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_dog(self):
        self._stop_dog_async()

    def _send_dog_cmd(self, cmd, label):
        if not self.dog_ctrl or not self.dog_ctrl.running:
            self.logger.warning("[%s] 控制程序未启动", label)
            return
        self.logger.info("发送指令: %s (%s)", label, cmd)
        self.dog_ctrl.send_command(cmd)

    def _parse_legacy_status(self, data_str):
        status = {}
        for pair in data_str.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                status[k.strip()] = v.strip()
        return status

    def _on_dog_response(self, cmd, success, data):
        label = CMD_LABELS.get(cmd, cmd)
        if cmd == "__process_died__":
            self.logger.error("控制进程意外退出，请重新启动")
            self.root.after(0, self._stop_dog)
            return

        if success:
            if cmd == "get_status" and data:
                if isinstance(data, str):
                    data = self._parse_legacy_status(data)
                status = _normalize_status_payload(data)
                self._status_data = status
                self.root.after(0, self._update_status_display, status)

                faults = status.get("faults", [])
                if isinstance(faults, list) and faults:
                    self.logger.warning("[查询状态] 故障数: %d", len(faults))
                    for fault in faults:
                        self.logger.warning("[故障] %s", _compact_json(fault))
                self.logger.info("[查询状态] 状态信息已更新")
            else:
                suffix = ""
                if isinstance(data, (dict, list)):
                    suffix = f": {_compact_json(data)}"
                elif data not in ("", None):
                    suffix = f": {data}"
                self.logger.info("[%s] 执行成功%s", label, suffix)
        else:
            self.logger.error("[%s] 执行失败: %s", label, data)

    def _update_status_display(self, status):
        defaults = {
            "connected": "未连接",
            "fault_count": "0",
        }
        for key, widget in self._status_labels.items():
            widget.configure(text=defaults.get(key, "--"))
        for key, value in status.items():
            if key in self._status_labels:
                self._status_labels[key].configure(text=_format_status_value(key, value))

    def _toggle_auto_refresh(self):
        if self._auto_refresh_var.get():
            self.logger.info("已开启自动刷新")
            self._auto_refresh_status()
        else:
            self.logger.info("已关闭自动刷新")

    def _auto_refresh_status(self):
        if not self._auto_refresh_var.get():
            return
        if self.dog_ctrl and self.dog_ctrl.running:
            self.dog_ctrl.send_command("get_status")
        self.root.after(STATUS_REFRESH_INTERVAL, self._auto_refresh_status)

    # ----------------------------------------------------------
    # 导航控制逻辑
    # ----------------------------------------------------------
    def _set_nav_connection_buttons(self, connected):
        self._btn_ws_connect.configure(state=tk.DISABLED if connected or self._nav_connecting else tk.NORMAL)
        self._btn_ws_disconnect.configure(state=tk.NORMAL if connected else tk.DISABLED)

    def _ws_connect(self):
        uri = self._ws_uri_var.get().strip()
        if not uri:
            messagebox.showwarning("提示", "请输入WebSocket地址")
            return

        if self._nav_connecting or (self.nav_ctrl and self.nav_ctrl.connected):
            return

        self._nav_connecting = True
        self._btn_ws_connect.configure(state=tk.DISABLED)
        self._btn_ws_disconnect.configure(state=tk.DISABLED)
        self.logger.info("正在连接导航: %s", uri)

        def _worker():
            ctrl = NavController(uri, self.logger, self._on_nav_result)
            ok = False
            err = None
            try:
                ctrl.start_loop()
                ok = ctrl.connect()
                if ok:
                    self.nav_ctrl = ctrl
                else:
                    ctrl.shutdown()
            except Exception as exc:
                err = str(exc)
                try:
                    ctrl.shutdown()
                except Exception:
                    pass

            def _finish():
                self._nav_connecting = False
                if ok:
                    self._set_nav_connection_buttons(True)
                else:
                    self.nav_ctrl = None
                    self._set_nav_connection_buttons(False)
                    if err:
                        self.logger.error("导航连接失败: %s", err)

            self.root.after(0, _finish)

        threading.Thread(target=_worker, daemon=True).start()

    def _ws_disconnect(self):
        if not self.nav_ctrl:
            self._set_nav_connection_buttons(False)
            return

        ctrl = self.nav_ctrl
        self.logger.info("正在断开导航连接")
        self._btn_ws_disconnect.configure(state=tk.DISABLED)

        def _worker():
            try:
                ctrl.shutdown()
            finally:
                if self.nav_ctrl is ctrl:
                    self.nav_ctrl = None
                self.root.after(0, self._finish_ws_disconnect)

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_ws_disconnect(self):
        self._nav_connecting = False
        self._set_nav_connection_buttons(False)

    def _require_nav_ctrl(self):
        if not self.nav_ctrl or not self.nav_ctrl.connected:
            self.logger.warning("导航未连接")
            return None
        return self.nav_ctrl

    def _nav_load_map(self):
        ctrl = self._require_nav_ctrl()
        if not ctrl:
            return
        map_id = self._map_id_var.get().strip()
        if not map_id:
            messagebox.showwarning("提示", "请输入地图ID")
            return
        ctrl.load_map(map_id)

    def _nav_start_multi(self):
        ctrl = self._require_nav_ctrl()
        if not ctrl:
            return
        map_id = self._map_id_var.get().strip()
        path_id = self._path_id_var.get().strip()
        if not map_id or not path_id:
            messagebox.showwarning("提示", "请输入地图ID和路径ID")
            return
        ctrl.start_multi_nav(map_id, path_id)

    def _nav_start_single(self):
        ctrl = self._require_nav_ctrl()
        if not ctrl:
            return
        try:
            x = float(self._nav_x_var.get())
            y = float(self._nav_y_var.get())
            z = float(self._nav_z_var.get())
        except ValueError:
            messagebox.showwarning("提示", "坐标必须为数字")
            return

        orientation = None
        ori_str = self._nav_ori_var.get().strip()
        if ori_str:
            parts = [p.strip() for p in ori_str.split(",")]
            if len(parts) != 4:
                messagebox.showwarning("提示", "四元数需要4个值: x,y,z,w")
                return
            try:
                orientation = {
                    "x": float(parts[0]),
                    "y": float(parts[1]),
                    "z": float(parts[2]),
                    "w": float(parts[3]),
                }
            except ValueError:
                messagebox.showwarning("提示", "四元数格式错误，应为: x,y,z,w")
                return

        ctrl.start_single_nav(x, y, z, orientation)

    def _nav_stop(self):
        ctrl = self._require_nav_ctrl()
        if ctrl:
            ctrl.stop_nav()

    def _nav_return_home(self):
        ctrl = self._require_nav_ctrl()
        if ctrl:
            ctrl.return_home()

    def _nav_pause(self):
        ctrl = self._require_nav_ctrl()
        if ctrl:
            ctrl.pause_nav()

    def _nav_continue(self):
        ctrl = self._require_nav_ctrl()
        if ctrl:
            ctrl.continue_nav()

    def _nav_query_status(self):
        ctrl = self._require_nav_ctrl()
        if ctrl:
            ctrl.get_nav_status()

    def _nav_query_loc(self):
        ctrl = self._require_nav_ctrl()
        if ctrl:
            ctrl.get_loc_status()

    def _nav_query_mapping(self):
        ctrl = self._require_nav_ctrl()
        if ctrl:
            ctrl.get_mapping_status()

    def _nav_get_maps(self):
        ctrl = self._require_nav_ctrl()
        if ctrl:
            ctrl.get_maps()

    def _nav_send_raw_request(self):
        ctrl = self._require_nav_ctrl()
        if not ctrl:
            return
        raw = self._nav_raw_req_var.get().strip()
        if not raw:
            messagebox.showwarning("提示", "请输入请求JSON")
            return
        try:
            req_func = json.loads(raw)
        except json.JSONDecodeError as exc:
            messagebox.showwarning("提示", f"请求JSON格式错误: {exc}")
            return
        ctrl.send_raw_request(req_func)

    def _stringify_nav_result(self, data):
        if data is None or data == "":
            return "OK"
        if isinstance(data, (dict, list)):
            return _compact_json(data)
        return str(data)

    def _on_nav_result(self, action, success, data):
        text = self._stringify_nav_result(data)
        if success:
            if action == "获取地图列表":
                if isinstance(data, dict) and data:
                    self.logger.info("[获取地图列表] %s", ", ".join(map(str, data.keys())))
                elif isinstance(data, list) and data:
                    self.logger.info("[获取地图列表] %s", ", ".join(map(str, data)))
                else:
                    self.logger.info("[获取地图列表] %s", text)
            else:
                self.logger.info("[%s] 成功: %s", action, text)
        else:
            self.logger.error("[%s] 失败: %s", action, text)

    # ----------------------------------------------------------
    # 通用
    # ----------------------------------------------------------
    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state="disabled")

    def _on_close(self):
        self.logger.info("系统关闭")
        self._auto_refresh_var.set(False)
        self._closing = True
        if self.dog_ctrl:
            if self._dog_stopping:
                return
            self._stop_dog_async()
            return
        self._finalize_close()

    def _finalize_close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_video_stream()
        if self.nav_ctrl:
            try:
                self.nav_ctrl.shutdown()
            except Exception:
                pass
            self.nav_ctrl = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    app = RobotControllerApp()
    app.run()
