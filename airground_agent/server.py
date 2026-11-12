#!/usr/bin/env python3
"""本地 Web 服务：把「空地协同」多智能体系统暴露为可交互前端。

- GET  /                -> webui/index.html
- GET  /api/config      -> 当前模型 / 工具集 / 执行通道 / 世界状态默认值
- POST /api/run         -> SSE 流式返回执行事件（planning / plan / select / step /
                           progress / replan / reflexion / result / error / end）
- POST /api/abort       -> 中止正在运行的任务（急停）
- POST /api/dog/compile -> 把一版任务规划编译为机器狗指令序列（不下发真机）

纯标准库实现，无需额外依赖；默认用 conda `agent` 环境运行（与生产测试同源）。
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
import uuid
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = os.path.dirname(os.path.abspath(__file__))
WEBUI = os.path.join(ROOT, "webui")
REAL_CONFIG = os.path.join(ROOT, "config", "real_model.yaml")
PORT = int(os.environ.get("AIRGROUND_WEB_PORT", "8777"))

sys.path.insert(0, ROOT)

from airground_agent.config.settings import Settings  # noqa: E402
from airground_agent.core.task_model import Plan, TraceStep  # noqa: E402
from airground_agent.dog import CompileConfig, compile_plan  # noqa: E402
from airground_agent.dog.frames import FrameTransform  # noqa: E402
from airground_agent.orchestrator import Orchestrator  # noqa: E402
from airground_agent.perception import build_perception_tools, load_scene  # noqa: E402
from airground_agent.tools.adapters import SimConfig  # noqa: E402
from airground_agent.tools.registry import ToolRegistry  # noqa: E402

# 正在运行的任务（用于 /api/abort 急停）
ACTIVE_RUNS: dict = {}
_RUNS_LOCK = threading.Lock()


def _mask(key: str) -> str:
    if not key:
        return ""
    return key[:6] + "..." + key[-4:] if len(key) > 12 else "***"


def build_settings(mode: str, cfg: dict):
    """根据前端请求构造 Settings。mode=mock 走离线后端；mode=real 走中转站。"""
    if mode == "mock":
        return Settings(llm={"backend": "mock"})

    s = Settings.from_file(REAL_CONFIG) if os.path.exists(REAL_CONFIG) else Settings(llm={"backend": "openai"})
    llm = s.llm
    if cfg.get("backend"):
        llm.backend = cfg["backend"]
    if cfg.get("model_name"):
        llm.model_name = cfg["model_name"]
    if cfg.get("base_url"):
        llm.base_url = cfg["base_url"]
    if cfg.get("api_key"):
        llm.api_key = cfg["api_key"]
    return s


def build_sim_config(exec_cfg: dict) -> SimConfig:
    """把前端的"执行通道"设置转成 SimConfig（缺省回落到 legacy，保证行为不变）。"""
    if not exec_cfg:
        return SimConfig()
    allowed = ("mode", "time_scale", "failure_rate", "comm_loss_rate",
               "timeout_ms", "seed", "playback_file", "uav_speed_mps",
               "dog_speed_mps", "min_battery_pct")
    data = {k: v for k, v in exec_cfg.items() if k in allowed and v is not None}
    return SimConfig(**data)


@lru_cache(maxsize=1)
def _get_scene():
    """加载真实 3DGS 场景（进程内缓存，避免每次请求重读 25MB 点云）。"""
    try:
        return load_scene()
    except Exception:
        return None


def get_config_payload() -> dict:
    scene = _get_scene()
    tools = ToolRegistry.default().list()
    if scene is not None:
        tools = tools + [t.name for t in build_perception_tools(scene)]
    real_available = os.path.exists(REAL_CONFIG)
    model_info = None
    if real_available:
        try:
            s = Settings.from_file(REAL_CONFIG)
            model_info = {
                "backend": s.llm.backend,
                "model_name": s.llm.model_name,
                "base_url": s.llm.base_url,
                "api_key_masked": _mask(s.llm.api_key),
            }
        except Exception:
            model_info = None
    return {
        "real_available": real_available,
        "model": model_info,
        "tools": tools,
        "default_ws": {
            "battery": 100,
            "safety_cleared": True,
            "edge_capable": True,
            "uav_online": True,
        },
        # 空地协同的物理执行平台（与计算层级 edge/cloud 正交）
        "platforms": [
            {"id": "uav", "label": "无人机", "color": "#38bdf8", "desc": "空中平台"},
            {"id": "dog", "label": "机器狗", "color": "#f59e0b", "desc": "地面四足平台"},
            {"id": "shared", "label": "协同算力", "color": "#a78bfa", "desc": "边侧/云端"},
        ],
        # 执行通道（ToolAdapter）：决定任务由谁真正做出来
        "exec_modes": [
            {"id": "legacy", "label": "离线 Mock", "desc": "零耗时、必然成功，仅验证算法"},
            {"id": "sim", "label": "硬件在环仿真", "desc": "真实时序 + 电量/里程/链路演化 + 故障注入"},
            {"id": "playback", "label": "录播回放", "desc": "按现场记录的真实时序复现（确定性）"},
            {"id": "ros2", "label": "ROS2 真机", "desc": "真实机器人通道（需 rclpy 与节点）"},
        ],
        "default_exec": {
            "mode": "sim",
            "time_scale": 20.0,
            "failure_rate": 0.0,
            "comm_loss_rate": 0.0,
            "timeout_ms": 120000.0,
            "seed": None,
        },
        "examples": [
            "无人机航拍变电站高空设备，机器狗巡检地面表计，融合成巡检短片并去人脸",
            "对变电站做一次空地协同巡检：无人机查绝缘子，机器狗查开关柜",
            "端侧 VLA 分别控制无人机与机器狗完成协同巡检",
        ],
        # 真实感知层：3DGS 实验室场景重建的占据栅格（前端可渲染真实地图）
        "scene": scene.to_dict() if scene is not None else None,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AirGroundWeb/0.1"

    # 关掉默认日志刷屏，保留异常可追溯
    def log_message(self, fmt, *args):  # noqa: A003
        pass

    # ---------------- 静态资源 / 配置 ----------------
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(WEBUI, "index.html"), "text/html; charset=utf-8")
        if path == "/api/config":
            return self._serve_json(get_config_payload())
        # 其它静态文件（webui 目录内）
        rel = path.lstrip("/")
        candidate = os.path.normpath(os.path.join(WEBUI, rel))
        if candidate.startswith(WEBUI) and os.path.isfile(candidate):
            ctype = self._ctype(candidate)
            return self._serve_file(candidate, ctype)
        self.send_error(404, "Not Found")

    # ---------------- 运行（SSE 流式） ----------------
    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/abort":
            return self._abort()
        if path == "/api/dog/compile":
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            except Exception as e:
                self.send_error(400, f"Bad Request: {e}")
                return
            return self._dog_compile(body)
        if path != "/api/run":
            self.send_error(404, "Not Found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            self.send_error(400, f"Bad Request: {e}")
            return
        self._stream_run(body)

    # ---------------- 实现 ----------------
    def _serve_file(self, fpath, ctype):
        try:
            with open(fpath, "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_json(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _ctype(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
            ".json": "application/json; charset=utf-8",
            ".ico": "image/x-icon",
        }.get(ext, "application/octet-stream")

    # ---------------- 机器狗指令编译（不下发真机） ----------------
    def _dog_compile(self, body: dict):
        """把一版任务规划编译成机器狗能吃的数据形式。

        body: {"plan": <Plan JSON>, "trace": [TraceStep...], "dog": {编译/标定参数}}
        返回: {"ok":true, "mission": {...}, "summary": {...}, "warnings": [...]}
        """
        plan_raw = body.get("plan")
        if not plan_raw:
            return self._serve_json({"ok": False, "error": "缺少 plan"})
        try:
            plan = Plan.model_validate(plan_raw)
        except Exception as e:  # noqa: BLE001
            return self._serve_json({"ok": False, "error": f"plan 解析失败: {e}"})
        try:
            trace = [TraceStep.model_validate(t) for t in (body.get("trace") or [])]
        except Exception:  # noqa: BLE001 - trace 只是航点来源，缺失不致命
            trace = []

        d = body.get("dog") or {}
        cfg = CompileConfig(
            map_id=str(d.get("map_id", "") or ""),
            path_id=str(d.get("path_id", "") or ""),
            nav_mode=str(d.get("nav_mode", "auto") or "auto"),
            frame=FrameTransform(
                yaw_deg=float(d.get("yaw_deg", 0.0) or 0.0),
                translate=list(d.get("translate") or [0.0, 0.0, 0.0]),
                scale=float(d.get("scale", 1.0) or 1.0),
                flip_x=bool(d.get("flip_x", False)),
                flip_y=bool(d.get("flip_y", False)),
            ),
            waypoint_stride_m=float(d.get("waypoint_stride_m", 1.0) or 1.0),
            max_waypoints=int(d.get("max_waypoints", 40) or 40),
            heading_mode=str(d.get("heading_mode", "along_path") or "along_path"),
            fixed_yaw_deg=float(d.get("fixed_yaw_deg", 0.0) or 0.0),
            nav_timeout_s=float(d.get("nav_timeout_s", 120.0) or 120.0),
            collect_seconds=float(d.get("collect_seconds", 3.0) or 3.0),
            infer_approach=bool(d.get("infer_approach", False)),
        )
        try:
            mission = compile_plan(plan, trace, cfg=cfg, scene=_get_scene(),
                                   goal=plan.root_goal)
        except Exception as e:  # noqa: BLE001
            return self._serve_json({"ok": False, "error": f"编译失败: {e}"})
        return self._serve_json({
            "ok": True,
            "mission": mission.to_dict(),
            "summary": mission.summary(),
            "warnings": mission.warnings,
            "problems": mission.validate(),
            "nav_frames": mission.nav_frames(),
            "daemon_lines": mission.daemon_lines(),
        })

    def _abort(self):
        """急停：中止正在运行的任务。"""
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            body = {}
        run_id = body.get("run_id") or ""
        with _RUNS_LOCK:
            orch = ACTIVE_RUNS.get(run_id)
        if orch is None:
            return self._serve_json({"ok": False, "error": "run_not_found", "run_id": run_id})
        orch.abort("用户中止")
        return self._serve_json({"ok": True, "run_id": run_id})

    def _stream_run(self, body: dict):
        goal = (body.get("goal") or "").strip()
        mode = body.get("mode", "real")
        ws_init = body.get("ws_init") or None
        cfg = body.get("cfg") or {}
        exec_cfg = body.get("exec") or {}
        run_id = f"run-{uuid.uuid4().hex[:8]}"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # 流式期间保持长连接；流结束后由下方 close_connection 显式关闭，
        # 否则客户端（curl / 浏览器 fetch reader）会一直等不到 done。
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        evq: "queue.Queue" = queue.Queue()

        def on_event(etype: str, payload: dict):
            evq.put((etype, payload))

        settings = build_settings(mode, cfg)
        sim_cfg = build_sim_config(exec_cfg)
        # 执行通道随本次运行构建（每次运行独立，避免仿真世界状态互相污染）
        orch = Orchestrator(settings, on_event=on_event, sim=sim_cfg)
        with _RUNS_LOCK:
            ACTIVE_RUNS[run_id] = orch

        holder: dict = {}
        err_holder: dict = {}

        def worker():
            try:
                holder["res"] = orch.run(goal, ws_init)
            except Exception:  # noqa: BLE001
                err_holder["err"] = traceback.format_exc()
            finally:
                with _RUNS_LOCK:
                    ACTIVE_RUNS.pop(run_id, None)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        def send(etype: str, payload: dict):
            try:
                self.wfile.write(f"event: {etype}\n".encode("utf-8"))
                self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        send("start", {"goal": goal, "mode": mode, "run_id": run_id,
                       "exec": {"mode": sim_cfg.mode, "time_scale": sim_cfg.time_scale,
                                "failure_rate": sim_cfg.failure_rate,
                                "comm_loss_rate": sim_cfg.comm_loss_rate,
                                "timeout_ms": sim_cfg.timeout_ms},
                       "adapter": orch.adapter_health()})
        while True:
            try:
                item = evq.get(timeout=0.5)
            except queue.Empty:
                if not t.is_alive():
                    break
                continue
            etype, payload = item
            send(etype, payload)

        if err_holder.get("err"):
            send("error", {"detail": err_holder["err"]})
        elif holder.get("res") is not None:
            try:
                payload = holder["res"].model_dump(mode="json")
                # 附加执行通道与真机遥测汇总，便于前端做"算法 vs 真机节奏"对比
                payload["_exec"] = {"mode": sim_cfg.mode, "time_scale": sim_cfg.time_scale}
                payload["_adapter"] = orch.adapter_health()
                # 顺带编译出机器狗指令序列（纯编译，不下发）：前端可直接查看/导出
                try:
                    mission = orch.to_dog_mission(holder["res"])
                    payload["_dog_mission"] = mission.to_dict()
                except Exception as e:  # noqa: BLE001 - 编译失败不应影响主结果
                    payload["_dog_mission"] = {"error": str(e)}
                send("result", payload)
            except Exception:  # noqa: BLE001
                send("error", {"detail": traceback.format_exc()})
        send("end", {})
        # 显式关闭，让客户端的 reader 立即收到 done（否则会挂到超时）
        self.close_connection = True


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    cfg = get_config_payload()
    model = cfg.get("model") or {}
    print("=" * 60)
    print(" 空地协同 · 多智能体任务分配与边缘智能决策系统 — Web 控制台")
    print("=" * 60)
    print(f" 访问地址 : http://127.0.0.1:{PORT}")
    print(f" 真实模型 : {'可用' if cfg['real_available'] else '未配置'} "
          f"({model.get('model_name') or '-'} @ {model.get('base_url') or '-'})")
    print(f" 工具集   : {', '.join(cfg['tools'])}")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.shutdown()


if __name__ == "__main__":
    main()
