"""虚拟狗：模仿 ZSIBOT alg_control_node 的导航 WebSocket 服务（本地测试用，非真机）。

严格复刻 `nav-05.py` / `transport.NavClient` 的帧协议：
- 客户端发 `app_req`（`head.type=app_req`, `frame_count=n`, `data.req_func`）
- 服务端回 `app_resp`（`head.type=app_resp`, 同 `frame_count`, `data.req_result.AppResponse`）

内置一个最小状态机：`StandBy`（空闲）/ `Navigating`（导航中，sim_duration 秒后自动"到站"回到 StandBy）。
这样 `DogMissionRunner.wait_nav_idle` 能真实地轮询到"已抵达"，而不是假成功。

用法：
    srv = VirtualDogWsServer(port=14567)
    srv.start()          # 后台线程起服务
    uri = srv.uri        # ws://127.0.0.1:14567
    ... 把 uri 喂给 NavClient / DogTransport ...
    srv.received         # 可审查收到的 req_func 列表
    srv.stop()
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional

try:  # 与 transport.py 一致：websockets 为可选依赖
    import websockets  # type: ignore
    _WS_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    websockets = None  # type: ignore
    _WS_AVAILABLE = False


class VirtualDogWsServer:
    """本地虚拟狗导航服务：复刻狗端 alg_control_node 的 WebSocket 接口。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 14567,
                 sim_duration: float = 0.6) -> None:
        self.host = host
        self.port = port
        self.sim_duration = sim_duration
        self._state = "StandBy"
        self._nav_until = 0.0
        self.received: List[Dict[str, Any]] = []  # 收到的 req_func（供断言/审计）
        self._lock = threading.Lock()
        self._server: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._last_error: str = ""
        self._stop_evt: Any = None

    # ---------------- 生命周期 ----------------
    @property
    def available(self) -> bool:
        return _WS_AVAILABLE

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self, timeout: float = 5.0) -> None:
        if not _WS_AVAILABLE:
            raise RuntimeError("未安装 websockets，无法启动虚拟狗 WS 服务（pip install websockets）")
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        deadline = time.time() + timeout
        while time.time() < deadline and self._server is None and not self._last_error:
            time.sleep(0.05)
        if self._server is None:
            msg = "虚拟狗 WS 服务启动超时（端口可能被占用）"
            if self._last_error:
                msg += f"：{self._last_error}"
            raise RuntimeError(msg)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_evt = asyncio.Event()

        async def _serve() -> None:
            # websockets 16.x：serve() 必须在已运行的循环内 await
            self._server = await websockets.serve(self._handler, self.host, self.port,
                                                 max_size=50 * 1024 * 1024)
            await self._stop_evt.wait()  # 一直服务，直到 stop() 置位
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass

        try:
            self._loop.run_until_complete(_serve())
        except Exception as e:  # 端口占用 / 启动失败：让 start() 抛清晰错误
            self._last_error = str(e)
            self._server = None

    def stop(self) -> None:
        if self._loop is not None and self._stop_evt is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop_evt.set)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._server = None
        self._loop = None
        self._stop_evt = None

    # ---------------- 协议处理 ----------------
    async def _handler(self, websocket: Any, *extra: Any) -> None:
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                head = data.get("head", {})
                if head.get("type") != "app_req":
                    continue
                fc = head.get("frame_count")
                req_func = data.get("data", {}).get("req_func", {})
                with self._lock:
                    self.received.append(req_func)
                resp = self._handle(req_func)
                out = {
                    "head": {"type": "app_resp", "frame_count": fc},
                    "data": {"req_result": {"AppResponse": resp}},
                }
                await websocket.send(json.dumps(out, ensure_ascii=False))
        except asyncio.CancelledError:
            pass
        except Exception:
            # 客户端断开等：忽略，不崩服务
            pass

    def _handle(self, req_func: Dict[str, Any]) -> Dict[str, Any]:
        op = next(iter(req_func.keys()), None) if isinstance(req_func, dict) else None
        with self._lock:
            if op == "get_nav_status":
                if self._state == "Navigating" and time.time() >= self._nav_until:
                    self._state = "StandBy"
                return {"status": "success", "msg": "", "data": {"state": self._state}}
            if op in ("start_nav", "start_multi_nav"):
                self._state = "Navigating"
                self._nav_until = time.time() + self.sim_duration
                return {"status": "success", "msg": "nav started", "data": {}}
            if op == "start_nav_return_home":
                self._state = "Navigating"
                self._nav_until = time.time() + self.sim_duration
                return {"status": "success", "msg": "return home started", "data": {}}
            if op == "stop_nav":
                self._state = "StandBy"
                return {"status": "success", "msg": "nav stopped", "data": {}}
            if op in ("pause_nav", "continue_nav"):
                return {"status": "success", "msg": f"{op} ok", "data": {}}
            if op == "loc_load_map":
                return {"status": "success", "msg": "map loaded",
                        "data": {"map_id": req_func.get("loc_load_map")}}
            # 其余（get_all_pgm_map / get_loc_status / get_mapping_status 等）一律放行
            return {"status": "success", "msg": "", "data": {}}
