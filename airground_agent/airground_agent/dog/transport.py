"""传输层：把指令真正送进机器狗的两条物理通道。

通道一 导航（WebSocket）
----------------------
`ws://<ip>:10010`，帧协议见 `protocol.py`。关键点：
- 请求带 `frame_count`，应答 `app_resp` 按同一 `frame_count` 配对（nav-05.py 的做法）；
- 一次请求 = 一个 Future，超时即失败，不做"假装成功"；
- `get_nav_status` 返回字符串状态（现场实测空闲态为 `StandBy`），
  runner 靠它判断"这段导航到底走完没有"。

通道二 动作（zsibot_control --daemon）
------------------------------------
厂家 demo 编译出的 C++ 可执行程序，stdin 写命令文本行、stdout 回一行 JSON。
这里用子进程 + 线程安全的串行读写封装（与 robot_controller.py 的做法一致），
命令与响应一一对应，绝不并发写 stdin。

依赖策略
--------
`websockets` 只在 Ubuntu 现场装了（requirements.txt），Windows 开发机通常没有。
因此这里做可选依赖：缺库时 `connect()` 返回明确错误而不是 ImportError 崩溃，
`dry_run` 通道始终可用 —— 这样编译产物在任意机器上都能先离线校验一遍。
"""
from __future__ import annotations

import asyncio
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # 可选依赖：仅真机下发时需要
    import websockets  # type: ignore
    _WS_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    websockets = None  # type: ignore
    _WS_AVAILABLE = False


#: 导航完成 / 失败 / 运行中的状态词（现场实测 + 协议推断，一律小写比较）
NAV_DONE_STATES = {"standby", "idle", "completed", "complete", "finish", "finished",
                   "success", "succeeded", "done", "arrived"}
NAV_FAIL_STATES = {"failed", "fail", "error", "aborted", "abort", "rejected",
                   "timeout", "lost", "blocked", "unreachable"}


@dataclass
class NavResponse:
    """一次导航请求的应答。"""

    ok: bool
    req_func: Dict[str, Any] = field(default_factory=dict)
    status: str = ""            # ok / error
    msg: str = ""
    data: Any = None
    latency_ms: float = 0.0
    error: str = ""

    @property
    def state(self) -> str:
        """状态查询类请求的语义状态（小写）。"""
        if isinstance(self.data, str):
            return self.data.strip().lower()
        if isinstance(self.data, dict):
            for k in ("state", "status", "nav_status"):
                if k in self.data and isinstance(self.data[k], str):
                    return self.data[k].strip().lower()
        return (self.msg or "").strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "req_func": self.req_func, "status": self.status,
                "msg": self.msg, "data": self.data,
                "latency_ms": round(self.latency_ms, 2), "error": self.error}


@dataclass
class DaemonResponse:
    """一次动作通道命令的应答。"""

    ok: bool
    command: str = ""
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "command": self.command, "message": self.message,
                "data": self.data, "latency_ms": round(self.latency_ms, 2),
                "error": self.error}


class NavClient:
    """导航 WebSocket 客户端（同步 API，内部后台事件循环）。"""

    def __init__(self, uri: str = "ws://192.168.234.234:10010",
                 default_timeout: float = 15.0,
                 max_msg_size: int = 50 * 1024 * 1024) -> None:
        self.uri = uri
        self.default_timeout = default_timeout
        self.max_msg_size = max_msg_size
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws: Any = None
        self._counter = 0
        self._pending: Dict[int, asyncio.Future] = {}
        self._lock: Optional[threading.Lock] = None
        self._connected = False
        self.last_error = ""

    # ---------------- 生命周期 ----------------
    @property
    def available(self) -> bool:
        return _WS_AVAILABLE

    def connect(self, timeout: float = 10.0) -> bool:
        if not _WS_AVAILABLE:
            self.last_error = "未安装 websockets 库（pip install websockets）"
            return False
        if self._connected:
            return True
        self._loop = asyncio.new_event_loop()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run_loop, args=(self._loop,), daemon=True)
        self._thread.start()
        fut = asyncio.run_coroutine_threadsafe(self._connect_async(), self._loop)
        try:
            fut.result(timeout=timeout)
            self._connected = True
            return True
        except Exception as e:  # 连接失败不得崩溃上层
            self.last_error = str(e)
            self._connected = False
            return False

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _connect_async(self) -> None:
        self._ws = await websockets.connect(self.uri, max_size=self.max_msg_size)
        asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                head = data.get("head", {})
                if head.get("type") != "app_resp":
                    continue
                fc = head.get("frame_count")
                if fc is None:
                    continue
                fut = self._pending.pop(fc, None)
                if fut is not None and not fut.done():
                    fut.set_result(data)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # 链路异常：唤醒所有等待者，避免上层永久挂起
            self.last_error = f"接收循环异常: {e}"
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(RuntimeError(self.last_error))
            self._pending.clear()

    def close(self) -> None:
        try:
            if self._connected and self._ws is not None and self._loop is not None:
                fut = asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
                fut.result(timeout=3.0)
        except Exception:
            pass
        self._connected = False
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        self._loop = None

    # ---------------- 请求 ----------------
    def request(self, req_func: Dict[str, Any], timeout: Optional[float] = None) -> NavResponse:
        """下发一个 req_func，等待配对应答。"""
        t0 = time.perf_counter()
        if not self._connected or self._loop is None:
            return NavResponse(ok=False, req_func=req_func, latency_ms=0.0,
                               error=self.last_error or "导航通道未连接")
        assert self._lock is not None
        with self._lock:
            self._counter += 1
            fc = self._counter
        frame = {
            "head": {"type": "app_req", "time_stamp": int(time.time() * 1000),
                     "source": "app", "frame_count": fc},
            "data": {"req_func": req_func},
        }
        try:
            fut = asyncio.run_coroutine_threadsafe(self._send_async(frame, fc), self._loop)
            raw = fut.result(timeout=timeout or self.default_timeout)
        except Exception as e:
            return NavResponse(ok=False, req_func=req_func,
                               latency_ms=(time.perf_counter() - t0) * 1000.0,
                               error=f"请求失败: {e}")
        res = raw.get("data", {}).get("req_result", {})
        if "AppResponse" in res:
            res = res["AppResponse"]
        ok = res.get("status") != "error"
        return NavResponse(ok=ok, req_func=req_func, status=str(res.get("status", "")),
                           msg=str(res.get("msg") or ""), data=res.get("data"),
                           latency_ms=(time.perf_counter() - t0) * 1000.0,
                           error="" if ok else str(res.get("msg") or "请求失败"))

    async def _send_async(self, frame: Dict[str, Any], fc: int) -> Dict[str, Any]:
        fut = asyncio.get_running_loop().create_future()
        self._pending[fc] = fut
        await self._ws.send(json.dumps(frame, ensure_ascii=False))
        return await fut

    # ---------------- 语义封装 ----------------
    def get_nav_status(self, timeout: Optional[float] = None) -> NavResponse:
        return self.request({"get_nav_status": None}, timeout=timeout)

    def wait_nav_idle(self, timeout: float = 120.0, poll_interval: float = 1.0,
                      token: Any = None) -> tuple:
        """轮询导航状态直到空闲（完成）或失败/超时。

        返回 (ok, 最后状态, 耗时秒)。token 为真机急停服务：取消后立刻退出。
        """
        t0 = time.time()
        last = ""
        while True:
            if token is not None and token.is_cancelled():
                return False, "cancelled", time.time() - t0
            r = self.get_nav_status(timeout=max(5.0, poll_interval * 3))
            if not r.ok:
                # 查询失败不立即判死：链路抖动在变电站现场很常见，重试到超时为止
                last = f"query_error:{r.error}"
            else:
                last = r.state
                if any(k in last for k in NAV_DONE_STATES):
                    return True, last, time.time() - t0
                if any(k in last for k in NAV_FAIL_STATES):
                    return False, last, time.time() - t0
            if time.time() - t0 > timeout:
                return False, last or "timeout", time.time() - t0
            time.sleep(poll_interval)

    def health(self) -> Dict[str, Any]:
        return {"ok": self._connected, "transport": "nav-ws", "uri": self.uri,
                "websockets_available": _WS_AVAILABLE,
                "connected": self._connected, "error": self.last_error}


class SdkDaemonClient:
    """`zsibot_control --daemon` 子进程封装（stdin 命令 / stdout JSON）。"""

    def __init__(self, exe_path: str = "./zsibot_control",
                 dog_ip: str = "192.168.234.1",
                 send_port: int = 8081, recv_port: int = 8080) -> None:
        self.exe_path = exe_path
        self.dog_ip = dog_ip
        self.send_port = send_port
        self.recv_port = recv_port
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self.last_error = ""

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float = 10.0) -> bool:
        if self.running:
            return True
        args = [self.exe_path, "--daemon", "--dog-ip", self.dog_ip,
                "--send-port", str(self.send_port), "--recv-port", str(self.recv_port)]
        try:
            self._proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, encoding="utf-8",
                errors="replace", bufsize=1)
        except FileNotFoundError:
            self.last_error = f"找不到控制程序: {self.exe_path}（先 make 编译 main_sdk.cpp）"
            return False
        except Exception as e:
            self.last_error = f"启动控制程序失败: {e}"
            return False
        # 厂家程序启动后才会建立与狗的链路，这里只等进程存活
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self.last_error = "控制程序启动后立即退出（检查狗 IP / 端口 / SDK 安装）"
                return False
            time.sleep(0.1)
        return True

    def send(self, command: str, timeout: float = 10.0) -> DaemonResponse:
        t0 = time.perf_counter()
        if not self.running or self._proc is None:
            return DaemonResponse(ok=False, command=command, latency_ms=0.0,
                                  error=self.last_error or "控制程序未启动")
        with self._lock:  # stdin/stdout 必须串行，否则命令与响应会错配
            try:
                assert self._proc.stdin is not None and self._proc.stdout is not None
                self._proc.stdin.write(command + "\n")
                self._proc.stdin.flush()
                line = self._proc.stdout.readline()
            except Exception as e:
                return DaemonResponse(ok=False, command=command,
                                      latency_ms=(time.perf_counter() - t0) * 1000.0,
                                      error=f"写入命令失败: {e}")
        if not line:
            return DaemonResponse(ok=False, command=command,
                                  latency_ms=(time.perf_counter() - t0) * 1000.0,
                                  error="无响应（进程可能已退出）")
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                return DaemonResponse(ok=bool(payload.get("ok")), command=command,
                                      message=str(payload.get("message", "")),
                                      data=payload.get("data") or {},
                                      latency_ms=(time.perf_counter() - t0) * 1000.0,
                                      error="" if payload.get("ok") else
                                      str(payload.get("message", "执行失败")))
            except json.JSONDecodeError:
                pass
        if line.startswith("ERROR:"):
            return DaemonResponse(ok=False, command=command, message=line[6:].strip(),
                                  latency_ms=(time.perf_counter() - t0) * 1000.0,
                                  error=line[6:].strip())
        return DaemonResponse(ok=True, command=command, message=line,
                              latency_ms=(time.perf_counter() - t0) * 1000.0)

    def get_status(self) -> DaemonResponse:
        return self.send("get_status")

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self.running:
                self.send("quit", timeout=3.0)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=3.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    def health(self) -> Dict[str, Any]:
        return {"ok": self.running, "transport": "sdk-daemon", "exe": self.exe_path,
                "dog_ip": self.dog_ip, "running": self.running, "error": self.last_error}


class DogTransport:
    """双通道组合：导航 + 动作。runner 与 adapter 只依赖这个类。"""

    def __init__(self, nav: Optional[NavClient] = None,
                 sdk: Optional[SdkDaemonClient] = None,
                 dry_run: bool = False) -> None:
        self.nav = nav
        self.sdk = sdk
        self.dry_run = dry_run
        self.sent: List[Dict[str, Any]] = []

    @classmethod
    def dry(cls) -> "DogTransport":
        return cls(dry_run=True)

    def connect(self, nav: bool = True, sdk: bool = True) -> Dict[str, Any]:
        report: Dict[str, Any] = {"nav": None, "sdk": None}
        if self.dry_run:
            report["nav"] = {"ok": True, "dry_run": True}
            report["sdk"] = {"ok": True, "dry_run": True}
            return report
        if nav and self.nav is not None:
            report["nav"] = {"ok": self.nav.connect(), "error": self.nav.last_error}
        if sdk and self.sdk is not None:
            report["sdk"] = {"ok": self.sdk.start(), "error": self.sdk.last_error}
        return report

    def nav_request(self, req_func: Dict[str, Any], timeout: Optional[float] = None) -> NavResponse:
        self.sent.append({"channel": "nav", "payload": req_func})
        if self.dry_run or self.nav is None:
            return NavResponse(ok=True, req_func=req_func, status="ok",
                               msg="dry-run", data="StandBy")
        return self.nav.request(req_func, timeout=timeout)

    def sdk_send(self, command: str, timeout: float = 10.0) -> DaemonResponse:
        self.sent.append({"channel": "sdk", "payload": command})
        if self.dry_run or self.sdk is None:
            return DaemonResponse(ok=True, command=command, message="dry-run")
        return self.sdk.send(command, timeout=timeout)

    def wait_nav_idle(self, timeout: float = 120.0, poll_interval: float = 1.0,
                      token: Any = None) -> tuple:
        if self.dry_run or self.nav is None:
            return True, "dry-run", 0.0
        return self.nav.wait_nav_idle(timeout=timeout, poll_interval=poll_interval, token=token)

    def emergency_stop(self) -> Dict[str, Any]:
        """急停：先停导航，再下发软急停（顺序不能反，否则狗会带着导航目标继续跑）。"""
        out: Dict[str, Any] = {}
        try:
            out["stop_nav"] = self.nav_request({"stop_nav": None}, timeout=5.0).to_dict()
        except Exception as e:
            out["stop_nav"] = {"ok": False, "error": str(e)}
        try:
            out["emergency_stop"] = self.sdk_send("emergency_stop", timeout=5.0).to_dict()
        except Exception as e:
            out["emergency_stop"] = {"ok": False, "error": str(e)}
        return out

    def close(self) -> None:
        if self.nav is not None:
            self.nav.close()
        if self.sdk is not None:
            self.sdk.stop()

    def health(self) -> Dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "nav": self.nav.health() if self.nav else {"ok": False, "reason": "未配置"},
            "sdk": self.sdk.health() if self.sdk else {"ok": False, "reason": "未配置"},
        }


def build_transport(uri: str = "", exe_path: str = "", dog_ip: str = "",
                    send_port: int = 8081, recv_port: int = 8080,
                    dry_run: bool = True) -> DogTransport:
    """按配置构造传输层。dry_run=True 时不下发任何真机指令。"""
    if dry_run:
        return DogTransport.dry()
    nav = NavClient(uri) if uri else None
    sdk = SdkDaemonClient(exe_path, dog_ip or "192.168.234.1", send_port, recv_port) \
        if exe_path else None
    return DogTransport(nav=nav, sdk=sdk, dry_run=False)
