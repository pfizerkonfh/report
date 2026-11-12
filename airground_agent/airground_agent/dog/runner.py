"""任务执行器：把 DogMission 按序送进机器狗，并产出可审计的执行报告。

与决策平台的关系
----------------
决策平台负责"想清楚该做什么"，本执行器负责"按真机语义把它做完"：

- **导航是阻塞的**：下发 `start_nav` 后必须轮询 `get_nav_status` 直到 StandBy，
  不能发完就当成功 —— 真机上"指令被接受"与"走到目的地"是两件事。
- **超时就是超时**：`DogCommand.timeout_s` 到点判 TIMEOUT，不做"假装成功"。
- **急停必须立刻生效**：`CancellationToken` 被置位后，先 `stop_nav` 再 `emergency_stop`，
  并终止后续指令（与决策平台的急停语义一致：不再重规划、不再继续跑）。
- **全程留痕**：每条指令的响应、耗时、导航最终状态都进报告，便于事后定位
  "到底是决策错了、坐标错了、还是狗没走到"。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..tools.adapters.base import CancellationToken
from .protocol import CHANNEL_NAV, CHANNEL_SDK, CHANNEL_WAIT, DogCommand, DogMission
from .transport import DogTransport

#: 这类导航指令需要等待"真正走完"，其余指令下发成功即可继续
BLOCKING_NAV_OPS = {"start_nav", "start_multi_nav", "start_nav_return_home"}


@dataclass
class CommandResult:
    cmd_id: str
    op: str
    channel: str
    ok: bool
    latency_ms: float = 0.0
    response: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    nav_state: str = ""
    subtask_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"cmd_id": self.cmd_id, "op": self.op, "channel": self.channel,
                "ok": self.ok, "latency_ms": round(self.latency_ms, 2),
                "response": self.response, "error": self.error,
                "nav_state": self.nav_state, "subtask_id": self.subtask_id}


@dataclass
class MissionReport:
    mission_id: str = ""
    goal: str = ""
    results: List[CommandResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    aborted: bool = False
    abort_reason: str = ""
    success: bool = False
    telemetry: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000.0 if self.finished_at else 0.0

    def to_dict(self) -> Dict[str, Any]:
        done = sum(1 for r in self.results if r.ok)
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "success": self.success,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "commands": len(self.results),
            "succeeded": done,
            "failed": len(self.results) - done,
            "total_ms": round(self.total_ms, 2),
            "telemetry": self.telemetry,
            "results": [r.to_dict() for r in self.results],
        }


class DogMissionRunner:
    """按序执行 DogMission。"""

    def __init__(self, transport: DogTransport,
                 token: Optional[CancellationToken] = None,
                 stop_on_failure: bool = True,
                 nav_poll_interval: float = 1.0,
                 progress_cb: Any = None) -> None:
        self.transport = transport
        self.token = token or CancellationToken()
        self.stop_on_failure = stop_on_failure
        self.nav_poll_interval = nav_poll_interval
        self.progress_cb = progress_cb

    def _emit(self, idx: int, total: int, cmd: DogCommand) -> None:
        if self.progress_cb is None:
            return
        try:
            self.progress_cb(idx / max(total, 1), f"{cmd.describe()}")
        except Exception:  # 进度回调不影响执行
            pass

    def run(self, mission: DogMission) -> MissionReport:
        problems = mission.validate()
        rep = MissionReport(mission_id=mission.mission_id, goal=mission.goal,
                            started_at=time.time())
        if problems:
            # 编译期就发现的协议错误：不下发，直接失败（避免拿狗去试错）
            rep.finished_at = time.time()
            rep.success = False
            rep.abort_reason = "指令校验未通过: " + "; ".join(problems)
            rep.telemetry["validation_problems"] = problems
            return rep

        total = len(mission.commands)
        for i, cmd in enumerate(mission.commands, start=1):
            self._emit(i, total, cmd)
            if self.token.is_cancelled():
                rep.aborted = True
                rep.abort_reason = self.token.reason or "用户中止"
                rep.telemetry["emergency"] = self.transport.emergency_stop()
                break

            res = self._run_one(cmd)
            rep.results.append(res)
            if not res.ok and self.stop_on_failure:
                rep.abort_reason = f"指令 {cmd.id}（{cmd.describe()}）失败: {res.error}"
                break

        rep.finished_at = time.time()
        rep.success = bool(rep.results) and all(r.ok for r in rep.results) and not rep.aborted
        rep.telemetry.update(self._collect_telemetry())
        return rep

    # ------------------------------------------------------------------
    def _run_one(self, cmd: DogCommand) -> CommandResult:
        t0 = time.perf_counter()

        if cmd.channel == CHANNEL_NAV:
            req = cmd.to_req_func()
            r = self.transport.nav_request(req)
            nav_state = r.state
            # 导航类必须等到"真的到了"，否则后续指令会在半路上被打断
            if r.ok and cmd.blocking and cmd.op in BLOCKING_NAV_OPS:
                ok, state, _ = self.transport.wait_nav_idle(
                    timeout=cmd.timeout_s, poll_interval=self.nav_poll_interval,
                    token=self.token)
                nav_state = state
                if not ok:
                    return CommandResult(cmd_id=cmd.id, op=cmd.op, channel=cmd.channel,
                                         ok=False, latency_ms=(time.perf_counter() - t0) * 1000.0,
                                         response=r.to_dict(), nav_state=state,
                                         subtask_id=cmd.subtask_id,
                                         error=f"导航未完成（状态={state}）")
            return CommandResult(cmd_id=cmd.id, op=cmd.op, channel=cmd.channel, ok=r.ok,
                                 latency_ms=(time.perf_counter() - t0) * 1000.0,
                                 response=r.to_dict(), nav_state=nav_state,
                                 error=r.error, subtask_id=cmd.subtask_id)

        if cmd.channel == CHANNEL_SDK:
            r = self.transport.sdk_send(cmd.op)
            return CommandResult(cmd_id=cmd.id, op=cmd.op, channel=cmd.channel, ok=r.ok,
                                 latency_ms=(time.perf_counter() - t0) * 1000.0,
                                 response=r.to_dict(), error=r.error,
                                 subtask_id=cmd.subtask_id)

        if cmd.channel == CHANNEL_WAIT:
            seconds = float(cmd.params.get("seconds", 1.0))
            slept = self._sleep_cancellable(seconds)
            return CommandResult(cmd_id=cmd.id, op="sleep", channel=cmd.channel,
                                 ok=not self.token.is_cancelled(),
                                 latency_ms=(time.perf_counter() - t0) * 1000.0,
                                 response={"seconds": seconds, "slept": round(slept, 2)},
                                 error="" if slept >= seconds * 0.99 else "等待被中止",
                                 subtask_id=cmd.subtask_id)

        # meta：仅记录，不下发
        return CommandResult(cmd_id=cmd.id, op=cmd.op or "note", channel=cmd.channel,
                             ok=True, latency_ms=0.0,
                             response={"note": cmd.note}, subtask_id=cmd.subtask_id)

    def _sleep_cancellable(self, seconds: float) -> float:
        step = 0.05
        elapsed = 0.0
        while elapsed < seconds:
            if self.token.is_cancelled():
                break
            import time as _t
            _t.sleep(min(step, seconds - elapsed))
            elapsed += step
        return min(elapsed, seconds)

    def _collect_telemetry(self) -> Dict[str, Any]:
        """从动作通道取一次真机状态（电量/温度等），失败不影响报告。"""
        try:
            r = self.transport.sdk_send("get_status", timeout=5.0)
            if r.ok and r.data:
                return {"robot_status": r.data}
        except Exception:
            pass
        return {}
