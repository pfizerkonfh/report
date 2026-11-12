"""工具适配层（ToolAdapter）抽象协议 —— 让系统能接真机，而不只是 Mock。

定位
----
`ToolSpec` 描述的是**语义契约**（Precondition / Effect / Rollback / RouteHint），
它回答"能不能做、做了会怎样"；`ToolAdapter` 描述的是**执行通道**，它回答
"如何真正把这件事做出来"。

两者分离后：
- 算法层（Planner / Reasoner / Router / 评测）完全不感知底层是 Mock、仿真还是真机；
- 换真机只需换 Adapter，不动任何决策逻辑 —— 这是从 POC 走向生产的关键解耦点。

真实时序语义（与 Mock 的根本区别）
--------------------------------
Mock 工具的 `latency_ms` 是**写死的常量**（不消耗真实时间）。真机不可能这样，
因此本适配层强制以下语义：

1. **真实墙钟耗时**：`invoke()` 必须真实消耗时间，用 `time.perf_counter()` 实测，
   而不是返回常量。仿真模式下用 `time_scale` 加速（仿真 60s → 真实 6s @10x）。
2. **真实超时**：任务超时就是超时，返回 `AdapterStatus.TIMEOUT`，不做"假装成功"。
3. **可取消**：通过 `CancellationToken` 在等待期间周期性检查，支持急停/中止。
4. **遥测回传**：真机会持续回传状态（电量、位置、链路、相位），存在 `telemetry` 里，
   供上层做真实的状态感知与控制决策。
5. **分阶段进度**：长任务通过 `progress_cb` 回报相位与百分比，前端可实时看到
   "无人机巡航中 45%"，而不是干等。
"""
from __future__ import annotations

import abc
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ...core.types import Effect, Platform, RouteTier
from ...core.world_state import WorldState


class AdapterStatus(str, Enum):
    """一次真实调用的结果状态（比 bool 更贴近真机语义）。"""
    SUCCEEDED = "succeeded"
    FAILED = "failed"          # 任务执行失败（真机报错 / 任务未达成）
    TIMEOUT = "timeout"        # 超过 timeout_ms 未返回（链路卡死 / 真机失控）
    CANCELLED = "cancelled"    # 被上层主动中止（急停 / 用户取消）
    LINK_LOST = "link_lost"    # 通信链路丢失（网络/数传中断）


class AdapterMode(str, Enum):
    """执行通道模式。"""
    LEGACY = "legacy"      # 原 Mock（零耗时，仅语义仿真）—— 纯算法验证用
    SIM = "sim"            # 硬件在环仿真：真实时序 + 真机行为模型
    PLAYBACK = "playback"  # 录播回放：按记录的真实时序与遥测复现现场
    ROS2 = "ros2"          # 真机：ROS2 action / service
    ZSIBOT = "zsibot"      # 真机：ZSIBOT 机器狗（导航 WebSocket + SDK daemon 双通道）


@dataclass
class SimClock:
    """仿真时钟：把"仿真时长"按加速比压缩为"真实墙钟时长"。

    time_scale=10 表示仿真中 60 秒的任务，真实只花 6 秒。
    time_scale=1  表示实时（真实机器人就是这个节奏）。
    """

    time_scale: float = 1.0

    def __post_init__(self) -> None:
        # 防止 0 / 负数导致除零或死循环
        if not self.time_scale or self.time_scale <= 0:
            self.time_scale = 1.0

    def sleep(self, sim_ms: float, token: Optional["CancellationToken"] = None,
              progress_cb: Optional[Callable[[float, str], None]] = None,
              phase: str = "") -> bool:
        """睡 '仿真 sim_ms 毫秒' 对应的真实时长。

        分片睡眠而非一次 sleep，这样取消请求能在 ~50ms 内被响应（真机急停要求）。
        返回 True 表示睡满；False 表示中途被取消。
        """
        total_real = max(0.0, sim_ms / 1000.0 / self.time_scale)
        if total_real <= 0:
            if progress_cb:
                progress_cb(1.0, phase)
            return True

        step = 0.05  # 50ms 分片，兼顾响应速度与 CPU 开销
        elapsed = 0.0
        while elapsed < total_real:
            if token is not None and token.is_cancelled():
                return False
            chunk = min(step, total_real - elapsed)
            time.sleep(chunk)
            elapsed += chunk
            if progress_cb:
                progress_cb(min(1.0, elapsed / total_real), phase)
        return True


class CancellationToken:
    """取消令牌：支持上层（Web UI / 急停按钮）中止正在执行的真机任务。"""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: str = ""

    def cancel(self, reason: str = "用户中止") -> None:
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason


@dataclass
class AdapterContext:
    """一次调用的上下文：告诉适配器"谁在做、在哪算、多快、能不能取消"。"""

    tool_name: str
    platform: Platform = Platform.SHARED
    tier: RouteTier = RouteTier.EDGE
    args: Dict[str, Any] = field(default_factory=dict)
    clock: SimClock = field(default_factory=SimClock)
    token: Optional[CancellationToken] = None
    timeout_ms: float = 30_000.0
    # 进度回调 (percent 0~1, phase_name) —— 长任务用于实时反馈
    progress_cb: Optional[Callable[[float, str], None]] = None
    # 世界状态快照（只读语义：适配器据此判断真机当前是否具备条件）
    facts: Dict[str, Any] = field(default_factory=dict)
    # 真实 WorldState 实例（供 LegacyMockAdapter 复用既有 Mock 工具实现；
    # 真机适配器不应直接改它，副作用统一由 Executor 按 Effect 回写）
    ws: Optional[WorldState] = None
    # 工具声明的语义副作用（footage / edited / anonymized …）。
    # 真机适配器必须把它们与物理遥测合并回写，否则后续步骤的前置校验会全线阻断
    # —— 这个坑是硬件在环仿真跑出来的：只回传电量、不回传"素材已生成"。
    declared_effects: List[Effect] = field(default_factory=list)


@dataclass
class AdapterOutcome:
    """一次真实调用的结果。

    注意 `latency_ms` 是**实测墙钟耗时**，`sim_duration_ms` 才是模型/真机认为
    "这件事花了多久"。二者分离，评测时才能区分"算法慢"与"仿真加速"。
    """

    status: AdapterStatus
    success: bool
    observation: str = ""
    latency_ms: float = 0.0          # 实测墙钟耗时（ms）
    sim_duration_ms: float = 0.0     # 仿真/真实任务时长（ms，未加速）
    telemetry: Dict[str, Any] = field(default_factory=dict)
    effects: Optional[List[Effect]] = None   # 真机实际产生的副作用（None 则用工具声明）
    error: str = ""

    @property
    def timed_out(self) -> bool:
        return self.status == AdapterStatus.TIMEOUT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "success": self.success,
            "observation": self.observation,
            "latency_ms": round(self.latency_ms, 2),
            "sim_duration_ms": round(self.sim_duration_ms, 2),
            "telemetry": self.telemetry,
            "error": self.error,
        }


class ToolAdapter(abc.ABC):
    """工具适配器抽象基类。

    实现类只需回答一个问题：**如何用真实通道把 tool_name 这件事做出来**，
    并且必须遵守真实时序语义（耗时/超时/可取消/遥测）。
    """

    #: 适配器名称（用于日志与前端展示）
    name: str = "abstract"
    #: 是否为"真实硬件通道"（ROS2 为 True；仿真/录播/Mock 为 False）
    real_hardware: bool = False

    @abc.abstractmethod
    def invoke(self, ctx: AdapterContext) -> AdapterOutcome:
        """执行一次真实调用。**必须真实消耗时间**并返回实测 latency_ms。"""
        raise NotImplementedError

    # ---- 可选能力 ----

    def health(self) -> Dict[str, Any]:
        """通道健康检查（真机用于判断链路/节点是否在线）。"""
        return {"ok": True, "adapter": self.name, "real_hardware": self.real_hardware}

    def abort(self, reason: str = "急停") -> None:
        """紧急中止（真机需下发急停指令）。默认无操作。"""


class LegacyMockAdapter(ToolAdapter):
    """兼容适配器：把原有 Mock 工具的 `run()` 包装进适配层。

    它**不消耗真实时间**（保留原有零耗时语义），用于纯算法回归测试。
    这是默认行为，保证既有 20+ 用例不受影响。
    """

    name = "legacy-mock"
    real_hardware = False

    def __init__(self, tool: Any) -> None:
        self._tool = tool

    def invoke(self, ctx: AdapterContext) -> AdapterOutcome:
        t0 = time.perf_counter()
        # Mock 工具签名要求真实的 WorldState 实例（有 .apply 方法），不能传 dict
        ws = ctx.ws if ctx.ws is not None else WorldState(facts=dict(ctx.facts))
        res = self._tool.run(ws, **ctx.args)
        elapsed = (time.perf_counter() - t0) * 1000.0
        return AdapterOutcome(
            status=AdapterStatus.SUCCEEDED if getattr(res, "success", False) else AdapterStatus.FAILED,
            success=bool(getattr(res, "success", False)),
            observation=getattr(res, "observation", ""),
            latency_ms=max(elapsed, getattr(res, "latency_ms", 0.0)),
            sim_duration_ms=getattr(res, "latency_ms", 0.0),
            telemetry={"source": "legacy-mock", "simulated": True},
            error=getattr(res, "error", ""),
        )
