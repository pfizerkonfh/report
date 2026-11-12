"""工具执行器：把「前置校验 → 路由决策 → 执行 → 时延度量」串成一步。

- 不满足 Precondition：直接阻断，返回失败并列出违例条件（不擅自替换工具），
  交由 Reasoner 升级重规划 —— 这是杜绝「工具误调用」的关键执行点。
- 路由决策来自 Router；云端降级时按 FallbackStrategy 调整 tier 并计入时延。

执行通道（ToolAdapter）
----------------------
本执行器同时支持两条执行路径：

- **legacy**：直接调用 `ToolSpec.run()`（原有 Mock，零耗时、必然成功）。
- **adapter**：调用 `ToolAdapter.invoke()`，具备真实时序语义 —— 真实墙钟耗时、
  真实超时、可取消、遥测回传。这是接真机（仿真 / 录播 / ROS2）的通道。

两条路径对上层的 `ExecResult` 完全一致，因此 Planner / Reasoner / 评测无需改动。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.types import RouteTier
from ..core.world_state import WorldState
from ..routing.network import NetworkGovernor
from ..routing.protocol import RouteDecision
from ..routing.router import Router
from ..tools.adapters import AdapterBundle, SimConfig, make_context
from ..tools.base import ToolResult, ToolSpec
from ..tools.registry import ToolRegistry


@dataclass
class ExecResult:
    ok: bool
    result: Optional[ToolResult] = None
    decision: Optional[RouteDecision] = None
    precondition_failed: List[str] = field(default_factory=list)
    blocked: bool = False         # 前置校验失败导致的阻断
    reason: str = ""


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, router: Router, net: NetworkGovernor,
                 adapters: Optional[AdapterBundle] = None,
                 sim: Optional[SimConfig] = None) -> None:
        self.registry = registry
        self.router = router
        self.net = net
        self.adapters = adapters
        self.sim = sim or SimConfig()

    def execute(self, tool_name: str, ws: WorldState, args: Optional[dict] = None,
                platform: Any = None, progress_cb: Optional[Callable[[float, str], None]] = None,
                token: Any = None) -> ExecResult:
        tool: Optional[ToolSpec] = self.registry.get(tool_name)
        if tool is None:
            return ExecResult(ok=False, blocked=False, reason=f"工具未注册: {tool_name}")

        # 1) 前置校验（硬约束）
        failed = tool.check_preconditions(ws)
        if failed:
            return ExecResult(
                ok=False, blocked=True,
                precondition_failed=[c.human() for c in failed],
                reason="前置条件不满足，阻断并升级重规划",
            )

        # 2) 路由决策
        decision = self.router.decide(tool, task_type=tool_name)
        tier = decision.chosen_tier

        # 3) 执行
        if self.adapters is not None:
            return self._execute_via_adapter(tool, ws, args, tier, decision,
                                             platform, progress_cb, token)
        return self._execute_legacy(tool, ws, args, tier, decision)

    # ---- 通道一：ToolAdapter（真实时序，可接真机）----
    def _execute_via_adapter(self, tool: ToolSpec, ws: WorldState, args, tier, decision,
                             platform, progress_cb, token) -> ExecResult:
        from ..core.types import Platform
        adapter = self.adapters.get(tool.name)
        plat = platform if isinstance(platform, Platform) else (
            Platform(str(platform)) if platform else tool.platform_hint)

        # 纯计算 / 查询类工具（感知层）：观测与副作用必须来自真实计算，
        # 执行通道只贡献时序与遥测。否则仿真通道会用模板 observation 与
        # 静态声明的 effects（如 route_len_m=0.0）覆盖掉真实规划结果。
        if getattr(tool, "compute_only", False):
            return self._execute_compute_only(tool, ws, args, tier, decision,
                                              plat, adapter, progress_cb, token)

        ctx = make_context(tool_name=tool.name, platform=plat, tier=tier, ws=ws, args=args,
                           time_scale=self.sim.time_scale, timeout_ms=self.sim.timeout_ms,
                           token=token, progress_cb=progress_cb,
                           declared_effects=list(tool.effects))
        try:
            out = adapter.invoke(ctx)
        except Exception as e:  # 真机异常必须收敛为失败，交给重规划/Reflexion
            return ExecResult(ok=False, blocked=False, decision=decision,
                              reason=f"执行通道异常: {e}")

        # 副作用回写：真机以遥测为准（out.effects 非空），否则沿用工具声明
        applied = out.effects if out.effects is not None else (tool.effects if out.success else [])
        if out.effects is not None and out.success:
            ws.apply_many(out.effects)

        res = ToolResult(success=out.success, observation=out.observation or out.error,
                         applied_effects=applied,
                         latency_ms=out.latency_ms + self.net.extra_latency(tier),
                         tier_used=tier, error=out.error,
                         sim_duration_ms=out.sim_duration_ms,
                         telemetry=out.telemetry, adapter=adapter.name)
        reason = out.error or "ok"
        if out.status.value == "timeout":
            reason = f"任务超时（>{self.sim.timeout_ms / 1000:.0f}s）"
        elif out.status.value == "link_lost":
            reason = "数传链路中断"
        elif out.status.value == "cancelled":
            reason = "任务被中止"
        return ExecResult(ok=res.success, result=res, decision=decision, reason=reason)

    # ---- 通道三：纯计算工具 = 真实计算 + 通道时序叠加 ----
    def _execute_compute_only(self, tool: ToolSpec, ws: WorldState, args, tier, decision,
                              plat, adapter, progress_cb, token) -> ExecResult:
        """感知层等查询/规划工具：结果必须真实计算，执行通道只叠加时序与遥测。

        顺序上先跑通道（拿到真实墙钟耗时、相位遥测、取消语义），再跑真实计算，
        确保真实计算结果最终落盘。通道侧不回写任何声明 effects，避免静态默认值
        （如 route_ready=False / route_len_m=0.0）覆盖真实值。
        """
        chan_ms, sim_ms, telemetry, chan_status = 0.0, 0.0, {}, "ok"
        if adapter is not None:
            try:
                ctx = make_context(tool_name=tool.name, platform=plat, tier=tier, ws=ws,
                                   args=args, time_scale=self.sim.time_scale,
                                   timeout_ms=self.sim.timeout_ms, token=token,
                                   progress_cb=progress_cb, declared_effects=[])
                out = adapter.invoke(ctx)
                chan_ms, sim_ms = out.latency_ms, out.sim_duration_ms
                telemetry = dict(out.telemetry or {})
                chan_status = out.status.value
                if chan_status == "cancelled":
                    return ExecResult(ok=False, blocked=False, decision=decision,
                                      reason="任务被中止")
            except Exception:  # 通道异常不否定本地计算能力
                telemetry["channel_error"] = "adapter_unavailable"

        # 真实计算：observation / effects / telemetry 以此为准
        try:
            res = tool.run(ws, **(args or {}))
        except Exception as e:
            return ExecResult(ok=False, blocked=False, decision=decision,
                              reason=f"计算异常: {e}")

        merged = dict(telemetry)
        for k, v in (res.telemetry or {}).items():
            merged[k] = v          # 真实计算结果优先
        if chan_status != "ok":
            merged["channel_status"] = chan_status

        res.tier_used = tier
        res.latency_ms = res.latency_ms + chan_ms + self.net.extra_latency(tier)
        res.sim_duration_ms = sim_ms
        res.telemetry = merged
        res.adapter = f"{adapter.name}+compute" if adapter else "compute"
        return ExecResult(ok=res.success, result=res, decision=decision,
                          reason=res.error or "ok")

    # ---- 通道二：legacy（原有 Mock，零耗时）----
    def _execute_legacy(self, tool: ToolSpec, ws: WorldState, args, tier, decision) -> ExecResult:
        try:
            res = tool.run(ws, **(args or {}))
        except Exception as e:  # 真实执行异常也归为失败，便于回退
            return ExecResult(ok=False, blocked=False, decision=decision, reason=f"执行异常: {e}")

        res.tier_used = tier
        res.latency_ms = res.latency_ms + self.net.extra_latency(tier)
        return ExecResult(ok=res.success, result=res, decision=decision, reason=res.error or "ok")
