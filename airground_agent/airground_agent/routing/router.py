"""任务路由器：判定哪些任务留在边侧、哪些下沉云端，并给出失败回退策略。

判定树（设计文档问题③）：
  1) 隐私高且可边侧  -> 边侧（人脸等敏感数据本地处理）
  2) 时延预算紧且可边侧 -> 边侧
  3) 算力高/需大模型   -> 云端
  4) 其余默认边侧
失败回退：
  - 选云端且网络降级/云不可用 -> 若可边侧则 DEGRADE_EDGE，否则 HOLD_STALE/RETRY_BACKOFF
  - 选边侧且网络降级 -> RETRY_BACKOFF；严重抖动 -> HOLD_STALE
"""
from __future__ import annotations

import uuid

from ..core.types import RouteTier
from ..tools.base import ToolSpec
from .network import NetworkGovernor
from .protocol import FallbackStrategy, RouteDecision


class Router:
    def __init__(self, net: NetworkGovernor) -> None:
        self.net = net

    def decide(self, tool: ToolSpec, task_type: str = "") -> RouteDecision:
        rtt = self.net.sample_rtt()
        degraded = self.net.is_degraded()
        hint = tool.route

        if hint.privacy_level.value == "high" and hint.edge_capable:
            tier = RouteTier.EDGE
            reason = "隐私敏感数据，强制边侧本地处理"
        elif hint.latency_budget_ms <= self._edge_budget() and hint.edge_capable:
            tier = RouteTier.EDGE
            reason = "时延预算紧且边侧可承载"
        elif hint.compute_intensity.value == "high" or (not hint.edge_capable):
            tier = RouteTier.CLOUD
            reason = "算力高/需云端大模型服务"
        else:
            tier = RouteTier.EDGE
            reason = "默认边侧"

        # 失败回退策略
        if tier == RouteTier.CLOUD and (degraded or not self.net.cloud_available()):
            if hint.edge_capable:
                tier = RouteTier.EDGE
                fb = FallbackStrategy.DEGRADE_EDGE
                reason += "；云端降级，回退边侧轻量执行"
            else:
                fb = FallbackStrategy.HOLD_STALE if self.net.should_hold_stale() else FallbackStrategy.RETRY_BACKOFF
                reason += "；云端不可用且无边侧替代，需 hold/重试"
        elif tier == RouteTier.EDGE and degraded:
            fb = FallbackStrategy.HOLD_STALE if self.net.should_hold_stale() else FallbackStrategy.RETRY_BACKOFF
            reason += "；边侧网络抖动，启用退避/保持"
        else:
            fb = FallbackStrategy.NONE

        return RouteDecision(
            task_id=f"rt-{uuid.uuid4().hex[:8]}",
            task_type=task_type or tool.name,
            latency_budget_ms=hint.latency_budget_ms,
            compute_intensity=hint.compute_intensity,
            privacy_level=hint.privacy_level,
            edge_capable=hint.edge_capable,
            network_rtt_ms=rtt,
            chosen_tier=tier,
            fallback_plan=fb,
            reason=reason,
        )

    def _edge_budget(self) -> float:
        return self.net.cfg.edge_latency_budget_ms
