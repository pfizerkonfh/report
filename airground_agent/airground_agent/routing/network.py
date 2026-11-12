"""网络质量治理（Network Governor）。

负责网络抖动的兜底：RTT/丢包注入、降级判定、stale-state hold、指数退避与边侧缓存钩子。
与路由协议配合，支撑「网络抖动兜底策略」（设计文档问题③）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..config.settings import RouterSettings
from ..core.types import RouteTier


@dataclass
class FaultConfig:
    rtt_ms: float = 30.0
    jitter_ms: float = 5.0
    loss_rate: float = 0.0
    cloud_down: bool = False


class NetworkGovernor:
    def __init__(self, cfg: RouterSettings) -> None:
        self.cfg = cfg
        self.fault = FaultConfig()

    def set_fault(self, rtt_ms: float = 30.0, jitter_ms: float = 5.0, loss_rate: float = 0.0, cloud_down: bool = False) -> None:
        self.fault = FaultConfig(rtt_ms, jitter_ms, loss_rate, cloud_down)

    def reset(self) -> None:
        self.fault = FaultConfig()

    def sample_rtt(self) -> float:
        return max(0.0, self.fault.rtt_ms + random.uniform(-self.fault.jitter_ms, self.fault.jitter_ms))

    def is_degraded(self) -> bool:
        return self.fault.rtt_ms > self.cfg.degrade_rtt_ms or self.fault.loss_rate > 0.2

    def cloud_available(self) -> bool:
        return not self.fault.cloud_down

    def extra_latency(self, tier: RouteTier) -> float:
        """云端在降级时附加抖动/重传惩罚；边侧基本不受影响。"""
        if tier == RouteTier.CLOUD and self.is_degraded():
            return self.fault.rtt_ms + random.uniform(0, self.fault.jitter_ms) * 3
        return 0.0

    def should_hold_stale(self) -> bool:
        """严重抖动/丢包时，推理层沿用 last-good 控制指令。"""
        return self.fault.loss_rate > 0.3 or (self.fault.rtt_ms > self.cfg.degrade_rtt_ms * 2)

    def backoff_ms(self, attempt: int) -> float:
        base = self.cfg.backoff_base_ms
        cap = self.cfg.backoff_max_ms
        return min(cap, base * (2 ** max(0, attempt)))
