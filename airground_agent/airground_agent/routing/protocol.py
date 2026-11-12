"""端-边-云任务路由协议（RouteDecision 消息）。

可落地的路由判定以一条结构化消息为载体，边/云节点均可解析、可记录、可回放。
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ..core.types import ComputeIntensity, PrivacyLevel, RouteTier


class FallbackStrategy(str, Enum):
    NONE = "none"                          # 无需回退（正常路径）
    ESCALATE_CLOUD = "escalate_cloud"      # 边侧失败 -> 升级云端（带中间结果）
    DEGRADE_EDGE = "degrade_edge"          # 云端不可用 -> 降级边侧（轻量模型 / 缓存）
    HOLD_STALE = "hold_stale"              # 严重抖动 -> 沿用 last-good 指令
    RETRY_BACKOFF = "retry_backoff"        # 瞬时失败 -> 指数退避重试
    EDGE_CACHE = "edge_cache"              # 云端结果回填边侧缓存


class RouteDecision(BaseModel):
    task_id: str
    task_type: str
    latency_budget_ms: float
    compute_intensity: ComputeIntensity
    privacy_level: PrivacyLevel
    edge_capable: bool
    network_rtt_ms: float
    chosen_tier: RouteTier
    fallback_plan: FallbackStrategy
    reason: str = ""
    timestamp: float = Field(default_factory=time.time)
