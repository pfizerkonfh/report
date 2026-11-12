"""工具框架：ToolSpec 显式承载 Precondition / Effect / Rollback / RouteHint。

对应设计文档问题①：用可判定的前置/效果语义替代纯自然语言描述，从机制上杜绝「工具误调用」。
- preconditions：执行前的硬约束，由 WorldState 引擎判定；不满足则阻断并升级重规划。
- effects：执行后对世界状态的副作用（用于观测与回放）。
- rollback：失败时可逆操作（尽量提供，非强制）。
- route：端-边-云路由偏好（详见 routing 模块）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.types import (
    ComputeIntensity, Condition, Effect, Platform, PrivacyLevel, RouteTier,
)
from ..core.world_state import WorldState


class RouteHint(BaseModel):
    tier_preference: RouteTier = RouteTier.EDGE
    compute_intensity: ComputeIntensity = ComputeIntensity.LOW
    latency_budget_ms: float = 60.0
    privacy_level: PrivacyLevel = PrivacyLevel.LOW
    edge_capable: bool = True
    # 云端依赖标识（用于失败回退判断是否可下沉边侧）
    cloud_service: str = ""


class ToolResult(BaseModel):
    success: bool
    observation: str = ""
    applied_effects: List[Effect] = Field(default_factory=list)
    latency_ms: float = 0.0
    tier_used: Optional[RouteTier] = None
    error: str = ""
    # 真机通道补充信息（legacy Mock 下为空）
    sim_duration_ms: float = 0.0                       # 任务真实耗时（未加速）
    telemetry: Dict[str, Any] = Field(default_factory=dict)   # 电量/里程/位置/链路/相位
    adapter: str = ""                                  # 执行通道名称


class ToolSpec(BaseModel):
    name: str
    description: str
    preconditions: List[Condition] = Field(default_factory=list)
    effects: List[Effect] = Field(default_factory=list)
    rollback: List[Effect] = Field(default_factory=list)
    route: RouteHint = Field(default_factory=RouteHint)
    # 该工具默认的物理执行平台建议（无人机 / 机器狗 / 协同算力）。
    # 仅作兜底提示：真正的平台由 Planner 分配到 SubTask.platform 决定。
    platform_hint: Platform = Platform.SHARED
    # 仿真基准时延（ms），用于路由与评测度量
    base_latency_ms: float = 0.0
    # 是否可被 Reflexion/重规划阶段修复（不可修复 -> 触发 Reflexion）
    fixable: bool = True
    # 纯计算 / 查询类工具（如感知层的占据栅格查询、A* 规划、高度剖面）。
    # 这类工具没有物理执行动作，其观测与副作用**必须来自真实计算**；
    # 执行通道（仿真 / 录播 / 真机）只叠加时序与遥测，不得用模板 observation
    # 或静态声明的 effects 覆盖真实计算结果。
    # 典型反例：route_plan 在仿真通道下若被模板覆盖，会回写 route_len_m=0.0，
    # 使后续依赖路径里程的前置校验失效。
    compute_only: bool = False

    # ---- 前置校验（不修改世界状态）----
    def check_preconditions(self, ws: WorldState) -> List[Condition]:
        return ws.check_all(self.preconditions)

    def can_run(self, ws: WorldState) -> bool:
        return len(self.check_preconditions(ws)) == 0

    # ---- 实际执行（子类实现）----
    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        raise NotImplementedError

    # 便于注册表展示
    def signature(self) -> str:
        pre = "; ".join(c.human() for c in self.preconditions) or "∅"
        eff = "; ".join(f"{e.key}<-{e.op}:{e.value}" for e in self.effects) or "∅"
        return f"{self.name} | pre:[{pre}] -> eff:[{eff}]"
