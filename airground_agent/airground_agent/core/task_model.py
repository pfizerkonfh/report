"""任务模型：Plan / SubTask / TraceStep / Reflection / ExperienceEntry。

Plan 是一个带依赖的 DAG（非单纯线性列表），这是长程任务「中段不丢失目标」的结构保障：
- 每个 SubTask 显式声明 depends_on 与 success_criteria；
- Reasoner 只在「依赖已满足且未完成的节点」上推进，从而保证全局进度可追踪、可回放。
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .goal_registry import GoalRegistry
from .types import FailureLayer, Platform, RouteTier, TaskStatus


class SubTask(BaseModel):
    id: str
    description: str
    depends_on: List[str] = Field(default_factory=list)
    tool_hint: str = ""                 # 规划层建议使用的工具名（Reasoner 校验对齐用）
    success_criteria: str = ""
    platform: Platform = Platform.SHARED  # 物理执行平台（无人机 / 机器狗 / 协同算力）
    status: TaskStatus = TaskStatus.PENDING

    def to_registry(self, reg: GoalRegistry) -> None:
        reg.register_subtask(self.id, self.success_criteria)


class Plan(BaseModel):
    goal_id: str
    root_goal: str
    subtasks: Dict[str, SubTask] = Field(default_factory=dict)
    success_criteria: str = ""
    created_at: float = Field(default_factory=time.time)

    # ---- 拓扑 ----
    def validate_acyclic(self) -> bool:
        visited: Dict[str, int] = {}

        def dfs(n: str, stack: set) -> bool:
            if n in stack:
                return False
            if visited.get(n) == 1:
                return True
            stack.add(n)
            for d in self.subtasks.get(n, SubTask(id=n, description="")).depends_on:
                if d not in self.subtasks:
                    continue
                if not dfs(d, stack):
                    return False
            stack.discard(n)
            visited[n] = 1
            return True

        return all(dfs(n, set()) for n in self.subtasks)

    def topo_order(self) -> List[str]:
        order: List[str] = []
        indeg: Dict[str, int] = {n: 0 for n in self.subtasks}
        for n, st in self.subtasks.items():
            for d in st.depends_on:
                if d in indeg:
                    indeg[n] += 1
        from collections import deque
        q = deque([n for n, d in indeg.items() if d == 0])
        while q:
            n = q.popleft()
            order.append(n)
            for m, st in self.subtasks.items():
                if n in st.depends_on and n in indeg:
                    indeg[m] -= 1
                    if indeg[m] == 0:
                        q.append(m)
        return order

    def ready_nodes(self) -> List[str]:
        """依赖已完成（DONE）且自身未完成的节点。"""
        ready = []
        for n, st in self.subtasks.items():
            if st.status == TaskStatus.DONE:
                continue
            deps_done = all(
                self.subtasks.get(d, SubTask(id=d, description="")).status == TaskStatus.DONE
                for d in st.depends_on
            )
            if deps_done:
                ready.append(n)
        return ready

    def is_complete(self) -> bool:
        return bool(self.subtasks) and all(
            st.status == TaskStatus.DONE for st in self.subtasks.values()
        )

    def register_to(self, reg: GoalRegistry) -> None:
        reg.register_goal(self.root_goal, self.success_criteria)
        for st in self.subtasks.values():
            st.to_registry(reg)


class TraceStep(BaseModel):
    step_id: int
    subtask_id: str
    thought: str = ""
    action: Optional[Dict[str, Any]] = None      # {"tool": ..., "args": ...}
    observation: str = ""
    precondition_ok: bool = True
    precondition_failed: List[str] = Field(default_factory=list)
    tier_used: Optional[RouteTier] = None        # 计算层级：edge / cloud
    platform: Optional[Platform] = None          # 物理平台：无人机 / 机器狗 / 协同算力
    latency_ms: float = 0.0
    # 真机遥测与仿真时长（仅非 legacy 通道有值）：
    #   latency_ms       —— 实测墙钟耗时（受 time_scale 加速影响）
    #   sim_duration_ms  —— 任务真实耗时（未加速，用于衡量真机节奏）
    #   telemetry        —— 电量/里程/位置/链路/相位等真机状态
    sim_duration_ms: float = 0.0
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    attempt: int = 0
    timestamp: float = Field(default_factory=time.time)


class Reflection(BaseModel):
    failed_subtask: str
    symptom: str
    root_cause_layer: FailureLayer
    critique: str
    lessons: str
    retry_recommended: bool = True
    created_at: float = Field(default_factory=time.time)


class ExperienceEntry(BaseModel):
    id: str = Field(default_factory=lambda: f"exp-{uuid.uuid4().hex[:8]}")
    goal_pattern: str            # 触发失败的目标特征（关键词）
    failure_layer: FailureLayer
    lesson: str
    created_at: float = Field(default_factory=time.time)
