"""统一评测指标体系（设计文档问题④）。

四指标：规划成功率 / Agent 动作合理度 / 端到端时延 / 用户满意度（CSAT）。
以 task + trace_id 贯穿规划层 / 动作层 / 端侧推理层，支持跨层归因。
"""
from __future__ import annotations

import time
import uuid
from typing import List, Optional

from pydantic import BaseModel, Field

from ..core.task_model import Plan, TraceStep
from ..core.types import FailureLayer, RunStatus


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


class EvalRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    goal: str = ""
    status: RunStatus = RunStatus.FAILED
    planning_success: bool = False
    action_reasonableness: float = 0.0     # 0~1
    precondition_violations: int = 0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    end_to_end_ms: float = 0.0
    csat: float = 0.0
    steps: int = 0
    root_cause_layer: Optional[FailureLayer] = None
    timestamp: float = Field(default_factory=time.time)


def compute_metrics(goal: str, status: RunStatus, plan: Optional[Plan], trace: List[TraceStep],
                    final_facts: dict, csat: float = 4.5,
                    root_cause: Optional[FailureLayer] = None) -> EvalRecord:
    latencies = [t.latency_ms for t in trace]
    violations = sum(1 for t in trace if not t.precondition_ok)
    total = max(1, len(trace))
    reasonableness = 1.0 - (violations / total)
    if status != RunStatus.SUCCESS:
        reasonableness *= 0.5
    planning_success = plan is not None and len(plan.subtasks) > 0
    return EvalRecord(
        goal=goal, status=status, planning_success=planning_success,
        action_reasonableness=round(reasonableness, 4),
        precondition_violations=violations,
        latency_p50=round(_percentile(latencies, 0.5), 2),
        latency_p95=round(_percentile(latencies, 0.95), 2),
        latency_p99=round(_percentile(latencies, 0.99), 2),
        end_to_end_ms=round(sum(latencies), 2),
        csat=csat, steps=len(trace),
        root_cause_layer=root_cause,
    )
