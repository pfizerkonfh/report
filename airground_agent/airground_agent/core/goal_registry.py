"""目标寄存器（Goal Registry）。

设计文档问题①的核心机制：长程任务中段「丢失全局目标 / 工具误调用」，本质是缺少
一个跨步骤共享、可校验的全局目标锚点。GoalRegistry 承担三件事：
1. 固化用户最终目标与成功判据（success_criteria）；
2. 登记每个子任务的成功判据，形成「子任务→最终目标」的可追溯链；
3. 提供 is_aligned()：当前动作是否服务于某个仍待完成的子任务 —— 任何未登记、
   与计划脱节的工具调用都会被判定为「偏离全局目标」并在 Reasoner 层被拦截。
"""
from __future__ import annotations

import uuid
from typing import Dict, Optional

from pydantic import BaseModel, Field

from .types import TaskStatus


class GoalRegistry(BaseModel):
    goal_id: str = Field(default_factory=lambda: f"goal-{uuid.uuid4().hex[:8]}")
    user_goal: str = ""
    final_success_criteria: str = ""
    # subtask_id -> success_criteria
    subtask_criteria: Dict[str, str] = Field(default_factory=dict)
    # subtask_id -> status
    subtask_status: Dict[str, TaskStatus] = Field(default_factory=dict)

    def register_goal(self, user_goal: str, final_success_criteria: str) -> None:
        self.user_goal = user_goal
        self.final_success_criteria = final_success_criteria

    def register_subtask(self, subtask_id: str, success_criteria: str) -> None:
        self.subtask_criteria[subtask_id] = success_criteria
        self.subtask_status.setdefault(subtask_id, TaskStatus.PENDING)

    def mark_running(self, subtask_id: str) -> None:
        self.subtask_status[subtask_id] = TaskStatus.RUNNING

    def mark_done(self, subtask_id: str) -> None:
        self.subtask_status[subtask_id] = TaskStatus.DONE

    def mark_failed(self, subtask_id: str) -> None:
        self.subtask_status[subtask_id] = TaskStatus.FAILED

    def is_pending(self, subtask_id: str) -> bool:
        return self.subtask_status.get(subtask_id) in (TaskStatus.PENDING, TaskStatus.RUNNING)

    def is_aligned(self, action_tool: str, subtask_id: Optional[str]) -> bool:
        """动作是否仍服务于最终目标。

        判据：必须显式绑定到一个仍待完成（PENDING/RUNNING）的计划内子任务。
        未绑定子任务、或绑定到已完成/不存在子任务的动作视为「偏离全局目标」。
        """
        if not subtask_id:
            return False
        if subtask_id not in self.subtask_criteria:
            return False
        return self.is_pending(subtask_id)

    def all_done(self) -> bool:
        if not self.subtask_status:
            return False
        return all(s == TaskStatus.DONE for s in self.subtask_status.values())

    def pending_count(self) -> int:
        return sum(1 for s in self.subtask_status.values() if s != TaskStatus.DONE)

    def summary(self) -> str:
        lines = [f"Goal[{self.goal_id}]: {self.user_goal}",
                 f"  成功判据: {self.final_success_criteria}"]
        for sid, crit in self.subtask_criteria.items():
            lines.append(f"  - {sid} [{self.subtask_status.get(sid).value}] : {crit}")
        return "\n".join(lines)
