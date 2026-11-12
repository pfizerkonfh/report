"""在线 A/B 与护栏自动回滚（设计文档问题④）。

- 影子模式：治疗组与对照组并行跑同一输入，仅对照组真实生效；
- 护栏：若治疗组端到端时延恶化超过阈值、或任务成功率下降，则自动回滚到对照组配置。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from .metrics import EvalRecord
from ..config.settings import EvalSettings
from ..core.log import get_logger

logger = get_logger("eval.ab")


@dataclass
class ABVerdict:
    keep_treatment: bool
    reason: str
    control: EvalRecord
    treatment: EvalRecord


class ABTest:
    def __init__(self, settings: EvalSettings) -> None:
        self.guardrail_latency_drop_ms = settings.ab_guardrail_latency_drop_ms

    def compare(self, control: EvalRecord, treatment: EvalRecord) -> ABVerdict:
        if treatment.status.value != "success" and control.status.value == "success":
            return ABVerdict(False, "治疗组任务失败，回滚", control, treatment)
        if treatment.end_to_end_ms - control.end_to_end_ms > self.guardrail_latency_drop_ms:
            return ABVerdict(False, "治疗组端到端时延恶化超护栏阈值，回滚", control, treatment)
        if treatment.action_reasonableness < control.action_reasonableness - 0.05:
            return ABVerdict(False, "治疗组动作合理度下降，回滚", control, treatment)
        return ABVerdict(True, "治疗组指标不劣于对照组，可灰度放量", control, treatment)
