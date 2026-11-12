"""跨层归因矩阵（设计文档问题④）。

把 badcase 的现象映射到根因层（规划层 / 动作层 / 端侧推理层 / 网络层 / 交互层），
用于定位「三层链路评测口径互不相通」时 badcase 的根因归属。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..core.types import FailureLayer

# (现象关键词, 推荐根因层, 说明)
ATTRIBUTION_MATRIX: List[tuple] = [
    ("子任务依赖存在环", FailureLayer.PLANNING, "规划器产出非法 DAG"),
    ("工具不存在", FailureLayer.PLANNING, "分解引用了未注册工具"),
    ("前置条件不满足", FailureLayer.PLANNING, "计划缺少满足前置的子任务（不可行计划）"),
    ("动作偏离全局目标", FailureLayer.ACTION, "Reasoner 调用了未对齐的工具"),
    ("前置校验阻断", FailureLayer.PLANNING, "执行期前置不满足，根因多为计划缺少满足前置的子任务"),
    ("VLA 时延", FailureLayer.INFERENCE, "端侧推理时延不达标"),
    ("云端降级", FailureLayer.NETWORK, "网络抖动触发边云回退"),
    ("目标歧义", FailureLayer.INTERACTION, "用户指令不可判定"),
]


@dataclass
class Attribution:
    symptom: str
    layer: FailureLayer
    reason: str


def attribute(symptom: str) -> Attribution:
    s = symptom.lower()
    for kw, layer, reason in ATTRIBUTION_MATRIX:
        if kw.lower() in s:
            return Attribution(symptom, layer, reason)
    return Attribution(symptom, FailureLayer.INTERACTION, "未匹配到已知模式，需人工研判")
