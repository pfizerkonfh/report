"""领域基础类型与声明式 Condition/Effect 语义。

设计要点（对应设计文档问题①）：
- 工具不再只有自然语言 description，而是显式声明 Precondition/Effect/Rollback，
  由 WorldState 引擎做可判定的前置校验与效果回写，从机制上杜绝「工具误调用」。
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RouteTier(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


class Platform(str, Enum):
    """空地协同的物理执行平台维度（与计算层级 RouteTier 正交）。

    - UAV：无人机（空中平台）
    - DOG：机器狗（地面四足平台）
    - SHARED：共享/协同算力（边侧或云端，无专属物理平台）
    """  # noqa: E501
    UAV = "uav"
    DOG = "dog"
    SHARED = "shared"


# 平台别名 -> 枚举值（兼容 LLM/用户的中文与英文表述）
_PLATFORM_ALIASES = {
    "uav": "uav", "drone": "uav", "无人机": "uav", "空中": "uav", "航拍": "uav",
    "飞行器": "uav", "aerial": "uav",
    "dog": "dog", "quadruped": "dog", "四足": "dog", "机器狗": "dog", "机器犬": "dog",
    "地面机器人": "dog", "ground": "dog",
    "shared": "shared", "both": "shared", "协同": "shared", "共享": "shared",
    "算力": "shared", "混合": "shared",
}


def normalize_platform(raw) -> "Platform":
    """把 LLM/用户输入的平台描述归一化为 Platform 枚举（未知/缺省 -> SHARED）。"""
    if raw is None:
        return Platform.SHARED
    s = str(raw).strip().lower()
    if s in _PLATFORM_ALIASES:
        return Platform(_PLATFORM_ALIASES[s])
    # 注意：这里用「航拍/航迹/航空」而非裸「航」，否则"航线可行性"(天气查询)会被误判为无人机
    if any(k in s for k in ("uav", "无人", "航拍", "航迹", "航空", "空中", "aerial", "drone")):
        return Platform.UAV
    if any(k in s for k in ("dog", "狗", "四足", "quadruped", "机器狗", "ground",
                            "地面", "表计", "开关柜", "近距")):
        return Platform.DOG
    return Platform.SHARED


PLATFORM_LABELS = {
    Platform.UAV: "无人机",
    Platform.DOG: "机器狗",
    Platform.SHARED: "协同算力",
}


class ComputeIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PrivacyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"   # 人脸/身份等敏感数据，必须本地（边侧）处理


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"


class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class FailureLayer(str, Enum):
    """统一评测的归因层（对应设计文档问题④）。"""
    PLANNING = "planning"        # 规划层：分解错误 / 可行性缺失
    ACTION = "action"            # Agent 动作层：工具误调用 / 前置违例
    INFERENCE = "inference"      # 端侧推理层：VLA 时延 / 精度
    NETWORK = "network"          # 网络 / 路由层：抖动 / 回退失效
    INTERACTION = "interaction"  # 交互层：目标歧义 / 用户不满


class ComparisonOp(str, Enum):
    EQ = "=="
    NE = "!="
    GE = ">="
    GT = ">"
    LE = "<="
    LT = "<"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class Condition(BaseModel):
    """前置条件：对 WorldState.facts[key] 的判定。"""
    key: str
    op: ComparisonOp = ComparisonOp.EXISTS
    value: Any = None
    description: str = ""

    def human(self) -> str:
        if self.op in (ComparisonOp.EXISTS, ComparisonOp.NOT_EXISTS):
            return f"{self.key} {self.op.value}"
        return f"{self.key} {self.op.value} {self.value!r}"


class Effect(BaseModel):
    """对世界状态的副作用。op ∈ {set, inc, dec, delete}。"""
    key: str
    op: str = "set"          # set | inc | dec | delete
    value: Any = None
    description: str = ""
