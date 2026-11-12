"""世界状态（World State）引擎。

- 单一事实源 `facts: dict`，支持 JSON 序列化，便于跨端-边-云传递与离线回放。
- 提供可判定的 `satisfies(Condition)` 与 `apply(Effect)`，是工具 Precondition/Effect
  语义的运行时载体，也是「每步执行前一致性校验」的数据基础。
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .types import ComparisonOp, Condition, Effect


class WorldState(BaseModel):
    facts: Dict[str, Any] = Field(default_factory=dict)
    revision: int = 0

    # ---- 读取 ----
    def get(self, key: str, default: Any = None) -> Any:
        return self.facts.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.facts

    # ---- 前置判定 ----
    def satisfies(self, cond: Condition) -> bool:
        present = cond.key in self.facts
        if cond.op == ComparisonOp.EXISTS:
            return present
        if cond.op == ComparisonOp.NOT_EXISTS:
            return not present
        if not present:
            return False
        cur = self.facts[cond.key]
        try:
            if cond.op == ComparisonOp.EQ:
                return cur == cond.value
            if cond.op == ComparisonOp.NE:
                return cur != cond.value
            if cond.op == ComparisonOp.GE:
                return float(cur) >= float(cond.value)
            if cond.op == ComparisonOp.GT:
                return float(cur) > float(cond.value)
            if cond.op == ComparisonOp.LE:
                return float(cur) <= float(cond.value)
            if cond.op == ComparisonOp.LT:
                return float(cur) < float(cond.value)
        except (TypeError, ValueError):
            return False
        return False

    def check_all(self, conds: List[Condition]) -> List[Condition]:
        """返回不满足的前置条件列表（空列表表示全部满足）。"""
        return [c for c in conds if not self.satisfies(c)]

    # ---- 效果回写 ----
    def apply(self, eff: Effect) -> None:
        op = eff.op.lower()
        if op == "set":
            self.facts[eff.key] = eff.value
        elif op == "inc":
            self.facts[eff.key] = self.facts.get(eff.key, 0) + eff.value
        elif op == "dec":
            self.facts[eff.key] = self.facts.get(eff.key, 0) - eff.value
        elif op == "delete":
            self.facts.pop(eff.key, None)
        else:
            raise ValueError(f"Unknown effect op: {eff.op}")
        self.revision += 1

    def apply_many(self, effects: List[Effect]) -> None:
        for e in effects:
            self.apply(e)

    # ---- 工具 ----
    def clone(self) -> "WorldState":
        return WorldState(facts=copy.deepcopy(self.facts), revision=self.revision)

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.facts)

    def diff(self, other: "WorldState") -> Dict[str, Any]:
        """返回相对 other 变化的事实，用于观测（Observation）生成。"""
        return {k: v for k, v in self.facts.items() if other.facts.get(k) != v}
