"""评测闭环：离线回放 + 批量套件 + 聚合。

- 离线回放（replay）：用记录的 trace 逐层二分，定位 badcase 根因层；
- 批量套件（run_suite）：对一组目标/场景跑出 EvalRecord 序列；
- 聚合（aggregate）：给出四指标均值与成功率分布。
"""
from __future__ import annotations

from typing import Callable, Dict, List

from .metrics import EvalRecord
from .attribution import attribute
from ..core.log import get_logger

logger = get_logger("eval")


class EvalHarness:
    def __init__(self) -> None:
        self.records: List[EvalRecord] = []

    def add(self, rec: EvalRecord) -> None:
        self.records.append(rec)

    def aggregate(self) -> Dict:
        if not self.records:
            return {}
        n = len(self.records)
        succ = sum(1 for r in self.records if r.status.value == "success")
        return {
            "runs": n,
            "planning_success_rate": round(sum(r.planning_success for r in self.records) / n, 4),
            "task_success_rate": round(succ / n, 4),
            "avg_action_reasonableness": round(sum(r.action_reasonableness for r in self.records) / n, 4),
            "avg_precondition_violations": round(sum(r.precondition_violations for r in self.records) / n, 4),
            "avg_latency_p50": round(sum(r.latency_p50 for r in self.records) / n, 2),
            "avg_latency_p95": round(sum(r.latency_p95 for r in self.records) / n, 2),
            "avg_latency_p99": round(sum(r.latency_p99 for r in self.records) / n, 2),
            "avg_end_to_end_ms": round(sum(r.end_to_end_ms for r in self.records) / n, 2),
            "avg_csat": round(sum(r.csat for r in self.records) / n, 4),
        }

    def replay(self, trace) -> Dict:
        """离线回放：逐层二分定位根因层。"""
        bad = [t for t in trace if not t.precondition_ok or (t.action and t.action.get("tool") is None)]
        findings = []
        for t in bad:
            sym = t.observation or "未知异常"
            findings.append({"step": t.step_id, "symptom": sym,
                            "layer": attribute(sym).layer.value, "reason": attribute(sym).reason})
        return {"bad_steps": len(bad), "findings": findings}

    def run_suite(self, cases: List[Dict], run_fn: Callable[[str, dict], "object"]) -> List[EvalRecord]:
        out = []
        for c in cases:
            res = run_fn(c["goal"], c.get("ws", {}))
            rec = getattr(res, "metrics", None)
            if rec:
                self.add(rec)
                out.append(rec)
        return out
