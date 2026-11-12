"""录播回放适配器 —— 用现场真实记录的数据驱动系统复现。

用途
----
把真机/仿真实跑时采集的**真实时序与遥测**录成 JSONL，之后可反复回放：
- 算法改动后，用**同一份现场数据**对比前后差异（A/B 测试不受真机可用性影响）；
- 复现某次现场故障（链路中断 / 低电返航），验证重规划与 Reflexion 路径；
- 汇报演示时无需真机在场。

与 SIM 的区别
------------
SIM 是"模型生成时序"，Playback 是"真实记录的时序"。后者更可信，但只覆盖录制过的场景。

JSONL 格式（每行一条）
----------------------
{"tool":"camera","platform":"uav","sim_ms":42000,"success":true,
 "observation":"...","telemetry":{"battery_after":72,"odometer_m":240}}
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from ...core.types import Effect
from .base import (
    AdapterContext, AdapterOutcome, AdapterStatus, CancellationToken, SimClock,
    ToolAdapter,
)


class PlaybackAdapter(ToolAdapter):
    """按记录的真实时序回放，不做随机、不发生未记录的故障（确定性）。"""

    name = "playback"
    real_hardware = False

    def __init__(self, file_path: str, time_scale: float = 10.0) -> None:
        self.file_path = file_path
        self.clock = SimClock(time_scale=time_scale)
        self._records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._cursor: Dict[str, int] = defaultdict(int)
        self._loaded = False
        self._ok = False
        self._load_error: str = ""

    # ---- 加载 ----

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return self._ok
        self._loaded = True
        if not self.file_path or not os.path.exists(self.file_path):
            self._load_error = f"录播文件不存在: {self.file_path}"
            return False
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("//"):
                        continue
                    rec = json.loads(line)
                    key = self._key(rec.get("tool", ""), rec.get("platform", ""))
                    self._records[key].append(rec)
        except Exception as e:
            self._load_error = f"录播文件解析失败: {e}"
            return False
        self._ok = True
        return True

    @staticmethod
    def _key(tool: str, platform: str) -> str:
        return f"{tool}@{platform or 'any'}"

    def health(self) -> Dict[str, Any]:
        ok = self._ensure_loaded()
        return {
            "ok": ok, "adapter": self.name, "real_hardware": False,
            "file": self.file_path,
            "records": sum(len(v) for v in self._records.values()),
            "error": self._load_error,
        }

    # ---- 回放 ----

    def invoke(self, ctx: AdapterContext) -> AdapterOutcome:
        t0 = time.perf_counter()
        if not self._ensure_loaded():
            return AdapterOutcome(status=AdapterStatus.FAILED, success=False,
                                  observation=self._load_error,
                                  latency_ms=0.0, telemetry={"adapter": self.name},
                                  error="playback_file_missing")

        key = self._key(ctx.tool_name, ctx.platform.value)
        bucket = self._records.get(key) or self._records.get(self._key(ctx.tool_name, ""))
        if not bucket:
            return AdapterOutcome(
                status=AdapterStatus.FAILED, success=False,
                observation=f"录播数据中没有 {ctx.tool_name}@{ctx.platform.value} 的记录",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                telemetry={"adapter": self.name, "requested": key},
                error="no_record")

        # 同一工具多次出现时按顺序取用，用尽后循环（保证长计划也能跑完）
        idx = self._cursor[key] % len(bucket)
        self._cursor[key] += 1
        rec = bucket[idx]

        sim_ms = float(rec.get("sim_ms", rec.get("latency_ms", 0.0)) or 0.0)
        # 真实消耗时间（按加速比压缩），支持中途取消
        ok = self.clock.sleep(sim_ms, token=ctx.token,
                              progress_cb=ctx.progress_cb, phase="playback")
        if not ok:
            return AdapterOutcome(
                status=AdapterStatus.CANCELLED, success=False,
                observation="回放被中止", latency_ms=(time.perf_counter() - t0) * 1000.0,
                telemetry={"adapter": self.name, "record": rec}, error="cancelled")

        success = bool(rec.get("success", True))
        status = AdapterStatus.SUCCEEDED if success else AdapterStatus.FAILED
        telemetry = dict(rec.get("telemetry") or {})
        telemetry.update({"adapter": self.name, "source": "playback",
                          "platform": ctx.platform.value, "replay_index": idx})

        return AdapterOutcome(
            status=status, success=success,
            observation=rec.get("observation", f"回放 {ctx.tool_name}（{sim_ms / 1000:.1f}s）"),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            sim_duration_ms=sim_ms, telemetry=telemetry,
            effects=self._effects(ctx, telemetry) if success else None,
            error=rec.get("error", "") if not success else "",
        )

    def _effects(self, ctx: AdapterContext, telemetry: Dict[str, Any]) -> Optional[List[Effect]]:
        """回放副作用 = 语义契约 + 录播遥测（遥测覆盖语义声明的同类事实）。"""
        merged: Dict[str, Effect] = {e.key: e for e in (ctx.declared_effects or [])}
        batt = telemetry.get("battery_after")
        if batt is not None:
            merged["battery"] = Effect(key="battery", op="set", value=round(float(batt), 1))
        odo = telemetry.get("odometer_m")
        if odo is not None:
            merged["odometer_m"] = Effect(key="odometer_m", op="set", value=round(float(odo), 1))
        return list(merged.values()) or None


def default_playback_path() -> str:
    """返回随包提供的示例录播文件（变电站空地协同巡检真实量级时序）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "..", "examples",
                                         "sample_playback.jsonl"))
