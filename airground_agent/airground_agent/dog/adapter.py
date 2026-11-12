"""机器狗真机适配器：让决策平台的工具调用直接落到 ZSIBOT 机器狗上。

挂在既有 ToolAdapter 体系上（`AdapterMode.ZSIBOT`），因此：
- Planner / Reasoner / Router / 评测**完全不用改**，切换 `sim.mode=zsibot` 即可接真狗；
- 真机语义（真实耗时、真实超时、可取消、遥测回传）由本适配器保证；
- 急停走 `emergency_stop()`：先停导航再软急停，与决策平台的急停语义一致。

工具 -> 真机动作映射
--------------------
真机世界里"工具"必须落到具体动作，映射不了就明确失败（不做假成功）：

- `camera`（平台判给机器狗时）-> 导航到目标点 + 到位采集窗口
- `route_plan` / 带 waypoints 的调用 -> 逐点导航
- `charge` -> 返航充电桩
- `vla` -> 端侧推理，不下发狗体（返回成功但不产生真机动作）
- 其余（weather / edit / anonymize / cloud_llm）-> 不属于机器狗，返回 unmapped 失败

坐标：入参是世界系（3DGS，X-Z 水平），下发前统一经 `FrameTransform` 换成狗地图系。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from ..core.types import Effect, Platform
from ..tools.adapters.base import AdapterContext, AdapterOutcome, AdapterStatus, ToolAdapter
from .frames import FrameTransform, default_transform
from .transport import DogTransport, build_transport

#: 工具 -> 真机动作类别
TOOL_TO_DOG: Dict[str, str] = {
    "camera": "nav_and_collect",
    "route_plan": "navigate",
    "charge": "return_home",
    "vla": "edge_only",
}

#: 机器狗不承载的工具（明确拒绝，交由无人机或云端）
DOG_OUT_OF_SCOPE = {"weather", "edit", "anonymize", "cloud_llm"}


class ZsibotDogAdapter(ToolAdapter):
    """ZSIBOT 机器狗真机通道。"""

    name = "zsibot-dog"
    real_hardware = True

    def __init__(self, transport: Optional[DogTransport] = None,
                 frame: Optional[FrameTransform] = None,
                 collect_seconds: float = 3.0,
                 nav_timeout_s: float = 120.0) -> None:
        self.transport = transport or DogTransport.dry()
        self.frame = frame or default_transform()
        self.collect_seconds = collect_seconds
        self.nav_timeout_s = nav_timeout_s

    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return {"ok": True, "adapter": self.name, "real_hardware": True,
                **self.transport.health()}

    def abort(self, reason: str = "急停") -> None:
        """真机急停：先停导航，再软急停。"""
        try:
            self.transport.emergency_stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def invoke(self, ctx: AdapterContext) -> AdapterOutcome:
        t0 = time.perf_counter()
        tool = ctx.tool_name
        if tool in DOG_OUT_OF_SCOPE:
            return self._fail(t0, ctx, f"工具 {tool} 不属于机器狗平台（应由无人机/云端承载）",
                              "out_of_scope")

        action = TOOL_TO_DOG.get(tool)
        if action is None:
            return self._fail(t0, ctx, f"工具 {tool} 未映射机器狗动作", "unmapped")

        if action == "edge_only":
            wall = (time.perf_counter() - t0) * 1000.0
            return AdapterOutcome(
                status=AdapterStatus.SUCCEEDED, success=True,
                observation=f"{tool} 在端侧完成，未下发机器狗动作",
                latency_ms=wall, sim_duration_ms=wall,
                telemetry={"adapter": self.name, "dog_action": "none"},
                effects=self._merge(ctx, telemetry={}))

        if ctx.platform != Platform.DOG and ctx.platform is not None:
            # 平台判给了无人机却走到狗通道：明确失败，避免误下发
            return self._fail(t0, ctx,
                              f"工具 {tool} 的平台是 {ctx.platform.value}，不下发给机器狗",
                              "platform_mismatch")

        if ctx.token is not None and ctx.token.is_cancelled():
            return AdapterOutcome(status=AdapterStatus.CANCELLED, success=False,
                                  observation="任务被中止", latency_ms=0.0,
                                  telemetry={"adapter": self.name},
                                  error="cancelled")

        # ---- 返航 ----
        if action == "return_home":
            r = self.transport.nav_request({"start_nav_return_home": None})
            ok, state, _ = self.transport.wait_nav_idle(timeout=self.nav_timeout_s,
                                                        token=ctx.token)
            wall = (time.perf_counter() - t0) * 1000.0
            status = AdapterStatus.SUCCEEDED if (r.ok and ok) else AdapterStatus.FAILED
            return AdapterOutcome(
                status=status, success=status == AdapterStatus.SUCCEEDED,
                observation=f"返航指令已下发，导航状态={state}",
                latency_ms=wall, sim_duration_ms=wall,
                telemetry={"adapter": self.name, "nav_state": state,
                           "dog_action": "return_home"},
                effects=self._merge(ctx, {}) if status == AdapterStatus.SUCCEEDED else None,
                error="" if r.ok and ok else f"返航失败（状态={state}）")

        # ---- 导航（+ 采集）----
        waypoints = self._waypoints_from_args(ctx.args)
        if not waypoints:
            return self._fail(t0, ctx,
                              f"{tool} 缺少可下发的目标坐标（需要 waypoints / target / x+z）",
                              "no_target")
        nav_ok, states = self._navigate(waypoints, ctx)
        wall = (time.perf_counter() - t0) * 1000.0

        if not nav_ok:
            return AdapterOutcome(
                status=AdapterStatus.FAILED, success=False,
                observation=f"机器狗导航未完成（末态={states[-1] if states else '未知'}）",
                latency_ms=wall, sim_duration_ms=wall,
                telemetry={"adapter": self.name, "nav_states": states,
                           "waypoints": len(waypoints)},
                error="nav_incomplete")

        if action == "nav_and_collect":
            self._sleep(self.collect_seconds, ctx.token)

        telemetry = self._read_status()
        telemetry.update({"adapter": self.name, "nav_states": states,
                          "waypoints": len(waypoints), "dog_action": action})
        return AdapterOutcome(
            status=AdapterStatus.SUCCEEDED, success=True,
            observation=(f"机器狗已完成 {len(waypoints)} 个航点导航"
                         + (f"并采集 {self.collect_seconds:.0f}s" if action == "nav_and_collect" else "")),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            sim_duration_ms=(time.perf_counter() - t0) * 1000.0,
            telemetry=telemetry,
            effects=self._merge(ctx, telemetry))

    # ------------------------------------------------------------------
    def _navigate(self, waypoints: Sequence[Sequence[float]], ctx: AdapterContext) -> tuple:
        states: List[str] = []
        for i, p in enumerate(waypoints):
            if ctx.token is not None and ctx.token.is_cancelled():
                self.transport.emergency_stop()
                return False, states
            X, Y, Z = self.frame.world_to_map(p[0], p[1], p[2])
            req: Dict[str, Any] = {"start_nav": {"position": {"x": X, "y": Y, "z": Z}}}
            if i + 1 < len(waypoints):
                nxt = waypoints[i + 1]
                req["start_nav"]["orientation"] = self.frame.orientation_towards(p, nxt)
            r = self.transport.nav_request(req)
            if not r.ok:
                return False, states + [r.error or "request_failed"]
            ok, state, _ = self.transport.wait_nav_idle(timeout=self.nav_timeout_s,
                                                        token=ctx.token)
            states.append(state)
            if not ok:
                return False, states
            if ctx.progress_cb is not None:
                try:
                    ctx.progress_cb((i + 1) / len(waypoints), f"航点 {i + 1}/{len(waypoints)}")
                except Exception:
                    pass
        return True, states

    def _waypoints_from_args(self, args: Dict[str, Any]) -> List[List[float]]:
        """从 Planner 给的参数里取世界系航点：waypoints / target / x+z。"""
        wps = args.get("waypoints")
        if isinstance(wps, (list, tuple)) and wps and isinstance(wps[0], (list, tuple)):
            return [[float(p[0]), float(p[1]), float(p[2])] for p in wps]
        tgt = args.get("target")
        if isinstance(tgt, (list, tuple)) and len(tgt) >= 3:
            return [[float(tgt[0]), float(tgt[1]), float(tgt[2])]]
        if args.get("x") is not None and args.get("z") is not None:
            return [[float(args["x"]), float(args.get("y", 0.0)), float(args["z"])]]
        gp = args.get("goal")
        if isinstance(gp, (list, tuple)) and len(gp) >= 3:
            return [[float(gp[0]), float(gp[1]), float(gp[2])]]
        return []

    def _read_status(self) -> Dict[str, Any]:
        try:
            r = self.transport.sdk_send("get_status", timeout=5.0)
            if r.ok and isinstance(r.data, dict):
                return {"robot_status": r.data,
                        "battery": _first_num(r.data, "power", "battery_power", "battery")}
        except Exception:
            pass
        return {}

    def _merge(self, ctx: AdapterContext, telemetry: Dict[str, Any]) -> List[Effect]:
        """真机副作用 = 工具声明的语义副作用 + 真机遥测（电量）。

        只回传电量不回传语义效果（footage/edited…）会让后续步骤的前置校验全线阻断，
        这是硬件在环仿真里踩过的坑，这里显式合并。
        """
        effects: List[Effect] = list(ctx.declared_effects or [])
        batt = telemetry.get("battery")
        if batt is not None:
            effects.append(Effect(key="battery", op="set", value=round(float(batt), 1)))
        return effects

    @staticmethod
    def _sleep(seconds: float, token: Any = None) -> None:
        step, elapsed = 0.05, 0.0
        while elapsed < seconds:
            if token is not None and token.is_cancelled():
                return
            time.sleep(min(step, seconds - elapsed))
            elapsed += step

    def _fail(self, t0: float, ctx: AdapterContext, msg: str, code: str) -> AdapterOutcome:
        return AdapterOutcome(status=AdapterStatus.FAILED, success=False, observation=msg,
                              latency_ms=(time.perf_counter() - t0) * 1000.0,
                              telemetry={"adapter": self.name, "platform": ctx.platform.value},
                              error=code)


def _first_num(data: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        v = data.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def build_dog_adapter(uri: str = "", exe_path: str = "", dog_ip: str = "",
                      send_port: int = 8081, recv_port: int = 8080,
                      dry_run: bool = True, frame: Optional[FrameTransform] = None,
                      collect_seconds: float = 3.0,
                      nav_timeout_s: float = 120.0) -> ZsibotDogAdapter:
    """便捷构造：按配置装配真机通道。"""
    transport = build_transport(uri=uri, exe_path=exe_path, dog_ip=dog_ip,
                                send_port=send_port, recv_port=recv_port, dry_run=dry_run)
    return ZsibotDogAdapter(transport=transport, frame=frame,
                            collect_seconds=collect_seconds, nav_timeout_s=nav_timeout_s)
