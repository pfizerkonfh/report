"""硬件在环仿真适配器（Hardware-In-the-Loop Simulation Adapter）。

它做什么
--------
对上层算法而言，这个适配器**表现得像真机**：

- 调用会**真实消耗墙钟时间**（可按 time_scale 加速，1x 即实时）；
- 真机状态**跨步骤持续演化**：电量真的随飞行时长下降、机器人位置真的移动、
  链路真的可能中断、低电真的会返航失败；
- 每一步都回传**遥测**（电量、位置、里程、链路强度、各相位耗时）；
- 长任务**分阶段回报进度**（起飞 → 巡航 → 作业 → 返航），前端可实时观察。

它不做什么
----------
它**不是**物理精确仿真（没有动力学/风场/视觉），只是"行为级时序仿真"：
用真机的时序特征与状态演进规律，来验证上层调度逻辑在真实时间尺度下是否成立。
物理精度应交给 Gazebo / Isaac Sim，见 `ros2.py` 的真机通道。

为什么这一步关键
----------------
Mock 模式下所有动作零耗时且必然成功 —— 这会掩盖真机上必然出现的三类问题：
1. 长任务期间世界状态已变（电量掉了、目标被占用），但算法还在按旧快照决策；
2. 超时与失败是常态，重规划/Reflexion 路径从未被真实触发过；
3. 端到端时延由"串行叠加的长任务"主导，端侧优化（65ms VLA）在其中占比可能极小。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...config.settings import VLASettings
from ...core.types import Effect, Platform
from ...tools.latency import simulate_vla_latency
from .base import (
    AdapterContext, AdapterOutcome, AdapterStatus, SimClock, ToolAdapter,
)


# ---------------------------------------------------------------------------
# 真机动力学参数（行为级，量级贴近真实平台）
# ---------------------------------------------------------------------------
@dataclass
class Dynamics:
    """平台动力学：速度与能耗。数值量级参考典型工业无人机 / 四足机器人。"""
    uav_speed_mps: float = 8.0        # 无人机巡航速度
    dog_speed_mps: float = 1.2        # 机器狗地面速度（含避障绕行已计入）
    uav_drain_pct_per_s: float = 0.35  # 无人机飞行耗电 %/秒
    dog_drain_pct_per_s: float = 0.12  # 机器狗运动耗电 %/秒
    takeoff_ms: float = 5_000.0
    land_ms: float = 4_000.0
    preflight_ms: float = 2_000.0     # 起飞前自检 / 航线校验
    boot_ms: float = 1_500.0          # 地面机器人上电自检
    min_battery_pct: float = 15.0     # 低于此电量拒绝起飞并触发返航
    charge_ms: float = 60_000.0       # 充满耗时
    vla_cycles: int = 20              # VLA 控制回路周期数（20Hz × 1s）


@dataclass
class SimProfile:
    """单个工具的行为时序画像。"""
    platform: Platform
    base_ms: float                  # 作业本身耗时（拍摄 / 推理 / 处理）
    travel_m: float = 0.0           # 需要移动的距离（0 = 原地或纯算力）
    takeoff: bool = False           # 是否需要起飞（无人机）
    return_home: bool = False       # 作业后是否返航
    compute: bool = False           # 纯算力任务（不移动、不耗平台电量）
    jitter_ms: float = 0.0          # 随机抖动（建模真实环境的非确定性）
    label: str = ""                 # 作业相位名称（用于进度展示）


# 工具 -> 行为画像（按平台区分，体现空地协同的差异）
def _profiles(dyn: Dynamics) -> Dict[str, Dict[str, SimProfile]]:
    return {
        "weather": {"default": SimProfile(Platform.SHARED, base_ms=800, compute=True,
                                          jitter_ms=300, label="气象与航线查询")},
        "camera": {
            "uav": SimProfile(Platform.UAV, base_ms=8_000, travel_m=120.0, takeoff=True,
                              return_home=True, jitter_ms=1_500, label="高空航拍采集"),
            "dog": SimProfile(Platform.DOG, base_ms=12_000, travel_m=40.0, takeoff=False,
                              return_home=False, jitter_ms=2_000, label="地面近距细查"),
            "default": SimProfile(Platform.UAV, base_ms=8_000, travel_m=120.0, takeoff=True,
                                  return_home=True, jitter_ms=1_500, label="航拍采集"),
        },
        "edit": {"default": SimProfile(Platform.SHARED, base_ms=45_000, compute=True,
                                       jitter_ms=8_000, label="云端剪辑渲染")},
        "anonymize": {"default": SimProfile(Platform.SHARED, base_ms=6_000, compute=True,
                                            jitter_ms=1_200, label="边侧人脸脱敏")},
        "vla": {"default": SimProfile(Platform.SHARED, base_ms=0.0, compute=True,
                                      jitter_ms=0.0, label="端侧 VLA 控制回路")},
        "charge": {"default": SimProfile(Platform.SHARED, base_ms=dyn.charge_ms, compute=True,
                                         label="自主充电")},
        "cloud_llm": {"default": SimProfile(Platform.SHARED, base_ms=2_500, compute=True,
                                            jitter_ms=800, label="云端大模型分析")},
    }


@dataclass
class SimWorld:
    """跨步骤持续演化的仿真世界（真机状态）。

    这是与 Mock 的本质区别之一：Mock 每次调用彼此独立，而真机的电量、位置、
    里程是**连续累积**的 —— 第 5 步还能不能飞，取决于前 4 步飞了多久。
    """

    dynamics: Dynamics = field(default_factory=Dynamics)
    battery: Dict[str, float] = field(default_factory=lambda: {"uav": 100.0, "dog": 100.0})
    pos: Dict[str, Tuple[float, ...]] = field(
        default_factory=lambda: {"uav": (0.0, 0.0, 0.0), "dog": (0.0, 0.0)})
    odometer_m: Dict[str, float] = field(default_factory=lambda: {"uav": 0.0, "dog": 0.0})
    busy_ms: Dict[str, float] = field(default_factory=lambda: {"uav": 0.0, "dog": 0.0, "shared": 0.0})
    link_rssi_dbm: float = -55.0
    faults_injected: List[str] = field(default_factory=list)
    _rng: Optional[random.Random] = None

    def rng(self, seed: Optional[int]) -> random.Random:
        if self._rng is None:
            self._rng = random.Random(seed)
        return self._rng

    def reset(self, battery: float = 100.0) -> None:
        self.battery = {"uav": battery, "dog": battery}
        self.pos = {"uav": (0.0, 0.0, 0.0), "dog": (0.0, 0.0)}
        self.odometer_m = {"uav": 0.0, "dog": 0.0}
        self.busy_ms = {"uav": 0.0, "dog": 0.0, "shared": 0.0}
        self.link_rssi_dbm = -55.0
        self.faults_injected = []


class SimulatedAdapter(ToolAdapter):
    """硬件在环仿真适配器。

    真实时序语义：
    - 每个相位真实 sleep（受 time_scale 加速），取消请求 ~50ms 内响应；
    - 超过 timeout_ms 判 TIMEOUT（不做"假装成功"）；
    - 失败/链路中断按配置概率注入，走与真机一致的失败通道。
    """

    name = "hil-sim"
    real_hardware = False

    def __init__(
        self,
        dynamics: Optional[Dynamics] = None,
        time_scale: float = 10.0,
        failure_rate: float = 0.0,
        comm_loss_rate: float = 0.0,
        timeout_ms: float = 30_000.0,
        seed: Optional[int] = None,
        world: Optional[SimWorld] = None,
        vla_settings: Any = None,
    ) -> None:
        self.dyn = dynamics or Dynamics()
        self.time_scale = time_scale if time_scale and time_scale > 0 else 1.0
        self.failure_rate = failure_rate
        self.comm_loss_rate = comm_loss_rate
        self.timeout_ms = timeout_ms
        self.seed = seed
        self.world = world or SimWorld(dynamics=self.dyn)
        self.vla_settings = vla_settings or VLASettings()
        self._profiles = _profiles(self.dyn)

    # ---- 元数据 ----

    def health(self) -> Dict[str, Any]:
        return {
            "ok": True, "adapter": self.name, "real_hardware": False,
            "time_scale": self.time_scale,
            "battery": {k: round(v, 1) for k, v in self.world.battery.items()},
            "odometer_m": {k: round(v, 1) for k, v in self.world.odometer_m.items()},
        }

    # ---- 主入口 ----

    def invoke(self, ctx: AdapterContext) -> AdapterOutcome:
        t0 = _now()
        prof = self._profile(ctx.tool_name, ctx.platform)
        rng = self.world.rng(self.seed)
        plat = prof.platform.value

        # 超时预算以本次调用为准（ctx 由上层按任务类型下发），适配器的默认值仅作兜底
        budget_ms = ctx.timeout_ms if ctx.timeout_ms and ctx.timeout_ms > 0 else self.timeout_ms
        telemetry: Dict[str, Any] = {
            "adapter": self.name,
            "platform": plat,
            "tier": ctx.tier.value,
            "phases": [],
            "simulated": True,
            "timeout_ms": budget_ms,
        }

        # 1) 链路丢失：真机最常见的"任务根本没开始/中途失联"
        if self._roll(rng, self.comm_loss_rate):
            telemetry["link_rssi_dbm"] = -105.0
            return self._finish(
                t0, AdapterStatus.LINK_LOST, False,
                f"[{plat}] 数传链路中断（RSSI -105dBm），任务未执行",
                telemetry, error="link_lost")

        # 2) 构造相位序列（真机行为：准备 -> 移动 -> 作业 -> 归位）
        phases = self._build_phases(ctx, prof, rng)
        planned_ms = sum(p[1] for p in phases)

        # 3) 起飞前电量检查：真机会拒绝起飞，而不是飞到一半坠机
        if prof.takeoff and self.world.battery["uav"] < self.dyn.min_battery_pct:
            return self._finish(
                t0, AdapterStatus.FAILED, False,
                f"[无人机] 电量 {self.world.battery['uav']:.0f}% 低于安全阈值 "
                f"{self.dyn.min_battery_pct:.0f}%，拒绝起飞并返航",
                telemetry, error="low_battery")

        # 4) 真实地"执行"：逐相位消耗真实墙钟时间
        done_ms = 0.0
        for phase_name, phase_ms, meta in phases:
            # 超时判定基于"仿真时长"是否超过预算（换算后等价于真机超时）
            if done_ms + phase_ms > budget_ms:
                telemetry["timeout_at_phase"] = phase_name
                telemetry["progress"] = round(done_ms / max(planned_ms, 1e-9), 3)
                return self._finish(
                    t0, AdapterStatus.TIMEOUT, False,
                    f"[{plat}] 在「{phase_name}」相位超时（>{budget_ms / 1000:.1f}s 仿真时长，"
                    f"已完成 {done_ms / 1000:.1f}s）",
                    telemetry, error="timeout", sim_duration_ms=done_ms)

            ok = ctx.clock.sleep(phase_ms, token=ctx.token,
                                 progress_cb=self._progress_cb(ctx, done_ms, planned_ms, phase_ms),
                                 phase=phase_name)
            if not ok:  # 被取消
                telemetry["progress"] = round(done_ms / max(planned_ms, 1e-9), 3)
                return self._finish(
                    t0, AdapterStatus.CANCELLED, False,
                    f"[{plat}] 任务在「{phase_name}」被中止（{ctx.token.reason if ctx.token else ''}）",
                    telemetry, error="cancelled", sim_duration_ms=done_ms)

            done_ms += phase_ms
            self._apply_phase_effect(phase_name, phase_ms, prof, plat, rng)
            telemetry["phases"].append({
                "name": phase_name, "sim_ms": round(phase_ms),
                "label": meta.get("label", phase_name),
            })

        # 5) 任务失败注入：真机作业也可能没达成（素材模糊、目标未识别）
        if self._roll(rng, self.failure_rate):
            self.world.faults_injected.append(f"{ctx.tool_name}:task_failed")
            return self._finish(
                t0, AdapterStatus.FAILED, False,
                f"[{plat}] {prof.label or ctx.tool_name} 未达成作业判据（已注入故障）",
                telemetry, error="task_failed")

        # 6) 成功：回写真机遥测与实际副作用
        self.world.busy_ms[plat] = self.world.busy_ms.get(plat, 0.0) + done_ms
        telemetry.update(self._telemetry_snapshot(prof, plat, done_ms))

        if ctx.tool_name == "charge":
            self.world.battery["uav"] = 100.0
            self.world.battery["dog"] = 100.0

        obs = self._observation(ctx, prof, plat, done_ms, telemetry)
        return self._finish(t0, AdapterStatus.SUCCEEDED, True, obs, telemetry,
                            effects=self._effects(ctx, prof, telemetry),
                            sim_duration_ms=done_ms)

    # ---- 内部：相位构造 ----

    def _build_phases(self, ctx: AdapterContext, prof: SimProfile,
                      rng: random.Random) -> List[Tuple[str, float, Dict[str, Any]]]:
        """把一次作业拆成真机的相位序列。"""
        dist = float(ctx.args.get("distance_m") or prof.travel_m or 0.0)
        phases: List[Tuple[str, float, Dict[str, Any]]] = []

        if prof.compute:
            base = prof.base_ms
            if ctx.tool_name == "vla":
                # VLA：以端侧优化后的单帧时延 × 控制周期数构成一段控制回路
                bd = simulate_vla_latency(self.vla_settings)
                base = bd.final * self.dyn.vla_cycles
            phases.append(("submit", 200.0, {"label": "下发任务"}))
            phases.append(("process", self._jitter(base, prof.jitter_ms, rng),
                           {"label": prof.label or "计算处理"}))
            phases.append(("collect", 200.0, {"label": "回收结果"}))
            return phases

        if prof.takeoff:
            phases.append(("preflight", self.dyn.preflight_ms, {"label": "起飞前自检"}))
            phases.append(("takeoff", self.dyn.takeoff_ms, {"label": "起飞"}))
        else:
            phases.append(("boot", self.dyn.boot_ms, {"label": "上电自检"}))

        speed = self.dyn.uav_speed_mps if prof.platform == Platform.UAV else self.dyn.dog_speed_mps
        travel_ms = (dist / speed * 1000.0) if dist > 0 and speed > 0 else 0.0
        if travel_ms > 0:
            phases.append(("cruise_out", travel_ms, {"label": f"前往目标点（{dist:.0f}m）"}))

        phases.append(("work", self._jitter(prof.base_ms, prof.jitter_ms, rng),
                       {"label": prof.label or "作业"}))

        if prof.return_home and travel_ms > 0:
            phases.append(("cruise_back", travel_ms, {"label": "返航"}))
        if prof.takeoff:
            phases.append(("land", self.dyn.land_ms, {"label": "降落"}))
        return phases

    def _apply_phase_effect(self, phase_name: str, phase_ms: float, prof: SimProfile,
                            plat: str, rng: random.Random) -> None:
        """相位完成后推进真机状态：耗电、位移、里程、链路波动。"""
        secs = phase_ms / 1000.0

        # 电量：只有移动/飞行相位耗电，纯算力不耗平台电池
        if not prof.compute:
            if plat == "uav" and phase_name in ("takeoff", "cruise_out", "cruise_back", "work", "land"):
                self.world.battery["uav"] -= self.dyn.uav_drain_pct_per_s * secs
            elif plat == "dog" and phase_name in ("cruise_out", "work"):
                self.world.battery["dog"] -= self.dyn.dog_drain_pct_per_s * secs
        self.world.battery["uav"] = max(0.0, min(100.0, self.world.battery["uav"]))
        self.world.battery["dog"] = max(0.0, min(100.0, self.world.battery["dog"]))

        # 位移与里程
        if phase_name in ("cruise_out", "cruise_back"):
            speed = self.dyn.uav_speed_mps if plat == "uav" else self.dyn.dog_speed_mps
            dist = speed * secs
            self.world.odometer_m[plat] = self.world.odometer_m.get(plat, 0.0) + dist
            if plat == "uav":
                x, y, z = self.world.pos["uav"]
                alt = 25.0 if phase_name == "cruise_out" else 0.0
                self.world.pos["uav"] = (round(x + dist * 0.7, 1), round(y + dist * 0.7, 1), alt)
            else:
                x, y = self.world.pos["dog"]
                self.world.pos["dog"] = (round(x + dist * 0.8, 1), round(y + dist * 0.6, 1))

        # 链路波动：真机 RSSI 会随距离/遮挡漂移
        self.world.link_rssi_dbm = round(
            max(-110.0, min(-40.0, self.world.link_rssi_dbm + rng.uniform(-4.0, 4.0))), 1)

    # ---- 内部：输出构造 ----

    def _telemetry_snapshot(self, prof: SimProfile, plat: str, done_ms: float) -> Dict[str, Any]:
        snap = {
            "battery": {k: round(v, 1) for k, v in self.world.battery.items()},
            "odometer_m": {k: round(v, 1) for k, v in self.world.odometer_m.items()},
            "position": {k: list(v) for k, v in self.world.pos.items()},
            "link_rssi_dbm": self.world.link_rssi_dbm,
            "platform_busy_ms": {k: round(v) for k, v in self.world.busy_ms.items()},
            "distance_m": round(prof.travel_m, 1),
        }
        if plat in ("uav", "dog"):
            snap["battery_after"] = round(self.world.battery[plat], 1)
        return snap

    def _observation(self, ctx: AdapterContext, prof: SimProfile, plat: str,
                     done_ms: float, telemetry: Dict[str, Any]) -> str:
        label = prof.label or ctx.tool_name
        if prof.compute:
            return f"[{plat}] {label}完成，任务耗时 {done_ms / 1000:.1f}s（{self.time_scale}x 加速仿真）"
        batt = telemetry.get("battery_after")
        return (f"[{plat}] {label}完成：里程 +{prof.travel_m:.0f}m，"
                f"剩余电量 {batt:.0f}%，仿真耗时 {done_ms / 1000:.1f}s")

    def _effects(self, ctx: AdapterContext, prof: SimProfile,
                 telemetry: Dict[str, Any]) -> List[Effect]:
        """真机实际产生的副作用 = 语义契约 + 物理遥测（物理量覆盖语义声明）。

        只回传电量而不回传"素材已生成"，会让后续剪辑/脱敏被前置校验全线阻断 ——
        必须把工具声明的语义 Effect 一并回写，物理类事实（电量）则以遥测为准覆盖。
        """
        merged: Dict[str, Effect] = {e.key: e for e in (ctx.declared_effects or [])}
        merged["battery_uav"] = Effect(key="battery_uav", op="set",
                                       value=round(self.world.battery["uav"], 1))
        merged["battery_dog"] = Effect(key="battery_dog", op="set",
                                       value=round(self.world.battery["dog"], 1))
        # 单值 battery 取两平台较低者：前置条件 battery>=N 表示"两个平台都够用"
        merged["battery"] = Effect(
            key="battery", op="set",
            value=round(min(self.world.battery["uav"], self.world.battery["dog"]), 1))
        if ctx.tool_name == "vla":
            merged["control_cmd"] = Effect(key="control_cmd", op="set", value="move(0.2,0,0)")
        return list(merged.values())

    def _finish(self, t0: float, status: AdapterStatus, success: bool, obs: str,
                telemetry: Dict[str, Any], error: str = "",
                effects: Optional[List[Effect]] = None,
                sim_duration_ms: float = 0.0) -> AdapterOutcome:
        wall = (_now() - t0) * 1000.0
        telemetry["_wall_ms"] = round(wall, 2)
        telemetry["time_scale"] = self.time_scale
        telemetry["status"] = status.value
        # 失败/超时也要回传真机状态，否则上层无法诊断"是没电了还是链路断了"
        telemetry.setdefault("battery", {k: round(v, 1) for k, v in self.world.battery.items()})
        telemetry.setdefault("odometer_m", {k: round(v, 1) for k, v in self.world.odometer_m.items()})
        telemetry.setdefault("link_rssi_dbm", self.world.link_rssi_dbm)
        telemetry.setdefault("platform_busy_ms",
                             {k: round(v) for k, v in self.world.busy_ms.items()})
        return AdapterOutcome(status=status, success=success, observation=obs,
                              latency_ms=wall,
                              sim_duration_ms=sim_duration_ms if sim_duration_ms > 0 else wall,
                              telemetry=telemetry, effects=effects, error=error)

    # ---- 内部：工具 ----

    def _profile(self, tool_name: str, platform: Platform) -> SimProfile:
        table = self._profiles.get(tool_name, {})
        return table.get(platform.value) or table.get("default") or SimProfile(
            platform=platform, base_ms=1_000.0, compute=True, label=tool_name)

    def _progress_cb(self, ctx: AdapterContext, done_ms: float, planned_ms: float,
                     phase_ms: float) -> Optional[Callable[[float, str], None]]:
        """把"当前相位的完成比例"换算为"整体进度"，供前端实时展示。"""
        if ctx.progress_cb is None:
            return None
        total = max(planned_ms, 1e-9)

        def cb(frac: float, phase: str) -> None:
            overall = min(1.0, (done_ms + frac * phase_ms) / total)
            ctx.progress_cb(overall, phase)  # type: ignore[misc]

        return cb

    @staticmethod
    def _jitter(base: float, jitter_ms: float, rng: random.Random) -> float:
        if jitter_ms <= 0:
            return max(0.0, base)
        return max(0.0, base + rng.uniform(-jitter_ms, jitter_ms))

    @staticmethod
    def _roll(rng: random.Random, rate: float) -> bool:
        return rate > 0 and rng.random() < rate


def _now() -> float:
    return time.perf_counter()


def make_clock(time_scale: float) -> SimClock:
    return SimClock(time_scale=time_scale)
