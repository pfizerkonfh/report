"""适配器装配工厂：按执行模式构建"工具 -> 执行通道"的映射。

算法层只依赖 `AdapterBundle.get(tool_name)`，因此切换 Mock / 仿真 / 录播 / ROS2
对 Planner、Reasoner、Router、评测完全透明。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...core.types import Effect
from .base import (
    AdapterContext, AdapterMode, AdapterOutcome, AdapterStatus, CancellationToken,
    LegacyMockAdapter, SimClock, ToolAdapter,
)
from .playback import PlaybackAdapter, default_playback_path
from .ros2 import Ros2Adapter
from .sim import Dynamics, SimulatedAdapter, SimWorld


class AdapterBundle:
    """一次运行所使用的执行通道集合。"""

    def __init__(self, default: ToolAdapter, overrides: Optional[Dict[str, ToolAdapter]] = None,
                 mode: AdapterMode = AdapterMode.LEGACY) -> None:
        self.default = default
        self.overrides = overrides or {}
        self.mode = mode

    def get(self, tool_name: str) -> ToolAdapter:
        return self.overrides.get(tool_name, self.default)

    def health(self) -> Dict[str, Any]:
        return {"mode": self.mode.value, **self.default.health()}

    def abort_all(self, reason: str = "急停") -> None:
        for a in [self.default, *self.overrides.values()]:
            try:
                a.abort(reason)
            except Exception:
                pass


class SimConfig:
    """仿真/执行通道配置（可由 settings.sim 或前端覆盖构造）。"""

    def __init__(
        self,
        mode: str = "legacy",
        time_scale: float = 10.0,
        failure_rate: float = 0.0,
        comm_loss_rate: float = 0.0,
        # 默认 120s：一次无人机架次约 49s（自检+起飞+往返+作业+降落），
        # 30s 会在返航相位误判超时 —— 这个默认值是硬件在环仿真跑出来的结论。
        timeout_ms: float = 120_000.0,
        seed: Optional[int] = None,
        playback_file: str = "",
        uav_speed_mps: float = 8.0,
        dog_speed_mps: float = 1.2,
        min_battery_pct: float = 15.0,
    ) -> None:
        self.mode = mode
        self.time_scale = time_scale
        self.failure_rate = failure_rate
        self.comm_loss_rate = comm_loss_rate
        self.timeout_ms = timeout_ms
        self.seed = seed
        self.playback_file = playback_file
        self.uav_speed_mps = uav_speed_mps
        self.dog_speed_mps = dog_speed_mps
        self.min_battery_pct = min_battery_pct

    @classmethod
    def from_obj(cls, obj: Any) -> "SimConfig":
        """从 pydantic model / dict / None 构造，容错优先。"""
        if obj is None:
            return cls()
        data = obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)
        known = {k: v for k, v in data.items() if v is not None}
        return cls(**known)


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """从 pydantic model / dict / 对象里取值，容错优先。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, "model_dump"):
        return obj.model_dump().get(key, default)
    return getattr(obj, key, default)


def build_adapter_bundle(cfg: SimConfig, registry: Any = None,
                         vla_settings: Any = None,
                         dog_cfg: Any = None) -> AdapterBundle:
    """按模式构建执行通道。默认 legacy（零改动，保证既有行为）。

    dog_cfg：机器狗真机通道配置（`settings.dog`，可为 model/dict/None），
    仅 `mode=zsibot` 时使用。
    """
    mode = _parse_mode(cfg.mode)

    if mode == AdapterMode.ZSIBOT:
        # 延迟导入：dog 包反向依赖本包的 base，函数内导入可避开循环依赖
        from ...dog.adapter import build_dog_adapter
        from ...dog.frames import FrameTransform
        frame = FrameTransform(
            yaw_deg=float(_cfg_get(dog_cfg, "yaw_deg", 0.0) or 0.0),
            translate=list(_cfg_get(dog_cfg, "translate", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]),
            scale=float(_cfg_get(dog_cfg, "scale", 1.0) or 1.0),
            flip_x=bool(_cfg_get(dog_cfg, "flip_x", False)),
            flip_y=bool(_cfg_get(dog_cfg, "flip_y", False)),
        )
        adapter = build_dog_adapter(
            uri=str(_cfg_get(dog_cfg, "ws_uri", "") or ""),
            exe_path=str(_cfg_get(dog_cfg, "daemon_exe", "") or ""),
            dog_ip=str(_cfg_get(dog_cfg, "dog_ip", "") or ""),
            send_port=int(_cfg_get(dog_cfg, "send_port", 8081) or 8081),
            recv_port=int(_cfg_get(dog_cfg, "recv_port", 8080) or 8080),
            dry_run=bool(_cfg_get(dog_cfg, "dry_run", True)),
            frame=frame,
            collect_seconds=float(_cfg_get(dog_cfg, "collect_seconds", 3.0) or 3.0),
            nav_timeout_s=float(_cfg_get(dog_cfg, "nav_timeout_s", 120.0) or 120.0),
        )
        return AdapterBundle(adapter, mode=mode)

    if mode == AdapterMode.SIM:
        dyn = Dynamics(uav_speed_mps=cfg.uav_speed_mps, dog_speed_mps=cfg.dog_speed_mps,
                       min_battery_pct=cfg.min_battery_pct)
        world = SimWorld(dynamics=dyn)
        adapter = SimulatedAdapter(dynamics=dyn, time_scale=cfg.time_scale,
                                   failure_rate=cfg.failure_rate,
                                   comm_loss_rate=cfg.comm_loss_rate,
                                   timeout_ms=cfg.timeout_ms, seed=cfg.seed,
                                   world=world, vla_settings=vla_settings)
        return AdapterBundle(adapter, mode=mode)

    if mode == AdapterMode.PLAYBACK:
        path = cfg.playback_file or default_playback_path()
        return AdapterBundle(PlaybackAdapter(path, time_scale=cfg.time_scale), mode=mode)

    if mode == AdapterMode.ROS2:
        return AdapterBundle(Ros2Adapter(), mode=mode)

    # legacy：沿用既有 Mock 工具实现
    overrides: Dict[str, ToolAdapter] = {}
    if registry is not None:
        for name in registry.list():
            tool = registry.get(name)
            if tool is not None:
                overrides[name] = LegacyMockAdapter(tool)
    fallback = LegacyMockAdapter(registry.get("cloud_llm")) if (
        registry is not None and registry.get("cloud_llm") is not None) else None
    return AdapterBundle(fallback or _NullAdapter(), overrides=overrides, mode=AdapterMode.LEGACY)


class _NullAdapter(ToolAdapter):
    """兜底适配器：无任何可用通道时明确失败，而不是静默成功。"""

    name = "null"
    real_hardware = False

    def invoke(self, ctx: AdapterContext) -> AdapterOutcome:
        return AdapterOutcome(status=AdapterStatus.FAILED, success=False,
                              observation="没有可用的执行通道",
                              telemetry={"adapter": self.name}, error="no_adapter")


def _parse_mode(raw: Any) -> AdapterMode:
    s = str(raw or "legacy").strip().lower()
    for m in AdapterMode:
        if m.value == s:
            return m
    return AdapterMode.LEGACY


def make_context(tool_name: str, platform: Any, tier: Any, ws: Any = None,
                 args: Optional[dict] = None, time_scale: float = 1.0,
                 timeout_ms: float = 30_000.0,                  token: Optional[CancellationToken] = None,
                 progress_cb: Any = None,
                 declared_effects: Optional[List[Effect]] = None) -> AdapterContext:
    """便捷构造 AdapterContext（Executor 用）。"""
    from ...core.types import Platform, RouteTier
    plat = platform if isinstance(platform, Platform) else Platform(str(platform or "shared"))
    t = tier if isinstance(tier, RouteTier) else RouteTier(str(tier or "edge"))
    return AdapterContext(
        tool_name=tool_name, platform=plat, tier=t, args=args or {},
        clock=SimClock(time_scale=time_scale), token=token, timeout_ms=timeout_ms,
        progress_cb=progress_cb,
        facts=dict(ws.facts) if ws is not None and hasattr(ws, "facts") else {},
        ws=ws, declared_effects=list(declared_effects or []),
    )
