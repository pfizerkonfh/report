"""可插拔的 Mock 工具集（生产环境替换为真实 ROS / 大模型 / 算子调用）。

每个工具严格声明 Precondition/Effect/Rollback 与 RouteHint，供 Reasoner 前置校验
与 Router 端-边-云决策。时延为仿真值（不真实 sleep），用于评测度量。
"""
from __future__ import annotations

import time
import uuid
from typing import Optional

from ..core.types import (
    ComputeIntensity, Condition, Effect, Platform, PrivacyLevel, RouteTier,
)
from ..core.world_state import WorldState
from ..config.settings import VLASettings
from .base import RouteHint, ToolResult, ToolSpec
from .latency import simulate_vla_latency


class WeatherTool(ToolSpec):
    def __init__(self) -> None:
        super().__init__(
            name="weather",
            description="查询变电站当日天气与无人机航线可行性",
            preconditions=[],
            effects=[Effect(key="weather_ok", op="set", value=True),
                     Effect(key="route_permitted", op="set", value=True)],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.LOW,
                            latency_budget_ms=30, privacy_level=PrivacyLevel.LOW, edge_capable=True),
            base_latency_ms=20.0,
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        ws.apply(Effect(key="weather_ok", op="set", value=True))
        ws.apply(Effect(key="route_permitted", op="set", value=True))
        return ToolResult(success=True, observation="天气晴、风速<5m/s，航线许可", applied_effects=self.effects, latency_ms=self.base_latency_ms)


class CameraTool(ToolSpec):
    def __init__(self) -> None:
        super().__init__(
            name="camera",
            description="无人机航拍采集巡检素材",
            preconditions=[Condition(key="battery", op=">=", value=20, description="电量充足"),
                           Condition(key="safety_cleared", op="==", value=True, description="安全 clearance 已通过")],
            effects=[Effect(key="footage", op="set", value=True),
                     Effect(key="footage_path", op="set", value=""),
                     Effect(key="battery", op="dec", value=10)],
            rollback=[Effect(key="footage", op="delete"), Effect(key="footage_path", op="delete")],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.LOW,
                            latency_budget_ms=60, privacy_level=PrivacyLevel.LOW, edge_capable=True),
            base_latency_ms=30.0,
            platform_hint=Platform.UAV,   # 默认由无人机航拍承载（地面采集由 Planner 改判为机器狗）
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        path = f"/data/footage/{uuid.uuid4().hex[:8]}.mp4"
        ws.apply(Effect(key="footage", op="set", value=True))
        ws.apply(Effect(key="footage_path", op="set", value=path))
        ws.apply(Effect(key="battery", op="dec", value=10))
        return ToolResult(success=True, observation=f"完成航拍，素材存于 {path}", applied_effects=self.effects, latency_ms=self.base_latency_ms)


class EditTool(ToolSpec):
    def __init__(self) -> None:
        super().__init__(
            name="edit",
            description="将原始素材剪辑为巡检短片（需 heavy 模型，下沉云端）",
            preconditions=[Condition(key="footage", op="==", value=True, description="已有原始素材")],
            effects=[Effect(key="edited", op="set", value=True),
                     Effect(key="edited_path", op="set", value="/data/out/clip.mp4")],
            route=RouteHint(tier_preference=RouteTier.CLOUD, compute_intensity=ComputeIntensity.HIGH,
                            latency_budget_ms=300, privacy_level=PrivacyLevel.LOW, edge_capable=False,
                            cloud_service="video-edit-svc"),
            base_latency_ms=150.0,
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        ws.apply(Effect(key="edited", op="set", value=True))
        ws.apply(Effect(key="edited_path", op="set", value="/data/out/clip.mp4"))
        return ToolResult(success=True, observation="剪辑完成，生成短片 clip.mp4", applied_effects=self.effects, latency_ms=self.base_latency_ms)


class AnonymizeTool(ToolSpec):
    def __init__(self) -> None:
        super().__init__(
            name="anonymize",
            description="对短片中人脸做匿名化（含人脸数据，必须边侧本地处理）",
            preconditions=[Condition(key="footage", op="==", value=True, description="已有原始素材")],
            effects=[Effect(key="anonymized", op="set", value=True),
                     Effect(key="final_path", op="set", value="/data/out/final_anon.mp4")],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.MEDIUM,
                            latency_budget_ms=60, privacy_level=PrivacyLevel.HIGH, edge_capable=True),
            base_latency_ms=50.0,
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        ws.apply(Effect(key="anonymized", op="set", value=True))
        ws.apply(Effect(key="final_path", op="set", value="/data/out/final_anon.mp4"))
        return ToolResult(success=True, observation="人脸已匿名化，输出 final_anon.mp4", applied_effects=self.effects, latency_ms=self.base_latency_ms)


class VLATool(ToolSpec):
    vla_settings: Optional[VLASettings] = None

    def __init__(self, vla_settings: Optional[VLASettings] = None) -> None:
        super().__init__(
            name="vla",
            description="端侧 VLA 模型推理，输出机器人控制指令",
            preconditions=[Condition(key="edge_capable", op="==", value=True, description="边侧算力可用")],
            effects=[Effect(key="control_cmd", op="set", value="move(0.2,0,0)")],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.HIGH,
                            latency_budget_ms=50, privacy_level=PrivacyLevel.MEDIUM, edge_capable=True),
            base_latency_ms=65.0,
        )
        self.vla_settings = vla_settings or VLASettings()

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        bd = simulate_vla_latency(self.vla_settings)
        cmd = "move(0.2,0,0)" if bd.final < 50 else "hold()"
        ws.apply(Effect(key="control_cmd", op="set", value=cmd))
        return ToolResult(success=True, observation=f"VLA 推理输出控制指令 {cmd}（时延 {bd.final:.1f}ms）",
                          applied_effects=self.effects, latency_ms=bd.final)


class ChargeTool(ToolSpec):
    def __init__(self) -> None:
        super().__init__(
            name="charge",
            description="为无人机/机器狗充电恢复电量",
            preconditions=[],
            effects=[Effect(key="battery", op="set", value=100)],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.LOW,
                            latency_budget_ms=500, privacy_level=PrivacyLevel.LOW, edge_capable=True),
            base_latency_ms=500.0,
            fixable=True,
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        ws.apply(Effect(key="battery", op="set", value=100))
        return ToolResult(success=True, observation="电量已恢复至 100%", applied_effects=self.effects, latency_ms=self.base_latency_ms)


class CloudLLMTool(ToolSpec):
    def __init__(self) -> None:
        super().__init__(
            name="cloud_llm",
            description="云端大模型通用分析/生成",
            preconditions=[],
            effects=[Effect(key="analysis", op="set", value="done")],
            route=RouteHint(tier_preference=RouteTier.CLOUD, compute_intensity=ComputeIntensity.HIGH,
                            latency_budget_ms=300, privacy_level=PrivacyLevel.LOW, edge_capable=False,
                            cloud_service="llm-gateway"),
            base_latency_ms=200.0,
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        ws.apply(Effect(key="analysis", op="set", value="done"))
        return ToolResult(success=True, observation="云端分析完成", applied_effects=self.effects, latency_ms=self.base_latency_ms)
