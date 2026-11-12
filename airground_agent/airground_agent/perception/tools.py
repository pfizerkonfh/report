"""真实感知工具集 —— 基于 3DGS 重建产物，不再是硬编码 Mock。

与既有 Mock 工具的本质区别
--------------------------
Mock 工具（如 camera 返回 uuid 假路径）不依赖任何真实数据，返回的观测是编造的。
本模块每个工具的 observation 都是对**真实占据栅格 / 真实点云**的计算结果：

- `scene_inspect`：真实场景规模、已知率、重建 PSNR / 高斯数量
- `map_query`：真实栅格查询，某点到底是自由 / 障碍 / 未知
- `route_plan`：在真实栅格上跑真实 A*，返回真实里程（米）
- `height_probe`：真实点云高度剖面（无人机看净空，机器狗看地面）

这些工具让 Planner/Reasoner 的前置校验第一次建立在真实场景数据之上。
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from pydantic import PrivateAttr

from ..core.types import (
    ComputeIntensity, Condition, Effect, Platform, PrivacyLevel, RouteTier,
)
from ..core.world_state import WorldState
from .scene import FREE, OCCUPIED, UNKNOWN, SceneModel
from ..tools.base import RouteHint, ToolResult, ToolSpec

_CELL_NAMES = {FREE: "自由", OCCUPIED: "障碍", UNKNOWN: "未知"}


class SceneBackedTool(ToolSpec):
    """持有真实 SceneModel 的工具基类（scene 为私有属性，不参与序列化）。

    本类所有工具都是**纯计算/查询**：结果必须由真实占据栅格 / 真实点云算出，
    因此统一标记 `compute_only=True`，执行通道只叠加时序，不得覆盖计算结果。
    """

    _scene: Optional[SceneModel] = PrivateAttr(default=None)

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("compute_only", True)
        super().__init__(**kwargs)

    def bind(self, scene: Optional[SceneModel]) -> "SceneBackedTool":
        self._scene = scene
        return self

    @property
    def scene(self) -> Optional[SceneModel]:
        return self._scene

    def _no_scene(self) -> ToolResult:
        return ToolResult(
            success=False,
            observation="感知层未加载真实场景数据（未找到 3DGS 重建产物）",
            error="scene_missing",
        )


class SceneInspectTool(SceneBackedTool):
    """汇报真实场景规模与重建质量。"""

    def __init__(self) -> None:
        super().__init__(
            name="scene_inspect",
            description="检查已加载的 3DGS 重建场景：栅格规模、可通行/障碍/未知占比、重建质量",
            preconditions=[],
            effects=[Effect(key="scene_loaded", op="set", value=True),
                     Effect(key="scene_known_ratio", op="set", value=0.0)],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.LOW,
                            latency_budget_ms=30, privacy_level=PrivacyLevel.LOW, edge_capable=True),
            base_latency_ms=5.0,
            platform_hint=Platform.SHARED,
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        sc = self.scene
        if sc is None:
            return self._no_scene()
        st, q = sc.stats(), sc.quality()
        ws.apply(Effect(key="scene_loaded", op="set", value=True))
        ws.apply(Effect(key="scene_known_ratio", op="set", value=st["known_ratio"]))
        ws.apply(Effect(key="scene_free_cells", op="set", value=st["cells_free"]))
        return ToolResult(
            success=True,
            observation=sc.describe(),
            applied_effects=self.effects,
            latency_ms=self.base_latency_ms,
            telemetry={"stats": st, "quality": q},
        )


class MapQueryTool(SceneBackedTool):
    """查询某个世界坐标在真实占据栅格中的可通行性。"""

    def __init__(self) -> None:
        super().__init__(
            name="map_query",
            description="查询地图上指定位置的可通行性（自由/障碍/未知）及最近可通行点",
            preconditions=[Condition(key="scene_loaded", op="==", value=True,
                                     description="已加载真实场景地图")],
            effects=[Effect(key="target_traversable", op="set", value=False),
                     Effect(key="target_cell", op="set", value="")],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.LOW,
                            latency_budget_ms=20, privacy_level=PrivacyLevel.LOW, edge_capable=True),
            base_latency_ms=3.0,
            platform_hint=Platform.DOG,   # 地面平台最依赖可通行性
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        sc = self.scene
        if sc is None:
            return self._no_scene()
        allow_unknown = bool(kwargs.get("allow_unknown", True))
        cell = self._resolve_cell(sc, kwargs)
        ix, iz = cell
        v = sc.cell_value(ix, iz)
        name = _CELL_NAMES.get(v, "未知")
        traversable = sc.traversable(ix, iz, allow_unknown)
        wx, wz = sc.cell_to_world(ix, iz)

        obs = (f"位置 ({wx:.2f}, {wz:.2f}) -> 格 ({ix},{iz}) 为「{name}」，"
               f"{'可通行' if traversable else '不可通行'}")
        detour = None
        if not traversable:
            nf = sc.nearest_free(ix, iz, allow_unknown)
            if nf:
                nx_, nz_ = sc.cell_to_world(nf[0], nf[1])
                detour = {"cell": list(nf), "world": [round(nx_, 2), round(nz_, 2)],
                          "distance_m": round(sc.res * max(abs(nf[0] - ix), abs(nf[1] - iz)), 2)}
                obs += f"；最近可通行点格 {list(nf)}，距离约 {detour['distance_m']}m"

        ws.apply(Effect(key="target_traversable", op="set", value=bool(traversable)))
        ws.apply(Effect(key="target_cell", op="set", value=f"{ix},{iz}"))
        return ToolResult(
            success=True,
            observation=obs,
            applied_effects=self.effects,
            latency_ms=self.base_latency_ms,
            telemetry={"cell": [ix, iz], "value": v, "status": name,
                       "traversable": traversable, "nearest_free": detour},
        )

    @staticmethod
    def _resolve_cell(sc: SceneModel, kwargs: dict) -> tuple:
        """容忍多种入参写法：cell / x+z / point / 缺省用场景起点。"""
        if kwargs.get("cell") is not None:
            c = kwargs["cell"]
            return int(c[0]), int(c[1])
        if kwargs.get("x") is not None and kwargs.get("z") is not None:
            return sc.world_to_cell(float(kwargs["x"]), float(kwargs["z"]))
        pt = kwargs.get("point")
        if isinstance(pt, (list, tuple)) and len(pt) >= 3:
            return sc.world_to_cell(float(pt[0]), float(pt[2]))
        return sc.default_start()


class RoutePlanTool(SceneBackedTool):
    """在真实占据栅格上做 A* 路径规划，返回真实里程。"""

    def __init__(self) -> None:
        super().__init__(
            name="route_plan",
            description="在真实占据地图上规划从起点到目标点的通行路径，返回真实里程与未知区占比",
            preconditions=[Condition(key="scene_loaded", op="==", value=True,
                                     description="已加载真实场景地图")],
            effects=[Effect(key="route_ready", op="set", value=False),
                     Effect(key="route_len_m", op="set", value=0.0),
                     Effect(key="route_steps", op="set", value=0)],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.MEDIUM,
                            latency_budget_ms=60, privacy_level=PrivacyLevel.LOW, edge_capable=True),
            base_latency_ms=15.0,
            platform_hint=Platform.DOG,   # 地面导航主要由机器狗执行
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        sc = self.scene
        if sc is None:
            return self._no_scene()
        allow_unknown = bool(kwargs.get("allow_unknown", True))
        start = self._resolve_point(sc, kwargs, "start", sc.default_start())
        goal = self._resolve_point(sc, kwargs, "goal", sc.default_goal())

        res = sc.plan_world(start, goal, allow_unknown=allow_unknown)
        if not res.get("success"):
            ws.apply(Effect(key="route_ready", op="set", value=False))
            return ToolResult(success=False,
                              observation=f"路径规划失败：{res.get('reason')}",
                              error="unreachable", latency_ms=self.base_latency_ms,
                              telemetry=res)

        ws.apply(Effect(key="route_ready", op="set", value=True))
        ws.apply(Effect(key="route_len_m", op="set", value=res["length_m"]))
        ws.apply(Effect(key="route_steps", op="set", value=res["steps"]))
        snap = "（起终点已纠偏到最近可通行格）" if res.get("snapped") else ""
        obs = (f"A* 规划成功：{res['steps']} 格，真实里程 {res['length_m']}m，"
               f"途经未知区 {res['unknown_cells']} 格（{res['unknown_ratio']:.0%}）{snap}")
        return ToolResult(success=True, observation=obs, applied_effects=self.effects,
                          latency_ms=self.base_latency_ms, telemetry=res)

    @staticmethod
    def _resolve_point(sc: SceneModel, kwargs: dict, prefix: str, default_cell) -> Sequence[float]:
        """支持 start=[x,y,z] / start_x+start_z / start_cell / 缺省。"""
        v = kwargs.get(prefix)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]
        sx = kwargs.get(f"{prefix}_x")
        sz = kwargs.get(f"{prefix}_z")
        if sx is not None and sz is not None:
            return [float(sx), sc.floor_y, float(sz)]
        c = kwargs.get(f"{prefix}_cell")
        if c is not None:
            wx, wz = sc.cell_to_world(int(c[0]), int(c[1]))
            return [wx, sc.floor_y, wz]
        wx, wz = sc.cell_to_world(default_cell[0], default_cell[1])
        return [wx, sc.floor_y, wz]


class HeightProbeTool(SceneBackedTool):
    """真实点云高度剖面：无人机看净空，机器狗看地面起伏。"""

    def __init__(self) -> None:
        super().__init__(
            name="height_probe",
            description="探测指定位置真实点云的高度剖面（净空高度、竖直跨度），判断能否通行或飞越",
            preconditions=[Condition(key="scene_loaded", op="==", value=True,
                                     description="已加载真实场景地图")],
            effects=[Effect(key="clearance_m", op="set", value=0.0)],
            route=RouteHint(tier_preference=RouteTier.EDGE, compute_intensity=ComputeIntensity.LOW,
                            latency_budget_ms=30, privacy_level=PrivacyLevel.LOW, edge_capable=True),
            base_latency_ms=8.0,
            platform_hint=Platform.UAV,   # 净空判断对无人机更关键
        )

    def run(self, ws: WorldState, **kwargs) -> ToolResult:
        sc = self.scene
        if sc is None:
            return self._no_scene()
        radius = float(kwargs.get("radius", 0.5))
        if kwargs.get("x") is not None and kwargs.get("z") is not None:
            wx, wz = float(kwargs["x"]), float(kwargs["z"])
        elif kwargs.get("cell") is not None:
            wx, wz = sc.cell_to_world(int(kwargs["cell"][0]), int(kwargs["cell"][1]))
        else:
            wx, wz = sc.cell_to_world(*sc.default_start())

        info = sc.probe(wx, wz, radius)
        if not info.get("available"):
            return ToolResult(success=False, observation=f"高度探测不可用：{info.get('reason')}",
                              error="no_pointcloud", latency_ms=self.base_latency_ms)
        if info.get("points", 0) == 0:
            return ToolResult(success=True,
                              observation=f"位置 ({wx:.2f},{wz:.2f}) 半径 {radius}m 内无重建点（空洞/未观测区）",
                              applied_effects=[], latency_ms=self.base_latency_ms, telemetry=info)

        clearance = info["clearance_above_floor_m"]
        ws.apply(Effect(key="clearance_m", op="set", value=clearance))
        verdict = "净空充足，可飞越" if clearance >= 2.0 else "净空受限，建议地面绕行"
        obs = (f"位置 ({wx:.2f},{wz:.2f}) 半径 {radius}m：{info['points']} 个重建点，"
               f"高度 {info['y_min']}~{info['y_max']}m（跨度 {info['vertical_extent_m']}m），"
               f"离地净空 {clearance}m —— {verdict}")
        return ToolResult(success=True, observation=obs, applied_effects=self.effects,
                          latency_ms=self.base_latency_ms, telemetry=info)


def build_perception_tools(scene: Optional[SceneModel]) -> list:
    """构造并绑定真实场景的感知工具集合。"""
    return [
        SceneInspectTool().bind(scene),
        MapQueryTool().bind(scene),
        RoutePlanTool().bind(scene),
        HeightProbeTool().bind(scene),
    ]
