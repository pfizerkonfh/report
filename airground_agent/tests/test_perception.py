"""感知层测试：验证真实 3DGS 重建数据被正确加载并参与真实计算。

分两类：
1. 合成场景 —— 确定性验证坐标转换、占据查询、连通性、A* 绕障等核心逻辑；
2. 真实场景 —— 加载 exp/ 下的 3DGS 产物（缺失则 skip），验证端到端真实数据可用。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airground_agent.core.world_state import WorldState
from airground_agent.perception import (
    HeightProbeTool, MapQueryTool, RoutePlanTool, SceneInspectTool, SceneModel, load_scene,
)
from airground_agent.perception.scene import FREE, OCCUPIED, UNKNOWN

REAL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "exp", "viewer", "showcase_assets",
)


def _synthetic(wall_rows=(0, 1, 2)) -> SceneModel:
    """5x5 合成场景：ix=2 列为障碍墙，仅在 iz>=3 处开口。"""
    occ = [[FREE] * 5 for _ in range(5)]
    for iz in wall_rows:
        occ[2][iz] = OCCUPIED
    return SceneModel(
        occ=occ, nx=5, nz=5, res=0.5, xmin=0.0, zmin=0.0, floor_y=0.0,
        traj=[[0, 0], [0, 1]], start=[0, 0], goal=[4, 0],
        points=[[0.0, 0.0, 0.0], [0.2, 1.5, 0.1], [0.4, 3.0, 0.0]],
    )


# ---------------------------------------------------------------------------
# 坐标转换
# ---------------------------------------------------------------------------
def test_world_cell_roundtrip():
    sc = _synthetic()
    for ix in range(5):
        for iz in range(5):
            wx, wz = sc.cell_to_world(ix, iz)
            assert sc.world_to_cell(wx, wz) == (ix, iz)


def test_world_to_cell_respects_origin():
    sc = SceneModel(occ=[[FREE]], nx=1, nz=1, res=0.2, xmin=-4.2307, zmin=-5.6167, floor_y=-5.58)
    assert sc.world_to_cell(-4.2307, -5.6167) == (0, 0)
    assert sc.world_to_cell(-4.2307 + 0.19, -5.6167 + 0.19) == (0, 0)
    assert sc.world_to_cell(-4.2307 + 0.25, -5.6167) == (1, 0)


def test_cell_to_world3_uses_floor_height():
    sc = SceneModel(occ=[[FREE]], nx=1, nz=1, res=0.2, xmin=0.0, zmin=0.0, floor_y=-5.58)
    _, y, _ = sc.cell_to_world3(0, 0)
    assert abs(y - (-5.58 + 0.3)) < 1e-6


def test_out_of_bounds_treated_as_unknown():
    sc = _synthetic()
    assert sc.cell_value(99, 99) == UNKNOWN
    assert not sc.is_free(-1, 0)


# ---------------------------------------------------------------------------
# 占据查询
# ---------------------------------------------------------------------------
def test_occupancy_query():
    sc = _synthetic()
    assert sc.is_occupied(2, 0)
    assert sc.is_free(0, 0)
    assert not sc.traversable(2, 1)


def test_nearest_free_snaps_off_obstacle():
    sc = _synthetic()
    nf = sc.nearest_free(2, 0)
    assert nf is not None
    assert sc.is_free(*nf)


# ---------------------------------------------------------------------------
# 连通性
# ---------------------------------------------------------------------------
def test_reachable_and_connected():
    sc = _synthetic()  # 墙在 ix=2 的 iz=0,1,2，iz>=3 可绕
    assert sc.connected((0, 0), (4, 0))
    reach = sc.reachable_from(0, 0)
    assert (4, 0) in reach
    assert (2, 0) not in reach  # 障碍格不可达


def test_unreachable_when_wall_is_full():
    sc = _synthetic(wall_rows=(0, 1, 2, 3, 4))  # 整列封锁
    assert not sc.connected((0, 0), (4, 0))


# ---------------------------------------------------------------------------
# A* 规划
# ---------------------------------------------------------------------------
def test_astar_plan_bypasses_wall():
    sc = _synthetic()
    path = sc.plan((0, 0), (4, 0))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (4, 0)
    # 必须绕行到 iz>=3 的开口
    assert any(iz >= 3 for _, iz in path)
    # 路径上不得出现障碍格
    assert all(not sc.is_occupied(ix, iz) for ix, iz in path)


def test_astar_returns_none_when_blocked():
    sc = _synthetic(wall_rows=(0, 1, 2, 3, 4))
    assert sc.plan((0, 0), (4, 0)) is None


def test_astar_same_cell_returns_single():
    sc = _synthetic()
    assert sc.plan((1, 1), (1, 1)) == [(1, 1)]


def test_plan_world_reports_real_distance():
    sc = _synthetic()
    res = sc.plan_world([0.25, 0.0, 0.25], [2.25, 0.0, 0.25])
    assert res["success"] is True
    # 起点(0,0) 终点(4,0)，绕行后路程必然大于直线距离 2.0m
    assert res["length_m"] > 2.0
    assert res["steps"] == len(res["cells"])
    assert res["length_m"] == pytest.approx(
        sum(
            ((res["world_path"][i + 1][0] - res["world_path"][i][0]) ** 2
             + (res["world_path"][i + 1][2] - res["world_path"][i][2]) ** 2) ** 0.5
            for i in range(len(res["world_path"]) - 1)
        ), abs=0.02)


# ---------------------------------------------------------------------------
# 点云探测
# ---------------------------------------------------------------------------
def test_probe_returns_height_profile():
    sc = _synthetic()
    info = sc.probe(0.0, 0.0, radius=1.0)
    assert info["available"] is True
    assert info["points"] == 3
    assert info["y_max"] == 3.0
    assert info["vertical_extent_m"] == 3.0


def test_probe_empty_area():
    sc = _synthetic()
    info = sc.probe(50.0, 50.0, radius=0.5)
    assert info["points"] == 0


# ---------------------------------------------------------------------------
# 统计与 facts 注入
# ---------------------------------------------------------------------------
def test_stats_and_facts():
    sc = _synthetic()
    st = sc.stats()
    assert st["cells_total"] == 25
    assert st["cells_occupied"] == 3
    assert st["extent_m"] == [2.5, 2.5]
    facts = sc.to_facts()
    assert facts["scene_loaded"] is True
    assert facts["scene_occupied_cells"] == 3


# ---------------------------------------------------------------------------
# 感知工具（合成场景）
# ---------------------------------------------------------------------------
def _ready_ws() -> WorldState:
    return WorldState(facts={"scene_loaded": True})


def test_scene_inspect_tool_reports_real_stats():
    tool = SceneInspectTool().bind(_synthetic())
    res = tool.run(_ready_ws())
    assert res.success
    assert "5x5" in res.observation
    assert res.telemetry["stats"]["cells_occupied"] == 3


def test_map_query_tool_on_obstacle_and_free():
    sc = _synthetic()
    tool = MapQueryTool().bind(sc)
    ws = _ready_ws()
    free = tool.run(ws, cell=[0, 0])
    assert free.success and free.telemetry["traversable"] is True
    blocked = tool.run(ws, cell=[2, 0])
    assert blocked.telemetry["traversable"] is False
    assert blocked.telemetry["nearest_free"] is not None


def test_route_plan_tool_sets_real_distance():
    tool = RoutePlanTool().bind(_synthetic())
    ws = _ready_ws()
    res = tool.run(ws, start=[0.25, 0.0, 0.25], goal=[2.25, 0.0, 0.25])
    assert res.success
    assert ws.get("route_ready") is True
    assert ws.get("route_len_m") > 2.0
    assert "真实里程" in res.observation


def test_height_probe_tool():
    tool = HeightProbeTool().bind(_synthetic())
    res = tool.run(_ready_ws(), x=0.0, z=0.0, radius=1.0)
    assert res.success
    assert res.telemetry["points"] == 3


def test_tools_fail_gracefully_without_scene():
    for tool in (SceneInspectTool(), MapQueryTool(), RoutePlanTool(), HeightProbeTool()):
        res = tool.bind(None).run(_ready_ws())
        assert not res.success
        assert res.error == "scene_missing"


# ---------------------------------------------------------------------------
# 真实 3DGS 数据（缺失则跳过）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_DIR, "nav_data.json")),
                    reason="未找到 3DGS 重建产物")
def test_real_scene_loads():
    sc = load_scene(data_dir=REAL_DIR)
    assert sc is not None
    st = sc.stats()
    assert st["grid"] == [53, 125]
    assert st["resolution_m"] == 0.2
    assert st["cells_free"] > 0
    assert st["known_ratio"] > 0


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_DIR, "nav_data.json")),
                    reason="未找到 3DGS 重建产物")
def test_real_scene_quality_metrics():
    sc = load_scene(data_dir=REAL_DIR)
    q = sc.quality()
    assert q["psnr_test_mean"] == 28.06
    assert q["gaussian_count"] == 387299
    assert q["camera_poses"] == 98


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_DIR, "nav_data.json")),
                    reason="未找到 3DGS 重建产物")
def test_real_scene_planning_is_consistent_with_reference():
    """在真实栅格上重跑 A*，结果应与 build_nav.py 产出的参考路径一致。"""
    sc = load_scene(data_dir=REAL_DIR)
    start, goal = sc.default_start(), sc.default_goal()
    assert sc.connected(start, goal)
    path = sc.plan(start, goal)
    assert path is not None
    # 与导出时的参考路径步数一致（同一套 A* 规则）
    assert len(path) == len(sc.reference_path) or abs(
        len(path) - len(sc.reference_path)) <= 2


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_DIR, "nav_data.json")),
                    reason="未找到 3DGS 重建产物")
def test_real_scene_route_tool_end_to_end():
    sc = load_scene(data_dir=REAL_DIR)
    ws = WorldState(facts=sc.to_facts())
    assert ws.get("scene_loaded") is True
    tool = RoutePlanTool().bind(sc)
    res = tool.run(ws)
    assert res.success
    assert ws.get("route_len_m") > 0
    assert res.telemetry["length_m"] > 0


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_DIR, "nav_data.json")),
                    reason="未找到 3DGS 重建产物")
def test_real_scene_probe_on_real_pointcloud():
    sc = load_scene(data_dir=REAL_DIR)
    wx, wz = sc.cell_to_world(*sc.default_start())
    info = sc.probe(wx, wz, radius=0.8)
    assert info["available"] is True


# ---------------------------------------------------------------------------
# 执行通道耦合：纯计算工具在仿真通道下不得丢失真实计算结果
# ---------------------------------------------------------------------------
def _executor_with_sim(tool, scene):
    from airground_agent.routing.network import NetworkGovernor
    from airground_agent.routing.router import Router
    from airground_agent.tools.adapters.factory import SimConfig, build_adapter_bundle
    from airground_agent.tools.executor import ToolExecutor
    from airground_agent.tools.registry import ToolRegistry

    reg = ToolRegistry()
    reg.register(tool)
    sim = SimConfig(mode="sim", time_scale=200.0)   # 大幅加速，避免测试变慢
    from airground_agent.config.settings import RouterSettings

    net = NetworkGovernor(RouterSettings())
    return ToolExecutor(reg, Router(net), net,
                        adapters=build_adapter_bundle(sim, reg), sim=sim)


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_DIR, "nav_data.json")),
                    reason="未找到 3DGS 重建产物")
def test_compute_only_not_overwritten_by_sim_channel():
    """回归：仿真通道曾把 route_plan 的真实结果覆盖成 route_len_m=0.0。"""
    sc = load_scene(data_dir=REAL_DIR)
    ws = WorldState(facts=sc.to_facts())
    tool = RoutePlanTool().bind(sc)
    ex = _executor_with_sim(tool, sc)

    out = ex.execute("route_plan", ws, platform=tool.platform_hint)
    assert out.ok is True
    # 真实计算必须保留，不能被模板 observation / 静态声明 effects 覆盖
    assert "A* 规划成功" in out.result.observation
    assert ws.get("route_ready") is True
    assert ws.get("route_len_m") > 0
    assert ws.get("route_steps") > 0
    assert out.result.telemetry["length_m"] > 0
    # 通道时序仍然叠加
    assert out.result.sim_duration_ms > 0


@pytest.mark.skipif(not os.path.exists(os.path.join(REAL_DIR, "nav_data.json")),
                    reason="未找到 3DGS 重建产物")
def test_compute_only_map_query_returns_real_cell():
    sc = load_scene(data_dir=REAL_DIR)
    ws = WorldState(facts=sc.to_facts())
    tool = MapQueryTool().bind(sc)
    ex = _executor_with_sim(tool, sc)

    out = ex.execute("map_query", ws, args={"cell": list(sc.default_start())},
                     platform=tool.platform_hint)
    assert out.ok is True
    assert "可通行" in out.result.observation
    assert ws.get("target_cell") == f"{sc.default_start()[0]},{sc.default_start()[1]}"


def test_compute_only_flag_default_and_inheritance():
    """compute_only 默认关闭；SceneBackedTool 子类自动开启。"""
    from airground_agent.tools.mock_tools import CameraTool

    assert CameraTool().compute_only is False
    assert RoutePlanTool().compute_only is True
    assert MapQueryTool().compute_only is True
    assert SceneInspectTool().compute_only is True
    assert HeightProbeTool().compute_only is True
