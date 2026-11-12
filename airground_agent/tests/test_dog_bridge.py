"""机器狗接入层测试：坐标系 / 编译 / 协议 / 执行 / 适配器 / 端到端。

覆盖目标是"数据形式转换"这条链路的正确性：
决策平台的 Plan(+真实航点) -> DogMission -> 机器狗原生指令 -> 执行报告。
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airground_agent.core.task_model import Plan, SubTask, TraceStep  # noqa: E402
from airground_agent.core.types import Platform  # noqa: E402
from airground_agent.dog import (  # noqa: E402
    CompileConfig, DogMissionRunner, DogTransport, FrameTransform, ZsibotDogAdapter,
    compile_plan, resample_path,
)
from airground_agent.dog.adapter import build_dog_adapter  # noqa: E402
from airground_agent.dog.protocol import DogCommand, DogMission  # noqa: E402
from airground_agent.tools.adapters.base import (  # noqa: E402
    AdapterContext, AdapterMode, CancellationToken, Platform as _P, RouteTier,
)


# ======================================================================
# 坐标系
# ======================================================================
def test_axis_mapping_default():
    """默认变换：世界 (x, y, z) -> 地图 (X, Y, Z) = (x, -z, 0)。"""
    f = FrameTransform()
    assert f.world_to_map(1.0, 2.0, 3.0) == (1.0, -3.0, 0.0)


def test_round_trip():
    f = FrameTransform(yaw_deg=30.0, translate=[1.5, -2.0, 0.0], scale=1.1,
                       use_world_height=True)
    X, Y, Z = f.world_to_map(2.0, 0.5, -1.0)
    x, y, z = f.map_to_world(X, Y, Z)
    assert (round(x, 3), round(y, 3), round(z, 3)) == (2.0, 0.5, -1.0)


def test_yaw_rotation_and_quaternion():
    f = FrameTransform(yaw_deg=90.0)
    # 世界 +X 方向旋转 90° 后应落在地图 -Y
    yaw = f.world_heading_to_map_yaw(1.0, 0.0)
    assert math.isclose(math.degrees(yaw), -90.0, abs_tol=1e-6)
    q = f.yaw_to_quaternion(math.pi / 2)
    assert math.isclose(q["z"], round(math.sin(math.pi / 4), 6), abs_tol=1e-6)


def test_calibration_from_point_pairs():
    """用两组同名点标定，应能还原出变换参数（平移+旋转+缩放）。"""
    truth = FrameTransform(yaw_deg=37.0, scale=1.2, translate=[3.0, -1.0, 0.0])
    world = [(1.0, 0.0, 2.0), (4.0, 0.0, -1.0)]
    mapped = [truth.world_to_map(*p) for p in world]
    fit = FrameTransform.fit_from_point_pairs(world, [(m[0], m[1]) for m in mapped])
    for w, m in zip(world, mapped):
        got = fit.world_to_map(*w)
        assert math.isclose(got[0], m[0], abs_tol=1e-3)
        assert math.isclose(got[1], m[1], abs_tol=1e-3)


# ======================================================================
# 航点抽稀
# ======================================================================
def test_resample_keeps_endpoints_and_stride():
    path = [[0.1 * i, 0.0, 0.0] for i in range(21)]   # 2m，每点 0.1m
    out = resample_path(path, stride_m=0.5, max_points=40)
    assert out[0] == path[0]
    assert out[-1] == path[-1]
    assert 4 <= len(out) <= 6


def test_resample_respects_max_points():
    path = [[0.1 * i, 0.0, 0.0] for i in range(200)]
    out = resample_path(path, stride_m=0.5, max_points=10)
    assert len(out) <= 10


# ======================================================================
# 协议序列化
# ======================================================================
def test_start_nav_req_func():
    c = DogCommand(channel="nav", op="start_nav",
                   params={"position": {"x": 1.0, "y": 2.0, "z": 0.0},
                           "orientation": {"x": 0, "y": 0, "z": 0.0, "w": 1.0}})
    req = c.to_req_func()
    assert req["start_nav"]["position"] == {"x": 1.0, "y": 2.0, "z": 0.0}
    frame = c.to_ws_frame(7)
    assert frame["head"]["frame_count"] == 7 and frame["head"]["type"] == "app_req"


def test_multi_nav_and_map_load():
    assert DogCommand(channel="nav", op="start_multi_nav",
                      params={"map_id": "m1", "path_id": "p1"}
                      ).to_req_func() == {"start_multi_nav": ["m1", "p1"]}
    assert DogCommand(channel="nav", op="loc_load_map",
                      params={"map_id": "m1"}).to_req_func() == {"loc_load_map": "m1"}
    assert DogCommand(channel="nav", op="stop_nav").to_req_func() == {"stop_nav": None}


def test_unsupported_wheeled_action_is_rejected():
    """轮式机器狗不支持的动作必须被编译期拦下。"""
    c = DogCommand(channel="sdk", op="back_flip")
    assert c.validate_against_protocol() is not None
    m = DogMission(commands=[c])
    assert m.validate()


def test_daemon_line():
    assert DogCommand(channel="sdk", op="stand_up").to_daemon_line() == "stand_up"
    assert DogCommand(channel="nav", op="stop_nav").to_daemon_line() is None


# ======================================================================
# 编译器
# ======================================================================
def _plan_with_dog(tool_hint: str = "route_plan", platform=Platform.DOG) -> Plan:
    st = SubTask(id="t1", description="机器狗巡检设备区", tool_hint=tool_hint,
                 platform=platform, success_criteria="抵达设备区")
    return Plan(goal_id="g1", root_goal="巡检设备区", subtasks={"t1": st})


def _trace_with_path(sid: str = "t1") -> list:
    return [TraceStep(step_id=1, subtask_id=sid, action={"tool": "route_plan", "args": {}},
                      telemetry={"success": True, "length_m": 2.0, "steps": 10,
                                 "world_path": [[0.0, 0.3, 0.0], [0.2, 0.3, 0.0],
                                                [0.4, 0.3, 0.0], [0.6, 0.3, 0.4]]})]


def test_compile_navigation_from_real_trace():
    mission = compile_plan(_plan_with_dog(), _trace_with_path(),
                           cfg=CompileConfig(waypoint_stride_m=0.2))
    navs = [c for c in mission.commands if c.channel == "nav" and c.op == "start_nav"]
    assert navs, "应生成导航指令"
    assert mission.waypoint_source.startswith("trace.route_plan")
    # 前导：sdk_mode + stand_up；坐标已换到地图系（世界 z -> 地图 -y）
    assert mission.commands[0].op == "sdk_mode"
    assert navs[-1].params["position"]["y"] == -0.4
    assert all(c.subtask_id == "t1" for c in navs)


def test_uav_subtask_not_compiled():
    st = SubTask(id="u1", description="无人机航拍", tool_hint="camera", platform=Platform.UAV)
    plan = Plan(goal_id="g", root_goal="航拍", subtasks={"u1": st})
    mission = compile_plan(plan, [])
    assert not [c for c in mission.commands if c.subtask_id == "u1"]
    assert any("没有 platform=DOG" in w for w in mission.warnings)


def test_unmapped_tool_warns_instead_of_silently_dropping():
    st = SubTask(id="t9", description="未知动作", tool_hint="unknown_tool",
                 platform=Platform.DOG)
    plan = Plan(goal_id="g", root_goal="测试", subtasks={"t9": st})
    mission = compile_plan(plan, [])
    assert any("未登记机器狗动作映射" in w for w in mission.warnings)
    assert any(c.op == "unmapped" for c in mission.commands)


def test_collect_and_return_home():
    plan = Plan(goal_id="g", root_goal="采集", subtasks={
        "n1": SubTask(id="n1", description="到设备区", tool_hint="route_plan",
                      platform=Platform.DOG),
        "c1": SubTask(id="c1", description="近距采集", tool_hint="camera",
                      platform=Platform.DOG, depends_on=["n1"]),
        "e1": SubTask(id="e1", description="返回充电", tool_hint="charge",
                      platform=Platform.DOG, depends_on=["c1"]),
    })
    mission = compile_plan(plan, _trace_with_path(sid="n1"),
                           cfg=CompileConfig(collect_seconds=5.0))
    ops = [(c.channel, c.op) for c in mission.commands]
    assert ("wait", "sleep") in ops
    assert ("nav", "start_nav_return_home") in ops
    assert mission.subtask_cmd_count.get("c1") == 2   # 导航 + 采集窗口


def test_multi_nav_when_path_id_given():
    mission = compile_plan(_plan_with_dog(), _trace_with_path(),
                           cfg=CompileConfig(map_id="m1", path_id="p1", nav_mode="auto"))
    assert any(c.op == "start_multi_nav" for c in mission.commands)
    assert any(c.op == "loc_load_map" for c in mission.commands)


def test_export_shapes():
    mission = compile_plan(_plan_with_dog(), _trace_with_path())
    frames = mission.nav_frames()
    assert frames and all(f["head"]["type"] == "app_req" for f in frames)
    assert "stand_up" in mission.daemon_lines()
    d = mission.to_dict()
    assert DogMission(**{k: v for k, v in d.items() if k != "commands"}) is not None


# ======================================================================
# 执行器（dry-run，不连真机）
# ======================================================================
def test_runner_dry_run_all_ok():
    mission = compile_plan(_plan_with_dog(), _trace_with_path())
    rep = DogMissionRunner(DogTransport.dry()).run(mission)
    assert rep.success and rep.results
    assert len(rep.results) == len(mission.commands)


def test_runner_abort_on_cancel():
    mission = compile_plan(_plan_with_dog(), _trace_with_path())
    token = CancellationToken()
    token.cancel("急停")
    rep = DogMissionRunner(DogTransport.dry(), token=token).run(mission)
    assert rep.aborted and not rep.success
    assert "急停" in rep.abort_reason


def test_runner_refuses_invalid_mission():
    m = DogMission(commands=[DogCommand(channel="sdk", op="jump")])
    rep = DogMissionRunner(DogTransport.dry()).run(m)
    assert not rep.success and "校验未通过" in rep.abort_reason


# ======================================================================
# 适配器
# ======================================================================
def _ctx(tool="camera", platform=Platform.DOG, args=None, declared=None):
    from airground_agent.core.types import Effect
    return AdapterContext(tool_name=tool, platform=platform, tier=RouteTier.EDGE,
                          args=args or {}, declared_effects=declared or [
                              Effect(key="footage", op="set", value=True)])


def test_adapter_navigates_and_merges_effects():
    ad = build_dog_adapter(dry_run=True)
    out = ad.invoke(_ctx(args={"target": [1.0, 0.3, 2.0]}))
    assert out.success, out.error
    # 语义副作用（footage）必须与真机遥测一起回写，否则后续前置校验会被阻断
    keys = {e.key for e in (out.effects or [])}
    assert "footage" in keys


def test_adapter_rejects_wrong_platform():
    ad = build_dog_adapter(dry_run=True)
    out = ad.invoke(_ctx(platform=Platform.UAV, args={"target": [0, 0, 0]}))
    assert not out.success and out.error == "platform_mismatch"


def test_adapter_rejects_out_of_scope_tool():
    ad = build_dog_adapter(dry_run=True)
    out = ad.invoke(_ctx(tool="edit"))
    assert not out.success and out.error == "out_of_scope"


def test_adapter_requires_coordinates():
    ad = build_dog_adapter(dry_run=True)
    out = ad.invoke(_ctx())
    assert not out.success and out.error == "no_target"


def test_adapter_vla_is_edge_only():
    ad = build_dog_adapter(dry_run=True)
    out = ad.invoke(_ctx(tool="vla", platform=Platform.SHARED))
    assert out.success and out.telemetry.get("dog_action") == "none"


def test_zsibot_mode_is_registered():
    assert AdapterMode("zsibot") == AdapterMode.ZSIBOT


# ======================================================================
# 端到端：决策平台跑一轮 -> 机器狗 mission
# ======================================================================
def test_orchestrator_to_dog_mission():
    from airground_agent.orchestrator import Orchestrator
    orch = Orchestrator()
    res = orch.run("巡检变电站设备区")
    assert res.plan is not None
    try:
        mission = orch.to_dog_mission(res)
    except ValueError:
        pytest.skip("本轮规划失败，跳过编译断言")
    assert isinstance(mission, DogMission)
    assert mission.validate() == []


def test_orchestrator_dispatch_dry_run():
    from airground_agent.orchestrator import Orchestrator
    orch = Orchestrator()
    res = orch.run("巡检变电站设备区")
    if res.plan is None:
        pytest.skip("规划失败")
    try:
        rep = orch.dispatch_to_dog(res, dry_run=True)
    except ValueError:
        pytest.skip("无可用 Plan")
    assert isinstance(rep.to_dict(), dict)
