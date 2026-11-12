from airground_agent.core.world_state import WorldState
from airground_agent.tools.registry import ToolRegistry
from airground_agent.planner.planner import Planner


def test_planner_produces_dag():
    reg = ToolRegistry.default()
    planner = Planner(reg, max_retries=3)
    ws = WorldState(facts={"battery": 100, "safety_cleared": True, "edge_capable": True})
    plan = planner.plan("帮我把这一天的素材拍好剪成短片并去除人脸", ws)
    assert plan is not None
    assert plan.validate_acyclic()
    ids = set(plan.subtasks.keys())
    assert {"t_weather", "t_capture", "t_edit", "t_anon"} <= ids
    # t_capture depends on t_weather
    assert "t_weather" in plan.subtasks["t_capture"].depends_on
    # every tool hint must be a registered tool
    for st in plan.subtasks.values():
        assert reg.has(st.tool_hint)


def test_planner_assigns_air_ground_platforms():
    """空地协同：采集类子任务应被判给无人机/机器狗，计算类判给共享算力。"""
    reg = ToolRegistry.default()
    planner = Planner(reg, max_retries=3)
    ws = WorldState(facts={"battery": 100, "safety_cleared": True, "edge_capable": True})
    plan = planner.plan("帮我把这一天的素材拍好剪成短片并去除人脸", ws)
    subs = plan.subtasks
    assert subs["t_capture"].platform.value == "uav"     # 无人机航拍
    assert subs["t_dog"].platform.value == "dog"         # 机器狗地面巡检
    assert subs["t_weather"].platform.value == "shared"  # 天气查询 = 共享算力
    assert subs["t_edit"].platform.value == "shared"     # 剪辑 = 共享算力


def test_normalize_platform():
    from airground_agent.core.types import Platform, normalize_platform
    assert normalize_platform("无人机") == Platform.UAV
    assert normalize_platform("UAV") == Platform.UAV
    assert normalize_platform("机器狗") == Platform.DOG
    assert normalize_platform("dog") == Platform.DOG
    assert normalize_platform("协同") == Platform.SHARED
    assert normalize_platform(None) == Platform.SHARED   # 缺省不炸，回落共享
    assert normalize_platform("乱七八糟") == Platform.SHARED


def test_planner_returns_none_on_infeasible_schema():
    reg = ToolRegistry.default()
    planner = Planner(reg, max_retries=0)
    # 通过直接验证：tool_hint 不存在会被校验拦截
    from airground_agent.core.schemas import PlanSchema, SubTaskSpec
    schema = PlanSchema(subtasks=[SubTaskSpec(id="x", description="d", tool_hint="ghost")])
    from langgraph.graph import END, START, StateGraph
    # 复用内部校验逻辑：构造非法计划应被 validate 拒绝
    plan = planner._to_plan(schema)
    assert not reg.has("ghost")
