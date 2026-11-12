from airground_agent import Orchestrator
from airground_agent.core.types import RunStatus
from airground_agent.core.task_model import ExperienceEntry


def test_reasoner_happy_path():
    orch = Orchestrator()
    ws = {"battery": 100, "safety_cleared": True, "edge_capable": True, "uav_online": True}
    res = orch.run("帮我把这一天的素材拍好剪成短片并去除人脸", ws)
    assert res.status == RunStatus.SUCCESS
    assert res.success_criteria_met is True
    # 空地协同：t_weather + t_capture(无人机) + t_dog(机器狗) + t_edit + t_anon
    assert len(res.trace) == 5
    assert all(t.precondition_ok for t in res.trace)
    assert res.metrics.planning_success is True
    assert res.metrics.action_reasonableness == 1.0
    # 计划中必须同时出现无人机与机器狗两个物理平台
    platforms = {st.platform.value for st in res.plan.subtasks.values()}
    assert {"uav", "dog"} <= platforms
    # 每一步执行都应带有所属平台
    assert all(t.platform is not None for t in res.trace)


def test_reasoner_reflexion_on_infeasible_precondition():
    orch = Orchestrator()
    ws = {"battery": 100, "safety_cleared": False, "edge_capable": True, "uav_online": True}
    res = orch.run("帮我把这一天的素材拍好剪成短片并去除人脸", ws)
    assert res.status == RunStatus.FAILED
    assert res.reflection is not None
    assert res.reflection.root_cause_layer.value == "planning"
    # 经验已回流
    assert len(orch.buffer.entries) >= 1
    assert isinstance(orch.buffer.entries[-1], ExperienceEntry)


def test_experience_buffer_persist(tmp_path):
    from airground_agent.reasoner.reflexion import ExperienceBuffer
    from airground_agent.core.types import FailureLayer
    p = tmp_path / "exp.jsonl"
    buf = ExperienceBuffer(str(p))
    buf.add(ExperienceEntry(goal_pattern="采集", failure_layer=FailureLayer.PLANNING, lesson="test-lesson"))
    buf2 = ExperienceBuffer(str(p))
    assert "test-lesson" in buf2.all_lessons()
