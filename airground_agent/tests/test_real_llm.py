"""真实 LLM 接入测试（默认跳过，需 config/real_model.yaml 且 Key 可用）。

运行：  python -m pytest tests/test_real_llm.py -m real_llm -s
说明：  该测试会真实调用中转站 API，产生少量 token 费用。
        若 CONFIG 不存在、或 Key 无效/网络不通（ping 失败），测试会自动 skip，
        不会让日常 `pytest` 全量变红。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from airground_agent.config.settings import Settings
from airground_agent.core.llm import build_backend
from airground_agent.core.schemas import ReasonerDecision
from airground_agent.core.world_state import WorldState
from airground_agent.planner.planner import Planner
from airground_agent.tools.registry import ToolRegistry

CONFIG = Path(__file__).resolve().parent.parent / "config" / "real_model.yaml"

pytestmark = pytest.mark.skipif(
    not CONFIG.exists(),
    reason="缺少真实模型配置 config/real_model.yaml（含 API Key），跳过真实 LLM 测试",
)


@pytest.fixture(scope="module")
def live_backend():
    """构建真实后端并做连通性自检；不可用则整个模块 skip（不视为失败）。"""
    settings = Settings.from_file(str(CONFIG))
    backend = build_backend(**settings.llm.model_dump())
    try:
        backend.ping()
    except RuntimeError as e:
        pytest.skip(f"真实端点不可用，跳过真实 LLM 测试：{e}")
    return backend


@pytest.mark.real_llm
def test_real_planner_returns_dag(live_backend):
    reg = ToolRegistry.default()
    planner = Planner(reg, backend=live_backend, max_retries=2)
    plan = planner.plan("帮我把这一天的素材拍好剪成短片并去除人脸", WorldState())
    assert plan is not None
    assert len(plan.subtasks) >= 1
    # 规划的子任务工具必须已注册，保证可落地执行
    for st in plan.subtasks.values():
        assert reg.has(st.tool_hint), f"规划使用了未注册工具: {st.tool_hint}"
    print("\n[real] plan subtasks:", [s.id for s in plan.subtasks.values()])


@pytest.mark.real_llm
def test_real_reasoner_returns_decision(live_backend):
    decision = live_backend.generate(
        "[ROLE:REASONER] 决定本步工具与参数，必须返回结构化 JSON。",
        "子任务：t1 无人机拍摄巡检素材；建议工具：camera；世界状态：{safety_cleared:true}",
        ReasonerDecision,
        context={},
    )
    assert decision.subtask_id == "t1"
    assert decision.tool == "camera"
    print("\n[real] decision:", decision.model_dump())
