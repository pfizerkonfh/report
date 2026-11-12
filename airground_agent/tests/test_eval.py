from airground_agent.core.task_model import TraceStep
from airground_agent.core.types import FailureLayer, RunStatus
from airground_agent.eval.metrics import compute_metrics
from airground_agent.eval.attribution import attribute, ATTRIBUTION_MATRIX
from airground_agent.eval.ab_test import ABTest, ABVerdict
from airground_agent.config.settings import Settings


def _trace():
    return [
        TraceStep(step_id=1, subtask_id="a", precondition_ok=True, latency_ms=20.0, observation="ok"),
        TraceStep(step_id=2, subtask_id="b", precondition_ok=True, latency_ms=50.0, observation="ok"),
        TraceStep(step_id=3, subtask_id="c", precondition_ok=False, latency_ms=10.0,
                  observation="前置校验阻断: safety_cleared == True"),
    ]


def test_compute_metrics_success_vs_failure():
    ok_trace = _trace()[:-1]
    rec = compute_metrics("g", RunStatus.SUCCESS, plan=None, trace=ok_trace, final_facts={}, csat=5.0)
    assert rec.precondition_violations == 0
    assert rec.action_reasonableness == 1.0
    assert rec.latency_p95 == 48.5
    rec2 = compute_metrics("g", RunStatus.FAILED, plan=None, trace=_trace(), final_facts={}, csat=2.0)
    assert rec2.precondition_violations == 1
    assert rec2.action_reasonableness == round(0.5 * (1 - 1 / 3), 4)


def test_attribution_matrix():
    a = attribute("前置校验阻断: safety_cleared == True")
    assert a.layer == FailureLayer.PLANNING
    a2 = attribute("动作偏离全局目标")
    assert a2.layer == FailureLayer.ACTION
    assert len(ATTRIBUTION_MATRIX) >= 5


def test_ab_test_guardrail_rollback():
    ab = ABTest(Settings().eval)
    control = compute_metrics("g", RunStatus.SUCCESS, None, _trace()[:-1], {}, csat=4.5)
    treatment_bad = compute_metrics("g", RunStatus.FAILED, None, _trace(), {}, csat=2.0)
    v = ab.compare(control, treatment_bad)
    assert isinstance(v, ABVerdict)
    assert v.keep_treatment is False
    # 不劣化应保留
    treatment_ok = compute_metrics("g", RunStatus.SUCCESS, None, _trace()[:-1], {}, csat=4.8)
    v2 = ab.compare(control, treatment_ok)
    assert v2.keep_treatment is True
