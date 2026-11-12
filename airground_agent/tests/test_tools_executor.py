from airground_agent.core.world_state import WorldState
from airground_agent.config.settings import Settings
from airground_agent.routing.network import NetworkGovernor
from airground_agent.routing.router import Router
from airground_agent.tools.registry import ToolRegistry
from airground_agent.tools.executor import ToolExecutor


def _executor():
    reg = ToolRegistry.default()
    net = NetworkGovernor(Settings().router)
    router = Router(net)
    return ToolExecutor(reg, router, net), reg


def test_precondition_block():
    ex, reg = _executor()
    ws = WorldState(facts={"battery": 5, "safety_cleared": True})  # battery < 20
    r = ex.execute("camera", ws, {})
    assert r.blocked is True
    assert any("battery" in f for f in r.precondition_failed)
    assert r.ok is False


def test_success_runs_and_mutates_ws():
    ex, reg = _executor()
    ws = WorldState(facts={"battery": 100, "safety_cleared": True, "edge_capable": True})
    r = ex.execute("camera", ws, {})
    assert r.ok is True
    assert ws.get("footage") is True
    assert r.decision is not None


def test_unregistered_tool():
    ex, reg = _executor()
    ws = WorldState(facts={})
    r = ex.execute("no_such_tool", ws, {})
    assert r.ok is False and r.blocked is False
