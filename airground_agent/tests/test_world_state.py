from airground_agent.core.world_state import WorldState
from airground_agent.core.types import Condition, Effect, ComparisonOp


def test_satisfies_basic():
    ws = WorldState(facts={"battery": 50, "footage": True})
    assert ws.satisfies(Condition(key="battery", op=ComparisonOp.GE, value=20))
    assert not ws.satisfies(Condition(key="battery", op=ComparisonOp.GE, value=80))
    assert ws.satisfies(Condition(key="footage", op=ComparisonOp.EXISTS))
    assert not ws.satisfies(Condition(key="edited", op=ComparisonOp.EXISTS))


def test_apply_effects():
    ws = WorldState(facts={"battery": 100})
    ws.apply(Effect(key="footage", op="set", value=True))
    ws.apply(Effect(key="battery", op="dec", value=10))
    ws.apply(Effect(key="marker", op="inc", value=3))
    assert ws.get("footage") is True
    assert ws.get("battery") == 90
    assert ws.get("marker") == 3
    ws.apply(Effect(key="footage", op="delete"))
    assert not ws.has("footage")


def test_check_all_returns_only_failed():
    ws = WorldState(facts={"battery": 10})
    conds = [
        Condition(key="battery", op=ComparisonOp.GE, value=20),
        Condition(key="safety", op=ComparisonOp.EXISTS),
    ]
    failed = ws.check_all(conds)
    assert len(failed) == 2


def test_clone_is_independent():
    ws = WorldState(facts={"a": 1})
    c = ws.clone()
    c.apply(Effect(key="a", op="set", value=2))
    assert ws.get("a") == 1 and c.get("a") == 2
