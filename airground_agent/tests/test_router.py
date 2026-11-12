from airground_agent.config.settings import Settings
from airground_agent.routing.network import NetworkGovernor
from airground_agent.routing.router import Router
from airground_agent.routing.protocol import FallbackStrategy
from airground_agent.tools.registry import ToolRegistry


def _router():
    reg = ToolRegistry.default()
    net = NetworkGovernor(Settings().router)
    return reg, net, Router(net)


def test_privacy_forces_edge():
    reg, net, router = _router()
    d = router.decide(reg.get("anonymize"), "anonymize")
    assert d.chosen_tier.value == "edge"
    assert d.fallback_plan in (FallbackStrategy.NONE, FallbackStrategy.RETRY_BACKOFF)


def test_high_compute_goes_cloud():
    reg, net, router = _router()
    d = router.decide(reg.get("edit"), "edit")
    assert d.chosen_tier.value == "cloud"


def test_cloud_degrade_fallback():
    reg, net, router = _router()
    net.set_fault(rtt_ms=400, jitter_ms=60, loss_rate=0.0)
    # edit 不可边侧 -> 云端降级时只能 hold/重试
    d = router.decide(reg.get("edit"), "edit")
    assert d.fallback_plan in (FallbackStrategy.HOLD_STALE, FallbackStrategy.RETRY_BACKOFF)
    # 可边侧工具 -> 仍走边侧，但网络抖动触发退避/保持
    d2 = router.decide(reg.get("anonymize"), "anonymize")
    assert d2.chosen_tier.value == "edge"
    assert d2.fallback_plan in (FallbackStrategy.RETRY_BACKOFF, FallbackStrategy.HOLD_STALE)
    net.reset()
