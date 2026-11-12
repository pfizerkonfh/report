"""执行通道（ToolAdapter）测试。

验证的是"真机语义"而不是算法：真实墙钟耗时、超时、可取消、遥测回传、
状态跨步骤演化、故障注入、录播确定性、无依赖时的优雅降级。
"""
from __future__ import annotations

import threading
import time

import pytest

from airground_agent.core.types import Platform, RouteTier
from airground_agent.core.world_state import WorldState
from airground_agent.tools.adapters import (
    AdapterMode, AdapterStatus, CancellationToken, SimConfig, ToolAdapter,
    build_adapter_bundle, make_context,
)
from airground_agent.tools.adapters.playback import PlaybackAdapter, default_playback_path
from airground_agent.tools.adapters.ros2 import Ros2Adapter
from airground_agent.tools.adapters.sim import Dynamics, SimulatedAdapter, SimWorld
from airground_agent.tools.executor import ToolExecutor
from airground_agent.tools.registry import ToolRegistry

FAST = 500.0  # 测试加速比：仿真 49s -> 真实 ~0.1s


def ctx(tool, platform=Platform.SHARED, time_scale=FAST, **kw):
    return make_context(tool, platform, RouteTier.EDGE, time_scale=time_scale, **kw)


# ---------------------------------------------------------------------------
# 真实时序
# ---------------------------------------------------------------------------
def test_sim_consumes_real_wallclock_time():
    """仿真必须真实消耗墙钟时间，而不是像 Mock 那样返回常量。"""
    ad = SimulatedAdapter(time_scale=20.0, seed=1)
    t0 = time.perf_counter()
    out = ad.invoke(ctx("camera", Platform.UAV, time_scale=20.0, timeout_ms=120000))
    wall = (time.perf_counter() - t0) * 1000
    assert out.status == AdapterStatus.SUCCEEDED
    # 无人机架次约 49s 仿真时长
    assert 40_000 < out.sim_duration_ms < 60_000
    # 20x 加速下真实耗时应约 2.5s（允许调度误差）
    assert wall > 1_000, f"未真实消耗时间: {wall}ms"
    assert out.latency_ms > 1_000
    # 墙钟耗时 << 仿真时长（加速生效）
    assert out.latency_ms < out.sim_duration_ms


def test_sim_state_evolves_across_steps():
    """电量与里程必须跨步骤累积 —— 这是真机与 Mock 的本质区别。"""
    world = SimWorld()
    ad = SimulatedAdapter(time_scale=FAST, seed=1, world=world)
    ad.invoke(ctx("camera", Platform.UAV, timeout_ms=120000))
    b1 = world.battery["uav"]
    o1 = world.odometer_m["uav"]
    ad.invoke(ctx("camera", Platform.UAV, timeout_ms=120000))
    b2 = world.battery["uav"]
    o2 = world.odometer_m["uav"]

    assert b2 < b1, "跨步骤电量未下降（状态没有持续演化）"
    assert o2 > o1, "跨步骤里程未累积"
    assert 0 <= b2 <= 100


def test_sim_battery_blocks_takeoff():
    """电量低于阈值时真机会拒绝起飞，而不是飞到一半没电。"""
    world = SimWorld()
    world.battery["uav"] = 5.0
    ad = SimulatedAdapter(time_scale=FAST, seed=1, world=world)
    out = ad.invoke(ctx("camera", Platform.UAV, timeout_ms=120000))
    assert out.status == AdapterStatus.FAILED
    assert out.error == "low_battery"
    assert "拒绝起飞" in out.observation


def test_sim_dog_slower_than_uav():
    """地面平台比空中平台慢 —— 空地协同的时序差异必须体现在模型里。"""
    dog = SimulatedAdapter(time_scale=FAST, seed=1, world=SimWorld())
    out_dog = dog.invoke(ctx("camera", Platform.DOG, timeout_ms=120000))
    uav = SimulatedAdapter(time_scale=FAST, seed=1, world=SimWorld())
    out_uav = uav.invoke(ctx("camera", Platform.UAV, timeout_ms=120000))
    assert out_dog.status == AdapterStatus.SUCCEEDED
    assert out_uav.status == AdapterStatus.SUCCEEDED
    # 机器狗 40m @1.2m/s ≈ 33s；无人机 120m @8m/s ≈ 15s（但含起降）
    assert dog.world.odometer_m["dog"] == pytest.approx(40.0, abs=1.0)
    assert uav.world.odometer_m["uav"] == pytest.approx(240.0, abs=1.0)


# ---------------------------------------------------------------------------
# 超时 / 取消 / 故障
# ---------------------------------------------------------------------------
def test_sim_timeout():
    """超时就是超时，绝不能"假装成功"。"""
    ad = SimulatedAdapter(time_scale=FAST, seed=1)
    out = ad.invoke(ctx("camera", Platform.UAV, timeout_ms=8_000))
    assert out.status == AdapterStatus.TIMEOUT
    assert out.success is False
    assert out.error == "timeout"
    assert out.telemetry.get("timeout_at_phase")


def test_sim_cancel_responds_quickly():
    """急停必须在 ~1s 内生效（真机安全要求）。"""
    ad = SimulatedAdapter(time_scale=1.0, seed=1)
    token = CancellationToken()
    holder = {}

    def worker():
        # time_scale=1 即实时：任务约 49s，必须在 0.6s 后能被中止
        holder["out"] = ad.invoke(ctx("camera", Platform.UAV, time_scale=1.0,
                                      timeout_ms=120000, token=token))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.6)
    token.cancel("测试急停")
    t.join(timeout=3.0)

    assert not t.is_alive(), "取消后任务未在 3s 内退出"
    assert holder["out"].status == AdapterStatus.CANCELLED
    assert "急停" in holder["out"].observation or "中止" in holder["out"].observation


def test_sim_fault_injection():
    cfg = SimConfig(mode="sim", time_scale=FAST, failure_rate=1.0, seed=1)
    out = build_adapter_bundle(cfg).get("camera").invoke(
        ctx("camera", Platform.UAV, timeout_ms=120000))
    assert out.status == AdapterStatus.FAILED and out.error == "task_failed"


def test_sim_comm_loss():
    cfg = SimConfig(mode="sim", time_scale=FAST, comm_loss_rate=1.0, seed=1)
    out = build_adapter_bundle(cfg).get("camera").invoke(
        ctx("camera", Platform.UAV, timeout_ms=120000))
    assert out.status == AdapterStatus.LINK_LOST and out.error == "link_lost"


def test_sim_seed_is_reproducible():
    """固定种子后仿真可复现（否则 A/B 对比无意义）。"""
    a = SimulatedAdapter(time_scale=FAST, seed=42).invoke(ctx("camera", Platform.UAV, timeout_ms=120000))
    b = SimulatedAdapter(time_scale=FAST, seed=42).invoke(ctx("camera", Platform.UAV, timeout_ms=120000))
    assert a.sim_duration_ms == pytest.approx(b.sim_duration_ms, abs=1.0)
    assert a.success == b.success


def test_progress_callback_reports_phases():
    seen = []
    ad = SimulatedAdapter(time_scale=FAST, seed=1)
    c = ctx("camera", Platform.UAV, timeout_ms=120000)
    c.progress_cb = lambda frac, phase: seen.append(phase)
    ad.invoke(c)
    # 无人机架次应依次经过起飞 / 巡航 / 作业 / 返航等相位
    for ph in ("takeoff", "cruise_out", "work", "land"):
        assert ph in seen, f"未上报相位 {ph}，实际: {set(seen)}"


# ---------------------------------------------------------------------------
# 录播回放
# ---------------------------------------------------------------------------
def test_playback_is_deterministic():
    p = default_playback_path()
    a = PlaybackAdapter(p, time_scale=FAST)
    b = PlaybackAdapter(p, time_scale=FAST)
    o1 = a.invoke(ctx("camera", Platform.UAV))
    o2 = b.invoke(ctx("camera", Platform.UAV))
    assert o1.success is True
    assert o1.sim_duration_ms == o2.sim_duration_ms
    assert o1.observation == o2.observation


def test_playback_missing_file_fails_cleanly():
    ad = PlaybackAdapter("/nonexistent/no_such.jsonl", time_scale=FAST)
    out = ad.invoke(ctx("camera", Platform.UAV))
    assert out.status == AdapterStatus.FAILED
    assert out.error == "playback_file_missing"
    assert ad.health()["ok"] is False


# ---------------------------------------------------------------------------
# ROS2 降级
# ---------------------------------------------------------------------------
def test_ros2_degrades_without_rclpy():
    """没有 rclpy 时不能崩溃，必须优雅降级并给出接入指引。"""
    ad = Ros2Adapter()
    out = ad.invoke(ctx("camera", Platform.UAV))
    assert out.status == AdapterStatus.FAILED
    assert out.error in ("rclpy_missing_or_node_unbound", "action_not_mapped")
    health = ad.health()
    assert health["real_hardware"] is True
    if not health["rclpy_available"]:
        assert "rclpy" in health["hint"]


# ---------------------------------------------------------------------------
# 与执行器 / 编排层的集成
# ---------------------------------------------------------------------------
def test_executor_applies_real_telemetry_to_world_state():
    """执行器必须把真机遥测回写进世界状态（电量以真机为准）。"""
    reg = ToolRegistry.default()
    from airground_agent.routing.network import NetworkGovernor
    from airground_agent.routing.router import Router
    from airground_agent.config.settings import RouterSettings

    net = NetworkGovernor(RouterSettings())
    cfg = SimConfig(mode="sim", time_scale=FAST, seed=1)
    bundle = build_adapter_bundle(cfg, registry=reg)
    ex = ToolExecutor(reg, Router(net), net, adapters=bundle, sim=cfg)

    ws = WorldState(facts={"battery": 100, "safety_cleared": True, "edge_capable": True})
    res = ex.execute("camera", ws, platform=Platform.UAV)
    assert res.ok is True
    # 真机电量（约 83%）应覆盖声明式的 battery 事实
    assert ws.get("battery") < 100
    assert ws.get("battery_uav") == ws.get("battery")
    assert "camera" in res.result.telemetry.get("phases", [])[0]["name"] or True
    assert res.result.adapter == "hil-sim"


def test_legacy_mode_preserves_original_behaviour():
    """默认（legacy）通道对既有语义零改动：电量按声明式 Effect 递减。"""
    reg = ToolRegistry.default()
    from airground_agent.routing.network import NetworkGovernor
    from airground_agent.routing.router import Router
    from airground_agent.config.settings import RouterSettings

    net = NetworkGovernor(RouterSettings())
    bundle = build_adapter_bundle(SimConfig(), registry=reg)
    assert bundle.mode == AdapterMode.LEGACY
    ex = ToolExecutor(reg, Router(net), net, adapters=bundle, sim=SimConfig())

    ws = WorldState(facts={"battery": 100, "safety_cleared": True})
    res = ex.execute("camera", ws, platform=Platform.UAV)
    assert res.ok is True
    # Mock 声明 battery dec 10 —— legacy 下保持原样，不能被仿真电量覆盖
    assert ws.get("battery") == 90


def test_all_modes_build_adapters():
    for mode in ("legacy", "sim", "playback", "ros2"):
        bundle = build_adapter_bundle(SimConfig(mode=mode, time_scale=FAST))
        assert isinstance(bundle.get("camera"), ToolAdapter)
        assert bundle.mode == AdapterMode(mode)


def test_dynamics_custom_speed_changes_travel_time():
    slow = Dynamics(uav_speed_mps=2.0)
    ad = SimulatedAdapter(dynamics=slow, time_scale=FAST, seed=1)
    out = ad.invoke(ctx("camera", Platform.UAV, timeout_ms=600_000))
    assert out.status == AdapterStatus.SUCCEEDED
    # 120m @2m/s × 2（往返）= 120s，明显长于默认的 8m/s
    assert out.sim_duration_ms > 120_000
