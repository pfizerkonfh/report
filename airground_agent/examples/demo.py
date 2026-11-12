"""端到端演示脚本（mock 后端，无需联网/API Key 即可运行）。

运行：
    cd airground_agent
    python -m examples.demo

覆盖：① 长程任务 happy path 与评测；② Reflexion 失败经验回流；
③ 端侧 VLA 时延优化（结构/量化/算子三层叠加收益）；④ 端-边-云路由与抖动兜底；
⑤ 统一评测聚合 + 跨层归因 + A/B 护栏。
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from airground_agent import Orchestrator
from airground_agent.config.settings import Settings, VLASettings
from airground_agent.tools.latency import simulate_vla_latency
from airground_agent.tools.registry import ToolRegistry
from airground_agent.routing.router import Router
from airground_agent.routing.network import NetworkGovernor
from airground_agent.eval.attribution import attribute

console = Console()


def demo_happy_path(orch: Orchestrator) -> None:
    console.print(Panel("① 长程任务：素材拍摄 → 剪辑 → 去人脸（happy path）", style="cyan"))
    ws = {"battery": 100, "safety_cleared": True, "edge_capable": True, "uav_online": True}
    res = orch.run("帮我把这一天的素材拍好剪成短片并去除人脸", ws)
    t = Table(title="执行轨迹")
    t.add_column("step"); t.add_column("subtask"); t.add_column("tool")
    t.add_column("tier"); t.add_column("lat(ms)"); t.add_column("ok")
    for s in res.trace:
        t.add_row(str(s.step_id), s.subtask_id, s.action.get("tool", ""),
                  str(s.tier_used), f"{s.latency_ms:.1f}", "Y" if s.precondition_ok else "N")
    console.print(t)
    m = res.metrics
    console.print(f"[green]状态={res.status.value}[/] 规划成功率={m.planning_success} "
                  f"动作合理度={m.action_reasonableness} 时延p95={m.latency_p95}ms "
                  f"端到端={m.end_to_end_ms}ms CSAT={m.csat}")


def demo_reflexion(orch: Orchestrator) -> None:
    console.print(Panel("② Reflexion：前置不可满足 → 重规划 → 经验回流", style="magenta"))
    ws = {"battery": 100, "safety_cleared": False, "edge_capable": True, "uav_online": True}
    res = orch.run("帮我把这一天的素材拍好剪成短片并去除人脸", ws)
    if res.reflection:
        console.print(f"[red]任务失败，根因层={res.reflection.root_cause_layer.value}[/]")
        console.print(f"自我批评：{res.reflection.critique}")
        console.print(f"沉淀经验：{res.reflection.lessons}")
    console.print(f"经验缓冲条目数={len(orch.buffer.entries)}")


def demo_vla_latency() -> None:
    console.print(Panel("③ 端侧 VLA 时延优化（3B/INT4，目标 20Hz→50ms 预算）", style="yellow"))
    base = VLASettings(baseline_latency_ms=65.0, enable_moe=False, enable_distill=False,
                       quant_mode="awq_int4", enable_flash_attn=False, enable_paged_attn=False)
    opt = VLASettings(baseline_latency_ms=65.0, enable_moe=True, enable_distill=True,
                      quant_mode="hybrid", enable_flash_attn=True, enable_paged_attn=True)
    b = simulate_vla_latency(base)
    o = simulate_vla_latency(opt)
    t = Table(title="VLA 单步推理时延分解")
    t.add_column("阶段"); t.add_column("基线(ms)"); t.add_column("优化(ms)")
    t.add_row("结构层(MoE+蒸馏)", f"{b.after_structure:.1f}", f"{o.after_structure:.1f}")
    t.add_row("量化层(AWQ/SmoothQuant)", f"{b.after_quant:.1f}", f"{o.after_quant:.1f}")
    t.add_row("算子层(Flash/PagedAttn)", f"{b.after_kernel:.1f}", f"{o.after_kernel:.1f}")
    t.add_row("最终单步", f"{b.final:.1f}", f"{o.final:.1f}")
    console.print(t)
    console.print(f"基线频率≈{1000/b.final:.1f}Hz，优化后≈{1000/o.final:.1f}Hz，"
                  f"满足≥20Hz：{'是' if 1000/o.final >= 20 else '否'}")


def demo_routing() -> None:
    console.print(Panel("④ 端-边-云路由与网络抖动兜底", style="green"))
    reg = ToolRegistry.default()
    net = NetworkGovernor(Settings().router)
    router = Router(net)
    t = Table(title="路由决策（不同网络条件）")
    t.add_column("工具"); t.add_column("网络"); t.add_column("tier"); t.add_column("fallback")
    for tool_name in ["camera", "edit", "anonymize", "vla"]:
        tool = reg.get(tool_name)
        net.reset(); d = router.decide(tool, tool_name)
        t.add_row(tool_name, "正常", d.chosen_tier.value, d.fallback_plan.value)
        net.set_fault(rtt_ms=400, jitter_ms=60)
        d2 = router.decide(tool, tool_name)
        t.add_row(tool_name, "RTT=400ms", d2.chosen_tier.value, d2.fallback_plan.value)
        net.reset()
    console.print(t)


def demo_eval(orch: Orchestrator) -> None:
    console.print(Panel("⑤ 统一评测：聚合 + 跨层归因 + A/B", style="blue"))
    cases = [
        {"goal": "帮我把这一天的素材拍好剪成短片并去除人脸",
         "ws": {"battery": 100, "safety_cleared": True, "edge_capable": True, "uav_online": True}},
        {"goal": "端侧 VLA 控制无人机巡检",
         "ws": {"edge_capable": True}},
        {"goal": "帮我把这一天的素材拍好剪成短片并去除人脸",
         "ws": {"battery": 100, "safety_cleared": False, "edge_capable": True, "uav_online": True}},
    ]
    for c in cases:
        orch.run(c["goal"], c["ws"])
    agg = orch.harness.aggregate()
    console.print("聚合指标：", agg)
    # 跨层归因示例
    for sym in ["前置校验阻断: safety_cleared == True", "动作偏离全局目标", "云端降级"]:
        a = attribute(sym)
        console.print(f"现象=「{sym}」 -> 根因层={a.layer.value} ({a.reason})")


def main() -> None:
    orch = Orchestrator()
    demo_happy_path(orch)
    demo_reflexion(orch)
    demo_vla_latency()
    demo_routing()
    demo_eval(orch)


if __name__ == "__main__":
    main()
