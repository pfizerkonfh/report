"""使用真实大模型（OpenAI 兼容中转站）跑端到端示例。

配置见 ../config/real_model.yaml（含 API Key，已被 .gitignore 忽略）。
运行：  python -m examples.run_real
或：    AIRGROUND_CONFIG=airground_agent/config/real_model.yaml python -m examples.run_real
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from airground_agent.config.settings import Settings
from airground_agent.orchestrator import Orchestrator

console = Console()
CONFIG = Path(__file__).resolve().parent.parent / "config" / "real_model.yaml"


def _mask(key: str) -> str:
    if not key:
        return "<empty>"
    return key[:6] + "..." + key[-4:] if len(key) > 12 else "***"


def main() -> int:
    if not CONFIG.exists():
        console.print(f"[red]未找到真实模型配置：{CONFIG}[/red]")
        console.print("请新建 config/real_model.yaml（可参考本文件头部注释），填入中转站地址与 API Key。")
        return 1

    settings = Settings.from_file(str(CONFIG))
    llm = settings.llm
    if llm.backend == "mock" or not llm.base_url:
        console.print("[yellow]警告：llm.backend 不是 openai 或未设置 base_url，本次将退回 mock 模式。[/yellow]")

    console.print(Panel.fit(
        f"[bold]真实模型测试环境[/bold]\n"
        f"backend   : {llm.backend}\n"
        f"model     : {llm.model_name}\n"
        f"base_url  : {llm.base_url}\n"
        f"api_key   : {_mask(llm.api_key)}",
        title="配置", border_style="cyan"))

    orch = Orchestrator(settings=settings)
    if llm.backend == "openai" and llm.base_url:
        try:
            orch.backend.ping()  # type: ignore[attr-defined]
            console.print("[green]连通自检通过[/green]\n")
        except RuntimeError as e:
            console.print(f"[red]连通自检失败：{e}[/red]")
            console.print("请修改 config/real_model.yaml 或环境变量后重试（python -m examples.check_endpoint）。")
            return 2

    ws_init = {
        "battery": 100, "safety_cleared": True, "edge_capable": True, "weather_ok": False,
        "footage": False, "edited": False, "anonymized": False, "uav_online": True,
    }
    goal = "帮我把这一天的素材拍好剪成短片并去除人脸"
    console.print(f"[bold]目标：[/bold]{goal}\n")

    res = orch.run(goal, ws_init)

    console.print(f"\n[bold]任务状态：[/bold]{res.status.value}")
    if res.plan:
        t = Table(title="规划 DAG（来自真实模型）")
        t.add_column("subtask_id")
        t.add_column("description")
        t.add_column("tool_hint")
        t.add_column("depends_on")
        for st in res.plan.subtasks.values():
            t.add_row(st.id, st.description, st.tool_hint, ", ".join(st.depends_on) or "-")
        console.print(t)

    if res.trace:
        t2 = Table(title="执行轨迹")
        t2.add_column("#")
        t2.add_column("subtask")
        t2.add_column("tool")
        t2.add_column("precond_ok")
        t2.add_column("tier")
        t2.add_column("latency_ms")
        t2.add_column("thought")
        for tr in res.trace:
            t2.add_row(str(tr.step_id), tr.subtask_id, tr.action.get("tool", ""),
                       str(tr.precondition_ok), str(tr.tier_used), f"{tr.latency_ms:.1f}",
                       (tr.thought or "")[:40])
        console.print(t2)

    if res.metrics:
        m = res.metrics
        console.print(
            f"[bold]评测：[/bold] 规划成功率={m.planning_success} | 动作合理度={m.action_reasonableness:.2f} | "
            f"端到端={m.end_to_end_ms:.1f}ms | p95={m.latency_p95:.1f}ms | CSAT={m.csat}")

    if res.reflection:
        console.print(f"[red]触发 Reflexion，根因层={res.reflection.root_cause_layer.value}[/red]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
