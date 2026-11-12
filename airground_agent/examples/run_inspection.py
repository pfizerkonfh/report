"""真实巡检任务端到端跑测：真实 LLM 规划 + 真实 3DGS 感知 + 硬件在环仿真执行。

目的：验证 Planner 在拥有真实感知工具（route_plan / map_query / scene_inspect / height_probe）
与真实场景事实（scene_*）的情况下，是否会主动把它们编进计划并真正调用。

运行： python -m examples.run_inspection
"""
from __future__ import annotations

import time
from pathlib import Path

from airground_agent.config.settings import Settings
from airground_agent.orchestrator import Orchestrator
from airground_agent.tools.adapters.factory import SimConfig

CONFIG = Path(__file__).resolve().parent.parent / "config" / "real_model.yaml"

GOAL = (
    "对我实验室场景做一次空地协同巡检："
    "先用无人机从空中查看场景整体与绝缘子等高空设备，"
    "再让机器狗进入室内沿地面通道巡检开关柜与表计，"
    "过程中需要依据 3DGS 重建出的占据栅格地图规划一条真实可行的巡检路径，"
    "最后融合成巡检短片并对画面做人脸脱敏。"
)


def main() -> int:
    if not CONFIG.exists():
        print(f"未找到真实模型配置：{CONFIG}")
        return 1
    settings = Settings.from_file(str(CONFIG))
    llm = settings.llm
    print("=" * 78)
    print("真实巡检任务 · 端到端跑测")
    print("=" * 78)
    print(f"  LLM        : {llm.backend} / {llm.model_name} @ {llm.base_url}")
    print(f"  感知层     : enabled={settings.perception.enabled} "
          f"dir={settings.perception.data_dir or '(自动发现)'}")
    print(f"  执行通道   : sim (time_scale=20, 真实墙钟受加速)")
    print(f"  任务       : {GOAL}")
    print("=" * 78)

    # sim 通道：真实时序 + 真实超时 + 遥测回传，time_scale 加速墙钟
    sim = SimConfig(mode="sim", time_scale=20.0, failure_rate=0.0,
                    comm_loss_rate=0.0, timeout_ms=120000)
    orch = Orchestrator(settings=settings, sim=sim)

    print(f"\n[感知层] {'已加载' if orch.scene else '未加载'}", end="")
    if orch.scene:
        print(f"  source={orch.scene.source}")
        print("  " + orch.scene.describe())
    print(f"\n[工具集] {orch.registry.list()}")

    ws_init = {
        "battery": 100, "safety_cleared": True, "edge_capable": True,
        "weather_ok": False, "footage": False, "edited": False,
        "anonymized": False, "uav_online": True,
    }

    print("\n[开始] 调用真实大模型规划中（中转站偶发 500，会自动重试）...\n")
    t0 = time.time()
    r = orch.run(GOAL, ws_init=ws_init, csat=4.5)
    el = time.time() - t0

    print("\n" + "=" * 78)
    print(f"结果：status={r.status.value}   步数={len(r.trace)}   墙钟={el:.1f}s")
    print("=" * 78)

    # 落盘，避免打印异常导致结果丢失
    out = Path(__file__).resolve().parent / "last_inspection.json"
    try:
        out.write_text(r.model_dump_json(indent=2), encoding="utf-8")
        print(f"[落盘] {out}")
    except Exception as e:  # pragma: no cover
        print(f"[落盘失败] {e}")

    # ---- 计划（真实 LLM 可能直接吐字符串列表，需兼容两种形态）----
    print("\n【计划】")
    subs = (r.plan.subtasks if r.plan else []) or []
    for i, st in enumerate(subs, 1):
        if isinstance(st, str):
            print(f"  {i}. {st[:96]}")
            continue
        plat = getattr(st, "platform", None)
        plat = plat.value if hasattr(plat, "value") else (plat or "-")
        sid = getattr(st, "id", "-")
        desc = getattr(st, "description", "") or ""
        hint = getattr(st, "tool_hint", "") or ""
        print(f"  {i}. [{str(plat):6s}] {str(sid):10s} {desc[:56]}")
        if hint:
            print(f"        tool_hint={hint}")

    # ---- 感知工具使用情况 ----
    perc = {"scene_inspect", "map_query", "route_plan", "height_probe"}
    used = [s for s in r.trace if (s.action or {}).get("tool") in perc]
    if used:
        print(f"\n【感知工具调用】命中 {len(used)} 次 -> "
              f"{sorted({(s.action or {}).get('tool') for s in used})}")
        for s in used:
            obs = (s.observation or "").replace("\n", " ")
            print(f"   - {s.subtask_id} {(s.action or {}).get('tool')}: {obs[:160]}")
    else:
        print("\n【感知工具调用】未调用（Planner 未把感知工具编入计划）")

    # ---- 轨迹 ----
    print("\n【执行轨迹】")
    for i, s in enumerate(r.trace, 1):
        tool = (s.action or {}).get("tool", "-")
        plat = s.platform.value if hasattr(s.platform, "value") else (s.platform or "-")
        tier = s.tier_used.value if hasattr(s.tier_used, "value") else (s.tier_used or "-")
        lat = getattr(s, "latency_ms", 0) or 0
        sim_d = getattr(s, "sim_duration_ms", 0) or 0
        obs = (s.observation or "")[:96].replace("\n", " ")
        print(f"  {i}. {s.subtask_id:10s} tool={tool:12s} plat={str(plat):6s} "
              f"tier={str(tier):5s} pre={str(s.precondition_ok):5s} "
              f"lat={lat:7.1f}ms sim={sim_d:7.0f}ms")
        print(f"       obs: {obs}")
        tm = getattr(s, "telemetry", None)
        if tm:
            print(f"       telemetry: {str(tm)[:150]}")

    # ---- 真实场景事实 ----
    print("\n【注入的真实场景事实（来自 3DGS 重建）】")
    hit = False
    for k in sorted(r.final_facts):
        if k.startswith("scene_"):
            print(f"   {k:24s} = {r.final_facts[k]}")
            hit = True
    if not hit:
        print("   （无 scene_* 事实）")

    # ---- 评测 ----
    if r.metrics is not None:
        print("\n【评测】")
        d = r.metrics.model_dump(mode="json") if hasattr(r.metrics, "model_dump") else dict(r.metrics)
        for k, v in d.items():
            print(f"   {k:24s} = {v}")
    print(f"\n   成功判据达成: {r.success_criteria_met}")

    print("\n完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
