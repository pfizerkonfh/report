#!/usr/bin/env python3
"""感知层演示：用真实 3DGS 重建产物驱动场景理解与路径规划。

运行：
    python -m examples.demo_perception

展示内容（全部基于真实重建数据，非 Mock）：
1. 加载 3DGS 场景（占据栅格 + 点云 + 重建质量）
2. 真实场景统计与重建质量
3. 占据查询 / 最近可通行点纠偏
4. 真实 A* 路径规划（真实里程）
5. 点云高度剖面（无人机净空 / 机器狗地面）
6. 通过工具接口调用（与决策系统同一套语义）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airground_agent.core.world_state import WorldState
from airground_agent.perception import (
    HeightProbeTool, MapQueryTool, RoutePlanTool, SceneInspectTool, load_scene,
)


def line(title: str) -> None:
    print("\n" + "=" * 68)
    print(f" {title}")
    print("=" * 68)


def main() -> int:
    scene = load_scene()
    if scene is None:
        print("未找到 3DGS 重建产物（nav_data.json）。")
        print("请确认 exp/viewer/showcase_assets/ 存在，或在 config/default.yaml 的")
        print("perception.data_dir 中指定目录。")
        return 1

    line("1. 场景加载")
    print(f"  数据来源：{scene.source}")
    print(f"  {scene.describe()}")

    line("2. 真实场景统计")
    for k, v in scene.stats().items():
        print(f"  {k:22s} = {v}")

    line("3. 重建质量（3DGS 指标）")
    for k, v in scene.quality().items():
        print(f"  {k:22s} = {v}")

    line("4. 占据查询（真实栅格）")
    ws = WorldState(facts=scene.to_facts())
    mq = MapQueryTool().bind(scene)
    for cell in (scene.default_start(), scene.default_goal(), (0, 0)):
        wx, wz = scene.cell_to_world(*cell)
        res = mq.run(ws, cell=list(cell))
        t = res.telemetry
        print(f"  格 {list(cell)} 世界({wx:6.2f},{wz:6.2f}) -> "
              f"{t['status']:4s} 可通行={str(t['traversable']):5s}")

    line("5. 真实 A* 路径规划")
    rp = RoutePlanTool().bind(scene)
    res = rp.run(ws)
    print(f"  {res.observation}")
    if res.success:
        t = res.telemetry
        print(f"  起点格 {t['start_cell']} -> 终点格 {t['goal_cell']}，"
              f"{t['steps']} 步，真实里程 {t['length_m']} m")
        print(f"  途经未知区 {t['unknown_cells']} 格（{t['unknown_ratio']:.0%}）")
        print(f"  路径前 3 个世界坐标：{t['world_path'][:3]}")

    line("6. 点云高度剖面（真实点云探测）")
    hp = HeightProbeTool().bind(scene)
    wx, wz = scene.cell_to_world(*scene.default_start())
    info = hp.run(ws, x=wx, z=wz, radius=0.8)
    print(f"  {info.observation}")
    print(f"  世界状态 clearance_m = {ws.get('clearance_m')}")

    line("7. 通过工具接口调用（与决策系统同一语义）")
    si = SceneInspectTool().bind(scene)
    print(f"  {si.run(ws).observation}")

    line("感知层就绪")
    print("  真实场景已注入 WorldState，前置校验可基于真实数据判定")
    print("  （例：Condition(key='scene_known_ratio', op='>=', value=0.5)）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
