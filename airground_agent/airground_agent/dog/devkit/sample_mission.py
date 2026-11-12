"""构造"已编译完成"的样例 DogMission，用于虚拟狗联调。

注意：这里的 `params.position` 已经是**机器狗地图系**坐标（模拟 compiler 的输出），
DogMissionRunner 直接透传给虚拟狗。坐标用 `default_transform`（纯轴映射）下的结果：
世界 [x, y, z] -> 地图 [x, -z, 0]。
"""
from __future__ import annotations

from typing import List

from ..protocol import DogCommand, DogMission


def make_sample_mission(goal: str = "虚拟狗联调：机器狗巡检开关柜区",
                        waypoints_world: List[List[float]] = None) -> DogMission:
    """构造一份含导航与动作的样例任务。

    waypoints_world: 世界系航点（默认 [(1,0.3,2),(2,0.3,4)]），会按 default 变换
    落到地图系 (x,-z,0)。若传了，则导航段用这些点；否则用内置两点。
    """
    if waypoints_world is None:
        waypoints_world = [[1.0, 0.3, 2.0], [2.0, 0.3, 4.0]]

    cmds: List[DogCommand] = [
        DogCommand(channel="sdk", op="sdk_mode", note="任务前置：获取控制权"),
        DogCommand(channel="sdk", op="stand_up", note="任务前置：起立"),
        DogCommand(channel="nav", op="loc_load_map", params={"map_id": "demo_map"},
                   note="任务前置：定位地图 demo_map"),
    ]
    for i, (x, y, z) in enumerate(waypoints_world, start=1):
        cmds.append(DogCommand(
            channel="nav", op="start_nav",
            params={"position": {"x": round(x, 4), "y": round(-z, 4), "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
            world_xyz=[x, y, z], subtask_id="t_dog",
            tool_hint="route_plan",
            note=f"前往目标点（航点 {i}/{len(waypoints_world)}）"))
    cmds.append(DogCommand(channel="wait", op="sleep", params={"seconds": 1.0},
                           note="到位后稳定采集窗口"))
    cmds.append(DogCommand(channel="sdk", op="stop_move", note="任务收尾"))

    return DogMission(
        goal=goal, map_id="demo_map", commands=cmds,
        waypoint_source="测试：手工指定地图坐标（default 轴映射）",
        subtask_cmd_count={"t_dog": sum(1 for c in cmds if c.subtask_id == "t_dog")},
    )
