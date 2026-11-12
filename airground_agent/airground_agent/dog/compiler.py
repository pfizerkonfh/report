"""任务规划 -> 机器狗指令序列的编译器。

这一层回答一个具体问题：**决策平台跑完一轮规划后，机器狗到底该做什么？**

输入是决策平台的产物：
- `Plan`：子任务 DAG，每个 SubTask 带 `platform`（无人机 / 机器狗 / 协同算力）与 `tool_hint`；
- `TraceStep` 列表：执行轨迹，其中 `route_plan` 的 telemetry 里带着**真实 A* 航点**
  （在 3DGS 重建的占据栅格上算出来的，不是编造的）；
- 可选 `SceneModel`：trace 缺失时用来现算一条真实路径兜底。

输出是 `DogMission`：一串机器狗原生指令（导航 `start_nav` / 动作 `stand_up` / 等待 / 备注），
每条都带 `subtask_id` 溯源，可直接导出 JSON、也可交给 runner 真发。

三条不妥协的原则
----------------
1. **只编译机器狗的子任务**。无人机与云端算力的子任务不进 mission，
   并在 warnings 里说明被跳过了 —— 静默丢弃会让人以为"全下发了"。
2. **航点必须是真实计算的**。优先用 trace 里 route_plan 的真实 A* 结果，
   其次是现场用 SceneModel 现算，绝不凭空生成坐标。
3. **映射不了的就报错**，不猜。工具到动作的映射表 `TOOL_ACTION` 是显式的，
   未登记的 tool_hint 写入 warnings 并产出 `meta.unmapped` 占位指令。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..core.task_model import Plan, TraceStep
from ..core.types import Platform
from .frames import FrameTransform, default_transform
from .protocol import CHANNEL_META, CHANNEL_NAV, CHANNEL_SDK, CHANNEL_WAIT, DogCommand, DogMission

#: 工具 -> 机器狗侧动作类别（编译器据此展开指令）
#:   navigate   导航到目标（用真实 A* 航点）
#:   collect    到位后采集（保持姿态 + 停留窗口）
#:   return_home 返航充电桩
#:   note_only  决策侧计算项，不下发真机
TOOL_ACTION: Dict[str, str] = {
    "route_plan": "navigate",
    "camera": "collect",
    "charge": "return_home",
    "map_query": "note_only",
    "scene_inspect": "note_only",
    "height_probe": "note_only",
    "vla": "note_only",
    "weather": "note_only",
    "edit": "note_only",
    "anonymize": "note_only",
    "cloud_llm": "note_only",
}

#: 机器狗不支持承载的工具（写在这里是为了给出更明确的告警文案）
DOG_UNSUITABLE = {"weather", "edit", "anonymize", "cloud_llm"}


class CompileConfig(BaseModel):
    """编译选项（现场可调，全部有安全默认值）。"""

    #: 狗端地图名（留空 = 用狗当前地图，不下发 loc_load_map）
    map_id: str = ""
    #: 狗端已录制路径 ID（给了且 nav_mode=auto 则走多点导航）
    path_id: str = ""
    #: 导航模式：waypoints 逐点下发 / multi 多点导航（需 path_id）/ auto 自动
    nav_mode: str = "auto"
    #: 坐标系变换（现场标定后覆盖）
    frame: FrameTransform = Field(default_factory=default_transform)
    #: 航点抽稀间距（米）。机器狗自身会做局部规划，不需要 0.2m 级别的稠密点。
    waypoint_stride_m: float = 1.0
    #: 单段导航航点上限（防止把长路径刷成上百条指令）
    max_waypoints: int = 40
    #: 朝向模式：along_path 沿前进方向 / fixed 固定朝向 / none 不指定
    heading_mode: str = "along_path"
    fixed_yaw_deg: float = 0.0
    #: 单条导航指令超时（秒）
    nav_timeout_s: float = 120.0
    #: 任务前导：切 SDK 模式 + 站立（+ 加载地图）
    preamble_mode: bool = True
    preamble_stand: bool = True
    #: 收尾动作：stop（停止运动）/ sit（匍匐）/ none
    finish: str = "stop"
    #: 采集窗口（秒）—— collect 类动作到位后的停留时间
    collect_seconds: float = 3.0
    #: 是否输出 note_only 的 meta 记录行（便于审计"决策侧算了什么"）
    include_meta: bool = True
    #: 兜底：机器狗子任务全是"决策侧计算"（如 vla）导致没有任何导航段时，
    #: 用真实场景现算一条"前往默认作业区"的导航段。**默认关闭** —— 打开后
    #: 会在 warnings 里标注该段是兜底生成的，非规划显式要求。
    infer_approach: bool = False
    #: 自定义工具映射覆盖（tool_hint -> action）
    tool_action_overrides: Dict[str, str] = Field(default_factory=dict)


def resample_path(path: Sequence[Sequence[float]], stride_m: float,
                  max_points: int) -> List[List[float]]:
    """按距离抽稀航点，保留起终点；超过 max_points 时再均匀降采样。

    栅格路径每点间隔约 0.2m，直接全量下发既无意义也会拖慢执行。
    """
    if not path:
        return []
    stride = max(stride_m, 1e-6)
    out: List[List[float]] = [[float(p[0]), float(p[1]), float(p[2])] for p in path[:1]]
    acc = 0.0
    prev = path[0]
    for p in path[1:]:
        d = math.hypot(float(p[0]) - float(prev[0]), float(p[2]) - float(prev[2]))
        acc += d
        if acc >= stride:
            out.append([float(p[0]), float(p[1]), float(p[2])])
            acc = 0.0
        prev = p
    last = [float(path[-1][0]), float(path[-1][1]), float(path[-1][2])]
    if not out or out[-1] != last:
        out.append(last)
    if max_points > 1 and len(out) > max_points:
        idx = [round(i * (len(out) - 1) / (max_points - 1)) for i in range(max_points)]
        seen, kept = set(), []
        for i in idx:
            if i not in seen:
                seen.add(i)
                kept.append(out[i])
        out = kept
    return out


def _trace_telemetry_for(trace: Sequence[TraceStep], subtask_id: str,
                         tool: str) -> Optional[Dict[str, Any]]:
    """从轨迹里找该子任务对应工具调用的 telemetry（真实计算结果）。"""
    hit: Optional[Dict[str, Any]] = None
    for st in trace:
        if st.subtask_id != subtask_id:
            continue
        tool_name = (st.action or {}).get("tool", "")
        if tool and tool_name and tool_name != tool:
            continue
        if st.telemetry:
            hit = st.telemetry          # 取最后一条（重规划后以最新为准）
    return hit


class PlanCompiler:
    """把 Plan（+ 真实航点）编译成机器狗指令序列。"""

    def __init__(self, cfg: Optional[CompileConfig] = None, scene: Any = None) -> None:
        self.cfg = cfg or CompileConfig()
        self.scene = scene

    # ------------------------------------------------------------------
    def compile(self, plan: Plan, trace: Optional[Sequence[TraceStep]] = None,
                goal: str = "", run_id: str = "") -> DogMission:
        cfg = self.cfg
        mission = DogMission(goal=goal or plan.root_goal, run_id=run_id,
                             map_id=cfg.map_id, path_id=cfg.path_id)
        trace = list(trace or [])
        seq = 0

        def _add(cmd: DogCommand) -> DogCommand:
            nonlocal seq
            seq += 1
            cmd.id = f"c{seq:03d}"
            err = cmd.validate_against_protocol()
            if err:
                mission.warnings.append(f"[{cmd.id}] {err}")
            mission.commands.append(cmd)
            if cmd.subtask_id:
                mission.subtask_cmd_count[cmd.subtask_id] = \
                    mission.subtask_cmd_count.get(cmd.subtask_id, 0) + 1
            return cmd

        # ---- 前导：进入 SDK 控制模式 -> 站立 -> 加载地图 ----
        if cfg.preamble_mode:
            _add(DogCommand(channel=CHANNEL_SDK, op="sdk_mode", timeout_s=10.0,
                            expect="接管控制权（SDK 模式）", note="任务前置：获取控制权"))
        if cfg.preamble_stand:
            _add(DogCommand(channel=CHANNEL_SDK, op="stand_up", timeout_s=15.0,
                            expect="站立待命", note="任务前置：起立"))
        if cfg.map_id and cfg.nav_mode != "waypoints":
            _add(DogCommand(channel=CHANNEL_NAV, op="loc_load_map",
                            params={"map_id": cfg.map_id}, timeout_s=30.0,
                            expect=f"加载定位地图 {cfg.map_id}", note="任务前置：定位地图"))

        # ---- 主体：按拓扑序遍历机器狗子任务 ----
        order = plan.topo_order() or list(plan.subtasks.keys())
        last_pose_world: Optional[List[float]] = None
        dog_subtasks = [sid for sid in order
                        if plan.subtasks.get(sid) is not None
                        and plan.subtasks[sid].platform == Platform.DOG]
        if not dog_subtasks:
            mission.warnings.append(
                "规划中没有 platform=DOG 的子任务，机器狗 mission 为空（本次任务可能全部分配给无人机/云端）")

        for sid in order:
            st = plan.subtasks.get(sid)
            if st is None or st.platform != Platform.DOG:
                continue
            tool = (st.tool_hint or "").strip().lower()
            action = cfg.tool_action_overrides.get(tool) or TOOL_ACTION.get(tool)
            if action is None:
                mission.warnings.append(
                    f"子任务 {sid}（tool_hint={st.tool_hint or '空'}）未登记机器狗动作映射，"
                    f"已产出占位指令，未下发真机动作")
                if cfg.include_meta:
                    _add(DogCommand(channel=CHANNEL_META, op="unmapped", subtask_id=sid,
                                    tool_hint=tool, note=f"未映射的子任务：{st.description}",
                                    expect="需要人工补充映射"))
                continue

            if action == "note_only":
                if cfg.include_meta:
                    _add(DogCommand(channel=CHANNEL_META, op="decision_side", subtask_id=sid,
                                    tool_hint=tool,
                                    note=f"该步骤由决策侧完成，无需下发机器狗：{st.description}",
                                    expect="无真机动作"))
                continue

            if action == "return_home":
                _add(DogCommand(channel=CHANNEL_NAV, op="start_nav_return_home",
                                subtask_id=sid, tool_hint=tool, timeout_s=cfg.nav_timeout_s,
                                note=st.description, expect="返回充电桩"))
                continue

            if action == "collect":
                pose = last_pose_world or self._fallback_pose()
                if pose is None:
                    mission.warnings.append(
                        f"子任务 {sid} 需要采集位姿，但上下文中没有可用坐标（前序导航缺失且场景未加载）")
                    continue
                mx, my, mz = cfg.frame.world_to_map(*pose)
                _add(DogCommand(channel=CHANNEL_NAV, op="start_nav", subtask_id=sid,
                                tool_hint=tool, timeout_s=cfg.nav_timeout_s,
                                params={"position": {"x": mx, "y": my, "z": mz},
                                        "orientation": self._orientation(pose, None)},
                                world_xyz=list(pose),
                                note=st.description, expect="导航至采集点"))
                _add(DogCommand(channel=CHANNEL_WAIT, op="sleep", subtask_id=sid,
                                tool_hint=tool, timeout_s=cfg.collect_seconds + 5.0,
                                params={"seconds": cfg.collect_seconds},
                                note="到位后稳定采集窗口", expect="完成近距采集"))
                continue

            if action == "navigate":
                wps, source = self._resolve_waypoints(plan, trace, sid, tool)
                if not wps:
                    mission.warnings.append(f"子任务 {sid} 没有可用航点（真实路径缺失且无法现算）")
                    continue
                mission.waypoint_source = source
                self._emit_nav(_add, wps, sid, tool, st.description)
                last_pose_world = list(wps[-1])
                continue

        # ---- 兜底：一个导航段都没有时（例如机器狗子任务只写了 vla 端侧推理）----
        has_nav = any(c.channel == CHANNEL_NAV for c in mission.commands)
        if (not has_nav and dog_subtasks and cfg.infer_approach):
            wps, source = self._resolve_waypoints(plan, trace, dog_subtasks[0], "route_plan")
            if wps:
                mission.warnings.append(
                    f"规划的机器狗子任务未显式要求导航，已按兜底策略生成「前往默认作业区」"
                    f"导航段（{source}），请确认这是现场需要的动作")
                mission.waypoint_source = source
                self._emit_nav(_add, wps, dog_subtasks[0], "route_plan", "兜底：前往默认作业区")
            else:
                mission.warnings.append(
                    "规划的机器狗子任务未显式要求导航，且场景不可用无法兜底现算，"
                    "mission 中不含导航段")

        # ---- 收尾 ----
        if cfg.finish == "stop":
            _add(DogCommand(channel=CHANNEL_SDK, op="stop_move", timeout_s=5.0,
                            expect="停止运动", note="任务收尾"))
        elif cfg.finish == "sit":
            _add(DogCommand(channel=CHANNEL_SDK, op="sit_down", timeout_s=10.0,
                            expect="匍匐待命", note="任务收尾"))

        problems = mission.validate()
        mission.warnings.extend(problems)
        return mission

    # ------------------------------------------------------------------
    def _emit_nav(self, add, wps: List[List[float]], sid: str, tool: str, desc: str) -> None:
        """把航点展开为导航指令（逐点下发或多点导航）。"""
        cfg = self.cfg
        use_multi = (cfg.nav_mode == "multi" or
                     (cfg.nav_mode == "auto" and cfg.path_id and cfg.map_id))
        if use_multi:
            add(DogCommand(channel=CHANNEL_NAV, op="start_multi_nav", subtask_id=sid,
                           tool_hint=tool, timeout_s=cfg.nav_timeout_s * max(1, len(wps) // 5),
                           params={"map_id": cfg.map_id, "path_id": cfg.path_id},
                           note=f"{desc}（多点路径 {cfg.path_id}，{len(wps)} 个航点）",
                           expect="沿录制路径多点导航"))
            return
        for i, p in enumerate(wps):
            nxt = wps[i + 1] if i + 1 < len(wps) else None
            mx, my, mz = cfg.frame.world_to_map(p[0], p[1], p[2])
            params: Dict[str, Any] = {"position": {"x": mx, "y": my, "z": mz}}
            if cfg.heading_mode != "none":
                params["orientation"] = self._orientation(p, nxt)
            add(DogCommand(channel=CHANNEL_NAV, op="start_nav", subtask_id=sid,
                           tool_hint=tool, timeout_s=cfg.nav_timeout_s, params=params,
                           world_xyz=list(p), note=f"{desc}（航点 {i + 1}/{len(wps)}）",
                           expect=f"导航至 ({mx:.2f}, {my:.2f})"))

    def _orientation(self, cur: Sequence[float], nxt: Optional[Sequence[float]]) -> Dict[str, float]:
        cfg = self.cfg
        if cfg.heading_mode == "fixed":
            return cfg.frame.yaw_to_quaternion(math.radians(cfg.fixed_yaw_deg))
        if nxt is None:
            # 终点：保持上一段朝向（不知道就朝 +X）
            return cfg.frame.yaw_to_quaternion(math.radians(cfg.fixed_yaw_deg))
        return cfg.frame.orientation_towards(cur, nxt)

    # ------------------------------------------------------------------
    def _resolve_waypoints(self, plan: Plan, trace: Sequence[TraceStep],
                           sid: str, tool: str) -> tuple:
        """取得真实航点，返回 (航点列表, 来源说明)。

        优先级：trace 中真实 A* 结果 > 现场用 SceneModel 现算 > 无。
        """
        tele = _trace_telemetry_for(trace, sid, tool or "route_plan")
        if tele:
            raw = tele.get("world_path")
            if raw:
                wps = resample_path(raw, self.cfg.waypoint_stride_m, self.cfg.max_waypoints)
                if wps:
                    return wps, "trace.route_plan 真实 A* 航点"
        # 兜底：现场用真实占据栅格现算一条
        if self.scene is not None:
            try:
                res = self.scene.plan_world(
                    self.scene.cell_to_world3(*self.scene.default_start()),
                    self.scene.cell_to_world3(*self.scene.default_goal()),
                    allow_unknown=True, snap=True)
                if res.get("success") and res.get("world_path"):
                    wps = resample_path(res["world_path"], self.cfg.waypoint_stride_m,
                                        self.cfg.max_waypoints)
                    if wps:
                        return wps, "scene.plan_world 现场现算（真实占据栅格 A*）"
            except Exception:  # 感知层不可用不应中断编译
                pass
        return [], ""

    def _fallback_pose(self) -> Optional[List[float]]:
        """没有上下文时，取场景默认目标点作为采集位姿。"""
        if self.scene is None:
            return None
        try:
            return list(self.scene.cell_to_world3(*self.scene.default_goal()))
        except Exception:
            return None


def compile_plan(plan: Plan, trace: Optional[Sequence[TraceStep]] = None,
                 cfg: Optional[CompileConfig] = None, scene: Any = None,
                 goal: str = "", run_id: str = "") -> DogMission:
    """便捷入口：Plan -> DogMission。"""
    return PlanCompiler(cfg=cfg, scene=scene).compile(plan, trace, goal=goal, run_id=run_id)
