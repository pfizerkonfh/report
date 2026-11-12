"""机器狗（ZSIBOT 轮式四足）协议数据模型 —— 决策平台与真机之间的"共同语言"。

为什么需要这一层
----------------
决策平台产出的是**语义层**的东西：`Plan`（子任务 DAG）、`route_plan` 的真实 A* 航点、
`camera` / `charge` 这类工具调用。机器狗只认两样东西：

1. **导航通道**：WebSocket `ws://<ip>:10010`，一帧一个 `req_func`
   （见厂家 demo `nav-05.py`）——点位、地图、多点路径、暂停/继续/返航。
2. **动作通道**：`zsibot_control --daemon` 的 stdin 文本行（`main_sdk.cpp`）——
   模式切换、站立、匍匐、平移、急停。

本模块把前者**无损编译**为后者：每条 `DogCommand` 都携带 `subtask_id` 溯源，
可以序列化成真机真正吃的两类字节流，也可以只导出 JSON 供人工/离线校验。

协议要点（严格对齐厂家 demo，勿自行发明）
-----------------------------------------
- 导航帧：`{"head":{"type":"app_req","time_stamp":ms,"source":"app","frame_count":n},
  "data":{"req_func":{...}}}`，应答 `app_resp` 按 `frame_count` 配对。
- `req_func` 形态：`{"loc_load_map": map_id}` / `{"start_nav": {"position":{x,y,z},
  "orientation":{x,y,z,w}}}` / `{"start_multi_nav": [map_id, path_id]}` /
  `{"stop_nav": null}` / `{"pause_nav": null}` / `{"continue_nav": null}` /
  `{"start_nav_return_home": null}` / `{"get_nav_status": null}` 等。
- 动作通道：一行一个命令字符串，回一行 `{"ok":true,"command":...}`。
- 轮式机器狗不支持跳跃/后空翻/双腿站立等动作 —— `UNSUPPORTED_SDK_OPS` 明确列出，
  编译期拒绝，不在真机上试探。
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

#: 通道类型
CHANNEL_NAV = "nav"      # 导航 WebSocket
CHANNEL_SDK = "sdk"      # zsibot_control daemon
CHANNEL_WAIT = "wait"    # 本地等待（采集窗口 / 稳定时间）
CHANNEL_META = "meta"    # 仅记录，不下发（决策侧计算项）

#: 导航通道的 req_func 白名单（与 nav-05.py 一致）
NAV_OPS = {
    "loc_load_map", "get_all_pgm_map", "start_nav", "start_multi_nav",
    "stop_nav", "pause_nav", "continue_nav", "start_nav_return_home",
    "get_nav_status", "get_loc_status", "get_mapping_status",
}

#: SDK 动作通道支持的命令（轮式机器狗可用部分）
SDK_OPS = {
    "sdk_mode", "remote_mode", "roamerx_mode",
    "forward", "backward", "turn_left", "turn_right",
    "move_left", "move_right", "stop_move", "emergency_stop",
    "sit_down", "stand_up", "balance_stand", "unload_squat", "get_status",
}

#: 轮式机器狗**不支持**的动作（明确拒绝，不静默丢弃）
UNSUPPORTED_SDK_OPS = {
    "jump", "forward_jump", "greet", "back_flip", "two_leg_stand",
    "enter_lab", "exit_lab", "enter_entertainment", "exit_entertainment",
}


class DogCommand(BaseModel):
    """一条机器狗可执行指令（决策平台视角的最小执行单元）。

    `channel` 决定它走哪条物理通道；`op` 是该通道的原生操作名；
    `params` 是已经**换算到机器狗坐标系**的参数 —— 到这里不做任何语义翻译，
    翻译工作在 compiler 里完成，本模块只负责承载与序列化。
    """

    id: str = Field(default_factory=lambda: f"cmd-{uuid.uuid4().hex[:8]}")
    channel: Literal["nav", "sdk", "wait", "meta"] = CHANNEL_META
    op: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    #: 溯源：来自决策平台的哪个子任务（可空 = 任务级前导/收尾指令）
    subtask_id: Optional[str] = None
    tool_hint: str = ""
    note: str = ""
    #: 人类可读预期效果（日志/前端展示用）
    expect: str = ""
    #: 该指令最长等待时间（秒）；导航类通常远大于动作类
    timeout_s: float = 60.0
    #: 是否等待完成再下发下一条（导航必须 True，模式切换可 False）
    blocking: bool = True
    #: 世界系原始坐标（溯源/调试用，真机不用它）
    world_xyz: Optional[List[float]] = None

    # ---------------- 序列化：导航通道 ----------------
    def to_req_func(self) -> Dict[str, Any]:
        """把本指令翻译为 nav-05 协议的 `req_func` 载荷。

        只有 `channel == nav` 的指令才有意义；其余返回空字典由调用方跳过。
        """
        if self.channel != CHANNEL_NAV:
            return {}
        op = self.op
        if op == "start_nav":
            req: Dict[str, Any] = {"position": dict(self.params.get("position", {}))}
            if self.params.get("orientation"):
                req["orientation"] = dict(self.params["orientation"])
            return {"start_nav": req}
        if op == "start_multi_nav":
            return {"start_multi_nav": [self.params.get("map_id", ""),
                                        self.params.get("path_id", "")]}
        if op == "loc_load_map":
            return {"loc_load_map": self.params.get("map_id", "")}
        if op in ("get_all_pgm_map", "stop_nav", "pause_nav", "continue_nav",
                  "start_nav_return_home", "get_nav_status", "get_loc_status",
                  "get_mapping_status"):
            return {op: None}
        raise ValueError(f"未知导航指令: {op}")

    def to_ws_frame(self, frame_count: int) -> Dict[str, Any]:
        """包装成完整的 app_req 帧（含 head）。"""
        return {
            "head": {
                "type": "app_req",
                "time_stamp": int(time.time() * 1000),
                "source": "app",
                "frame_count": frame_count,
            },
            "data": {"req_func": self.to_req_func()},
        }

    # ---------------- 序列化：动作通道 ----------------
    def to_daemon_line(self) -> Optional[str]:
        """把本指令翻译为 `zsibot_control --daemon` 的 stdin 文本行。"""
        if self.channel != CHANNEL_SDK:
            return None
        return self.op

    # ---------------- 展示 ----------------
    def describe(self) -> str:
        if self.channel == CHANNEL_NAV:
            p = self.params.get("position", {})
            if p:
                return (f"nav.{self.op} -> ({p.get('x', 0):.2f}, {p.get('y', 0):.2f}, "
                        f"{p.get('z', 0):.2f})")
            return f"nav.{self.op} {self.params or ''}".strip()
        if self.channel == CHANNEL_SDK:
            return f"sdk.{self.op}"
        if self.channel == CHANNEL_WAIT:
            return f"wait {self.params.get('seconds', 0)}s"
        return f"meta.{self.op or 'note'}: {self.note}"

    def validate_against_protocol(self) -> Optional[str]:
        """编译期自检：协议不允许的指令必须立刻暴露，而不是到了真机才失败。"""
        if self.channel == CHANNEL_NAV and self.op not in NAV_OPS:
            return f"导航通道不支持的指令: {self.op}"
        if self.channel == CHANNEL_SDK:
            if self.op in UNSUPPORTED_SDK_OPS:
                return f"轮式机器狗不支持的动作: {self.op}"
            if self.op not in SDK_OPS:
                return f"动作通道不支持的指令: {self.op}"
        return None


class DogMission(BaseModel):
    """一次下发给机器狗的完整任务 —— 决策平台输出物在真机侧的最终形态。"""

    mission_id: str = Field(default_factory=lambda: f"mission-{uuid.uuid4().hex[:8]}")
    goal: str = ""
    run_id: str = ""
    #: 机器狗本体地图（狗端 pgm 地图名）。留空表示用狗当前已加载的地图。
    map_id: str = ""
    #: 狗端已录制的路径 ID；给了就走多点导航，否则走逐点导航
    path_id: str = ""
    commands: List[DogCommand] = Field(default_factory=list)
    #: 航点来源说明（真实 A* / 默认兜底 / 人工指定），用于判断数据可信度
    waypoint_source: str = ""
    #: 编译期告警（不可映射的子任务、缺失坐标等）—— 不静默丢弃问题
    warnings: List[str] = Field(default_factory=list)
    #: 各子任务 -> 指令数（用于前端/报告展示"规划如何落地"）
    subtask_cmd_count: Dict[str, int] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    # ---------------- 导出 ----------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "run_id": self.run_id,
            "map_id": self.map_id,
            "path_id": self.path_id,
            "waypoint_source": self.waypoint_source,
            "warnings": list(self.warnings),
            "subtask_cmd_count": dict(self.subtask_cmd_count),
            "created_at": self.created_at,
            "commands": [c.model_dump(mode="json") for c in self.commands],
        }

    def to_json(self, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def nav_frames(self, start_frame_count: int = 1) -> List[Dict[str, Any]]:
        """导出可直接喂给导航 WebSocket 的帧序列（按 frame_count 递增编号）。"""
        frames, n = [], start_frame_count
        for c in self.commands:
            if c.channel != CHANNEL_NAV:
                continue
            frames.append(c.to_ws_frame(n))
            n += 1
        return frames

    def daemon_lines(self) -> List[str]:
        """导出可直接写入 `zsibot_control --daemon` stdin 的命令序列。"""
        out = []
        for c in self.commands:
            line = c.to_daemon_line()
            if line:
                out.append(line)
        return out

    def summary(self) -> Dict[str, Any]:
        by_channel: Dict[str, int] = {}
        for c in self.commands:
            by_channel[c.channel] = by_channel.get(c.channel, 0) + 1
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "commands": len(self.commands),
            "by_channel": by_channel,
            "waypoints": sum(1 for c in self.commands
                             if c.channel == CHANNEL_NAV and c.op == "start_nav"),
            "warnings": len(self.warnings),
            "map_id": self.map_id,
            "path_id": self.path_id,
        }

    def validate(self) -> List[str]:
        """下发前全量自检。返回问题列表（空 = 可下发）。"""
        problems: List[str] = []
        for c in self.commands:
            err = c.validate_against_protocol()
            if err:
                problems.append(f"[{c.id}] {err}")
        if self.path_id and not self.map_id:
            problems.append("指定了 path_id 但未指定 map_id（多点导航需要二者成对）")
        return problems
