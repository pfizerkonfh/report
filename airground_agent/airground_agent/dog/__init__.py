"""机器狗接入层（ZSIBOT 轮式四足）—— 决策平台与真机之间的桥。

数据流
------
```
决策平台 RunResult(Plan + Trace)
        │
        │  compiler：只取 platform=DOG 的子任务，取出真实 A* 航点
        ▼
   DogMission（机器狗指令序列，每条带 subtask_id 溯源）
        │
        │  protocol：序列化
        ├─► nav 通道：WebSocket req_func 帧（ws://<ip>:10010）
        └─► sdk 通道：zsibot_control --daemon 的 stdin 命令文本
        │
        │  runner：按序下发、轮询导航状态、急停、留痕
        ▼
   MissionReport（每条指令的响应/耗时/末态）
```

三条通道形态，按需取用
----------------------
1. **离线编译**：`compile_plan(plan, trace, cfg, scene)` -> mission JSON，
   不连狗也能先校验"规划到底能翻译成什么"。
2. **联调执行**：`DogMissionRunner(transport).run(mission)`，`dry_run=True`
   时只记录不下发，用来验证指令序列与时序。
3. **真机在线**：`sim.mode = zsibot`，决策平台执行到机器狗子任务时，
   `ZsibotDogAdapter` 直接把工具调用下发到真狗。
"""
from .frames import FrameTransform, default_transform
from .protocol import DogCommand, DogMission
from .compiler import CompileConfig, PlanCompiler, compile_plan, resample_path, TOOL_ACTION
from .transport import (
    DogTransport, NavClient, NavResponse, DaemonResponse, SdkDaemonClient, build_transport,
)
from .runner import CommandResult, DogMissionRunner, MissionReport
from .adapter import ZsibotDogAdapter, build_dog_adapter

__all__ = [
    "FrameTransform", "default_transform",
    "DogCommand", "DogMission",
    "CompileConfig", "PlanCompiler", "compile_plan", "resample_path", "TOOL_ACTION",
    "DogTransport", "NavClient", "NavResponse", "DaemonResponse", "SdkDaemonClient",
    "build_transport",
    "CommandResult", "DogMissionRunner", "MissionReport",
    "ZsibotDogAdapter", "build_dog_adapter",
]
