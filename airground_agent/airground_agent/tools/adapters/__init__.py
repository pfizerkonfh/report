"""工具适配层（ToolAdapter）：把「语义契约」与「执行通道」解耦。

- `ToolSpec`（tools/base.py）回答：能不能做、做了会怎样（Precondition/Effect）。
- `ToolAdapter`（本包）回答：如何真正做出来（Mock / 仿真 / 录播 / ROS2 真机）。

换真机只需换 Adapter，Planner / Reasoner / Router / 评测逻辑零改动。
"""
from .base import (
    AdapterContext, AdapterMode, AdapterOutcome, AdapterStatus, CancellationToken,
    LegacyMockAdapter, SimClock, ToolAdapter,
)
from .factory import AdapterBundle, SimConfig, build_adapter_bundle, make_context
from .playback import PlaybackAdapter, default_playback_path
from .ros2 import Ros2Adapter
from .sim import Dynamics, SimProfile, SimulatedAdapter, SimWorld

__all__ = [
    "AdapterContext", "AdapterMode", "AdapterOutcome", "AdapterStatus",
    "CancellationToken", "LegacyMockAdapter", "SimClock", "ToolAdapter",
    "AdapterBundle", "SimConfig", "build_adapter_bundle", "make_context",
    "PlaybackAdapter", "default_playback_path",
    "Ros2Adapter",
    "Dynamics", "SimProfile", "SimulatedAdapter", "SimWorld",
]
