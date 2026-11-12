"""工具注册表。"""
from __future__ import annotations

from typing import Dict, List, Optional

from .base import ToolSpec
from .mock_tools import (
    AnonymizeTool, CameraTool, ChargeTool, CloudLLMTool, EditTool, VLATool, WeatherTool,
)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> List[str]:
        return list(self._tools.keys())

    def match_by_hint(self, hint: str) -> Optional[ToolSpec]:
        return self._tools.get(hint)

    @classmethod
    def default(cls, vla_settings=None, extra_tools=None) -> "ToolRegistry":
        """默认工具集。extra_tools 用于注入真实感知工具（3DGS 场景）。"""
        reg = cls()
        for t in (WeatherTool(), CameraTool(), EditTool(), AnonymizeTool(),
                  VLATool(vla_settings), ChargeTool(), CloudLLMTool()):
            reg.register(t)
        for t in (extra_tools or []):
            reg.register(t)
        return reg
