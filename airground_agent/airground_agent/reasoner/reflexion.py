"""Reflexion 经验回流：任务失败后回放轨迹，沉淀 self-critique / lessons。

设计文档问题①：Reflexion 在任务失败后回放轨迹，写 self-critique/lessons 进 Experience
Buffer，下次规划时作为 context 注入 Planner，形成「失败 -> 经验 -> 改进」的持续改进闭环。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

from ..core.task_model import ExperienceEntry
from ..core.log import get_logger

logger = get_logger("reflexion")


class ExperienceBuffer:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.entries: List[ExperienceEntry] = []
        if path and os.path.exists(path):
            self._load()

    def add(self, entry: ExperienceEntry) -> None:
        self.entries.append(entry)
        logger.info("经验回流", layer=entry.failure_layer.value, lesson=entry.lesson[:40])
        if self.path:
            self._persist(entry)

    def lessons_for(self, goal: str) -> List[str]:
        """返回与当前目标相关的历史经验（简单关键词匹配）。"""
        g = (goal or "").lower()
        out = []
        for e in self.entries:
            if not e.goal_pattern or e.goal_pattern.lower() in g or g in e.goal_pattern.lower():
                out.append(e.lesson)
        return out

    def all_lessons(self) -> List[str]:
        return [e.lesson for e in self.entries]

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.entries.append(ExperienceEntry.model_validate_json(line))
        except Exception as e:  # pragma: no cover
            logger.warning("经验文件加载失败", error=str(e))

    def _persist(self, entry: ExperienceEntry) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        except Exception as e:  # pragma: no cover
            logger.warning("经验持久化失败", error=str(e))
