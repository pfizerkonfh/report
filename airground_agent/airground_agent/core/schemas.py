"""LLM 结构化输入输出 schema（与 LangChain with_structured_output 兼容）。"""
from __future__ import annotations

from typing import Dict, List

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .types import normalize_platform


class SubTaskSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    id: str
    description: str = ""
    depends_on: List[str] = Field(default_factory=list, validation_alias=AliasChoices("depends_on", "dependsOn"))
    tool_hint: str = Field(default="", validation_alias=AliasChoices("tool_hint", "tool"))
    success_criteria: str = Field(default="", validation_alias=AliasChoices("success_criteria", "successCriteria"))
    platform: str = Field(default="shared",
                          validation_alias=AliasChoices("platform", "platform_hint", "executor"))

    @field_validator("platform", mode="before")
    @classmethod
    def _norm_platform(cls, v):
        return normalize_platform(v).value


class PlanSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    root_goal: str = Field(default="", validation_alias=AliasChoices("root_goal", "rootGoal"))
    success_criteria: str = Field(default="", validation_alias=AliasChoices("success_criteria", "successCriteria"))
    subtasks: List[SubTaskSpec] = Field(default_factory=list, validation_alias=AliasChoices("subtasks", "subTasks"))


class ReasonerDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    thought: str = ""
    subtask_id: str = Field(default="", validation_alias=AliasChoices("subtask_id", "subtaskId"))
    tool: str = ""
    args: Dict = Field(default_factory=dict)
    aligned: bool = True
    rationale: str = ""
