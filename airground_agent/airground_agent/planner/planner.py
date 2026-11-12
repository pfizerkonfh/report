"""规划层 Planner（LangGraph）。

职责（设计文档问题①）：
- 仅做「目标分解 + 排程」，输出带 success_criteria 的 DAG，**不触碰工具参数**；
- 对分解结果做结构化校验（无环、工具可达、依赖闭合）；
- 校验失败时携带 errors 触发重规划循环，直至成功或达到 max_retries。

Planner 不负责执行，也不维护世界状态写回；它产出 Plan 后由 Orchestrator 登记到
GoalRegistry。Reflexion 沉淀的经验（ExperienceEntry）作为 context 注入，形成持续改进。
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.llm import LLMBackend, get_backend
from ..core.schemas import PlanSchema
from ..core.task_model import Plan, SubTask
from ..core.types import Platform, normalize_platform
from ..core.world_state import WorldState
from ..tools.registry import ToolRegistry
from ..core.log import get_logger

logger = get_logger("planner")

PLANNER_SYSTEM = """[ROLE:PLANNER]
你是「空地协同」巡检系统的规划器：把长程任务分解为可调度子任务（DAG），并决定每个子任务
交给哪个物理平台执行。
要求：
1. 仅输出一个 JSON 对象，结构为 {"root_goal": str, "success_criteria": str, "subtasks": [ {id, description, depends_on:[...], tool_hint, success_criteria, platform}, ... ]}。
2. 每个子任务必须有唯一 id、描述、依赖 depends_on(数组)、建议工具 tool_hint(必须来自工具集)、成功判据 success_criteria。
3. platform（物理执行平台）只能取以下三值之一：
   - "uav"    = 无人机（空中平台）：高空视野、航拍、大范围扫描、绝缘子/线路巡检；
   - "dog"    = 机器狗（地面四足平台）：地面近距离、表计读数、开关柜、狭窄区域、局部细节；
   - "shared" = 共享/协同算力：天气查询、剪辑、脱敏、云端分析等无专属物理平台的计算任务。
   空地协同的核心：凡涉及实地采集/巡检的子任务，必须显式判给 uav 或 dog，不要一律用 shared。
4. 依赖必须形成有向无环图（DAG）；工具名必须来自已注册工具集。
5. 成功判据要可观测（如"生成 footage""输出 anonymized"）。
6. 控制规模：4~6 个子任务即可；每个字段力求简短（中文≤12字），不要编造工具参数。
仅做分解与排程，不要编造工具参数。
"""


class PlannerState(TypedDict):
    goal: str
    ws: Dict[str, Any]
    registry_names: List[str]
    attempt: int
    max_retries: int
    errors: List[str]
    experience: List[str]
    plan_schema: Optional[PlanSchema]
    plan: Optional[Plan]
    status: str   # running | done | failed


class Planner:
    def __init__(self, registry: ToolRegistry, backend: Optional[LLMBackend] = None,
                 max_retries: int = 3) -> None:
        self.registry = registry
        self.backend = backend or get_backend()
        self.max_retries = max_retries
        self._graph = self._build()

    def _build(self) -> Any:
        def decompose(state: PlannerState) -> Dict:
            tools = ", ".join(state["registry_names"]) or "（无）"
            user_parts = [
                f"用户目标：{state['goal']}",
                f"世界状态摘要：{state['ws']}",
                f"已注册工具集（tool_hint 必须严格取自此处，禁止自造）：{tools}",
            ]
            if state.get("errors"):
                user_parts.append(
                    "上次规划未通过校验，请修正：\n- " + "\n- ".join(state["errors"])
                )
            if state.get("experience"):
                user_parts.append(
                    "历史经验（务必避免重犯）：\n- " + "\n- ".join(state["experience"])
                )
            user_msg = "\n".join(user_parts)
            schema = self.backend.generate(
                PLANNER_SYSTEM,
                user_msg,
                PlanSchema,
                context={
                    "goal": state["goal"],
                    "ws": state["ws"],
                    "tools": state["registry_names"],
                    "errors": state["errors"],
                    "attempt": state["attempt"],
                    "experience": state.get("experience", []),
                },
            )
            return {"plan_schema": schema}

        def validate(state: PlannerState) -> Dict:
            schema = state["plan_schema"]
            errors: List[str] = []
            plan: Optional[Plan] = None
            if not schema or not schema.subtasks:
                errors.append("分解结果为空")
            else:
                plan = self._to_plan(schema)
                if not plan.validate_acyclic():
                    errors.append("子任务依赖存在环")
                for st in plan.subtasks.values():
                    if st.tool_hint and not self.registry.has(st.tool_hint):
                        errors.append(f"子任务 {st.id} 的工具不存在: {st.tool_hint}")
            if errors:
                logger.warning("规划校验未通过", errors=errors, attempt=state["attempt"])
                if state["attempt"] >= state["max_retries"]:
                    return {"errors": errors, "status": "failed"}
                return {"errors": errors, "attempt": state["attempt"] + 1, "status": "running"}
            logger.info("规划通过", subtasks=list(plan.subtasks.keys()))
            return {"plan": plan, "status": "done"}

        def route(state: PlannerState) -> str:
            if state["status"] == "done":
                return "done"
            if state["status"] == "failed":
                return "fail"
            return "retry"

        g = StateGraph(PlannerState)
        g.add_node("decompose", decompose)
        g.add_node("validate", validate)
        g.add_edge(START, "decompose")
        g.add_edge("decompose", "validate")
        g.add_conditional_edges("validate", route, {"retry": "decompose", "done": END, "fail": END})
        return g.compile()

    def _resolve_platform(self, spec) -> Platform:
        """确定单个子任务的物理执行平台。

        优先级：显式 platform > 描述/id 关键词推断 > 工具默认平台提示 > SHARED。
        """
        explicit = normalize_platform(spec.platform)
        if explicit != Platform.SHARED:
            return explicit
        inferred = normalize_platform(f"{spec.id or ''} {spec.description or ''}")
        if inferred != Platform.SHARED:
            return inferred
        tool = self.registry.get(spec.tool_hint) if spec.tool_hint else None
        if tool is not None and tool.platform_hint != Platform.SHARED:
            return tool.platform_hint
        return Platform.SHARED

    def _to_plan(self, schema: PlanSchema) -> Plan:
        subs = {
            s.id: SubTask(id=s.id, description=s.description, depends_on=list(s.depends_on),
                          tool_hint=s.tool_hint, success_criteria=s.success_criteria,
                          platform=self._resolve_platform(s))
            for s in schema.subtasks
        }
        return Plan(goal_id=f"goal-{uuid.uuid4().hex[:8]}", root_goal=schema.root_goal,
                    subtasks=subs, success_criteria=schema.success_criteria)

    def plan(self, goal: str, ws: WorldState, experience: Optional[List[str]] = None) -> Optional[Plan]:
        init: PlannerState = {
            "goal": goal,
            "ws": ws.snapshot(),
            "registry_names": self.registry.list(),
            "attempt": 0,
            "max_retries": self.max_retries,
            "errors": [],
            "experience": experience or [],
            "plan_schema": None,
            "plan": None,
            "status": "running",
        }
        result = self._graph.invoke(init)
        return result.get("plan")
