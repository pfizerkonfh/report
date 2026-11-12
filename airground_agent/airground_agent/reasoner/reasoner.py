"""推理层 Reasoner（LangGraph ReAct 循环）。

职责（设计文档问题①）：
- 逐步选工具、校验 Precondition、执行、观测效果（ReAct：Thought→Action→Observation）；
- Thought 必须引用 GoalRegistry，任何「偏离全局目标」的动作被拦截；
- Precondition 不满足时**阻断并升级重规划**，而非擅自替换工具；
- 多次重规划仍失败 -> 触发 **Reflexion**：回放轨迹、写 self-critique/lessons 入 ExperienceBuffer。

控制状态走 LangGraph 状态图（select→reason→execute→post→[replan/reflect]），
重状态（Plan/WorldState/GoalRegistry）挂在实例上共享，避免大模型对象在图中序列化。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.llm import LLMBackend, get_backend
from ..core.schemas import ReasonerDecision
from ..core.task_model import Plan, Reflection, TaskStatus, TraceStep
from ..core.types import FailureLayer
from ..core.world_state import WorldState
from ..planner.planner import Planner
from ..tools.adapters import CancellationToken
from ..tools.executor import ToolExecutor
from ..tools.registry import ToolRegistry
from ..core.log import get_logger
from .reflexion import ExperienceBuffer

logger = get_logger("reasoner")

REASONER_SYSTEM = """[ROLE:REASONER]
你是执行子任务的推理器，遵循 ReAct（Thought→Action→Observation）。
规则：
1. Thought 必须说明本动作如何服务 GoalRegistry 中登记的最终目标与对应子任务成功判据。
2. Action 只能选择当前子任务建议的工具（tool_hint），不得擅自调用无关工具。
3. 若工具 Precondition 不满足，不要替换工具，应上报阻断并等待重规划。
4. 每步都要校验「动作是否仍服务全局目标」，偏离则拒绝执行。
"""


class ReasonerState(TypedDict):
    status: str                 # running | success | blocked | reflect | failed
    current_node: Optional[str]
    step: int
    replan_attempt: int
    errors: List[str]
    trace: List[TraceStep]
    reflection: Optional[Reflection]


class Reasoner:
    def __init__(self, registry: ToolRegistry, executor: ToolExecutor, planner: Planner,
                 buffer: ExperienceBuffer, backend: Optional[LLMBackend] = None,
                 max_steps: int = 32, max_replan_attempts: int = 2,
                 on_event: Optional[Callable[[str, Dict], None]] = None,
                 token: Optional[CancellationToken] = None) -> None:
        self.registry = registry
        self.executor = executor
        self.planner = planner
        self.buffer = buffer
        self.backend = backend or get_backend()
        self.max_steps = max_steps
        self.max_replan_attempts = max_replan_attempts
        self._on_event = on_event
        self._token = token          # 中止令牌（由上层持有，用于急停 / 用户取消）
        self._graph = self._build()

        # 重状态（每次 run 重置）
        self._plan: Optional[Plan] = None
        self._ws: Optional[WorldState] = None
        self._goal_reg = None
        self._decision: Optional[ReasonerDecision] = None

    def _emit(self, etype: str, payload: Dict) -> None:
        if self._on_event is not None:
            try:
                self._on_event(etype, payload)
            except Exception:  # 观测钩子不应影响主流程
                pass

    # ---------------- 图构建 ----------------
    def _build(self):
        g = StateGraph(ReasonerState)
        g.add_node("select_node", self._select)
        g.add_node("reason", self._reason)
        g.add_node("execute", self._execute)
        g.add_node("replan", self._replan)
        g.add_node("reflect", self._reflect)
        g.add_edge(START, "select_node")
        g.add_conditional_edges("select_node", self._route_select,
                                {"reason": "reason", "done": END, "failed_end": END,
                                 "cancelled": END})
        g.add_edge("reason", "execute")
        g.add_conditional_edges("execute", self._route_post,
                                {"continue": "select_node", "replan": "replan", "done": END,
                                 "cancelled": END})
        g.add_conditional_edges("replan", self._route_replan,
                                {"loop": "select_node", "reflect": "reflect"})
        g.add_edge("reflect", END)
        return g.compile()

    # ---------------- 节点 ----------------
    def _select(self, state: ReasonerState) -> Dict:
        # 急停优先：已中止时必须立即停机，而不是继续选下一个子任务去重试。
        # 真机场景下"急停后仍在自主行动"是不可接受的安全事故。
        if self._token is not None and self._token.is_cancelled():
            self._emit("cancelled", {"reason": self._token.reason})
            return {"status": "cancelled", "errors": [self._token.reason]}
        if state["step"] > self.max_steps:
            self._emit("blocked", {"reason": "步数超过上限，疑似失控循环"})
            return {"status": "failed", "errors": ["步数超过上限，疑似失控循环"]}
        if self._plan.is_complete():
            return {"status": "success"}
        ready = self._plan.ready_nodes()
        if not ready:
            self._emit("blocked", {"reason": "无可执行的就绪节点（存在未满足的依赖）"})
            return {"status": "blocked", "errors": ["无可执行的就绪节点（存在未满足的依赖）"]}
        node = ready[0]
        self._goal_reg.mark_running(node)
        st = self._plan.subtasks.get(node)
        self._emit("select", {"node": node,
                              "description": st.description if st else "",
                              "tool_hint": st.tool_hint if st else "",
                              "platform": st.platform.value if st else "shared",
                              "depends_on": st.depends_on if st else []})
        return {"current_node": node, "status": "running"}

    def _reason(self, state: ReasonerState) -> Dict:
        node = self._plan.subtasks[state["current_node"]]
        ws_summary = self._ws.snapshot()
        tools = ", ".join(self.registry.list())
        user_prompt = (
            f"最终目标：{self._goal_reg.user_goal}\n"
            f"当前子任务：[{node.id}] {node.description}（成功判据：{node.success_criteria}）\n"
            f"世界状态：{ws_summary}\n可用工具：{tools}\n"
            f"请决定本步 Thought/Action，subtask_id 必须为 {node.id}。"
        )
        decision = self.backend.generate(
            REASONER_SYSTEM, user_prompt, ReasonerDecision,
            context={"node": node, "ws": ws_summary, "plan": self._plan.model_dump(),
                     "trace": [t.model_dump() for t in state["trace"]]},
        )
        # 对齐校验：动作必须绑定到当前计划内、仍待完成的子任务
        if decision.subtask_id != node.id or not self._goal_reg.is_aligned(decision.tool, node.id):
            logger.warning("动作偏离全局目标，已拦截", tool=decision.tool, node=node.id)
            return self._reason_blocked(
                node, [f"动作偏离全局目标：tool={decision.tool} 不服务节点 {node.id}"])
        self._decision = decision
        return {}

    def _progress_emitter(self, node: str):
        """把执行通道的相位进度转成事件，供前端实时展示"无人机巡航中 45%"。

        真机长任务会持续回传进度，这里做节流（每 10% 一次），避免事件风暴。
        """
        if self._on_event is None:
            return None
        last = {"bucket": -1}

        def cb(frac: float, phase: str) -> None:
            if frac is None or frac < 0:
                return  # 真机未上报进度
            bucket = int(min(1.0, max(0.0, frac)) * 10)
            if bucket == last["bucket"] and frac < 1.0:
                return
            last["bucket"] = bucket
            self._emit("progress", {"node": node, "phase": phase,
                                    "percent": round(min(1.0, max(0.0, frac)), 3)})

        return cb

    def _reason_blocked(self, node, errors):
        self._emit("blocked", {"node": node.id, "reason": "; ".join(errors)})
        return {"status": "blocked", "errors": errors}

    def _execute(self, state: ReasonerState) -> Dict:
        decision = self._decision
        node = state["current_node"]
        new_step = state["step"] + 1
        subtask = self._plan.subtasks[node]
        res = self.executor.execute(decision.tool, self._ws, decision.args,
                                    platform=subtask.platform,
                                    progress_cb=self._progress_emitter(node),
                                    token=self._token)
        step = TraceStep(
            step_id=new_step, subtask_id=node,
            thought=decision.thought,
            action={"tool": decision.tool, "args": decision.args},
            observation="", precondition_ok=not res.blocked,
            precondition_failed=res.precondition_failed,
            tier_used=res.decision.chosen_tier if res.decision else None,
            platform=subtask.platform,
            latency_ms=res.result.latency_ms if res.result else 0.0,
            sim_duration_ms=res.result.sim_duration_ms if res.result else 0.0,
            telemetry=res.result.telemetry if res.result else {},
        )
        if res.blocked:
            step.observation = "前置校验阻断：" + "; ".join(res.precondition_failed)
            logger.warning("前置阻断", node=node, failed=res.precondition_failed)
            self._emit("step", step.model_dump(mode="json"))
            return {"status": "blocked", "errors": res.precondition_failed,
                    "step": new_step, "trace": state["trace"] + [step]}
        if not res.ok:
            step.observation = "执行失败：" + res.reason
            logger.error("执行失败", node=node, reason=res.reason)
            self._emit("step", step.model_dump(mode="json"))
            # 中止属于"停机"而非"可重试失败"：不再进入重规划/Reflexion
            if self._token is not None and self._token.is_cancelled():
                self._emit("cancelled", {"node": node, "reason": self._token.reason})
                return {"status": "cancelled", "errors": [self._token.reason],
                        "step": new_step, "trace": state["trace"] + [step]}
            return {"status": "blocked", "errors": [res.reason],
                    "step": new_step, "trace": state["trace"] + [step]}
        step.observation = res.result.observation
        self._plan.subtasks[node].status = TaskStatus.DONE
        self._goal_reg.mark_done(node)
        logger.info("子任务完成", node=node, tier=step.tier_used, latency=round(step.latency_ms, 1))
        self._emit("step", step.model_dump(mode="json"))
        return {"status": "running", "step": new_step, "trace": state["trace"] + [step]}

    def _replan(self, state: ReasonerState) -> Dict:
        if state["replan_attempt"] < self.max_replan_attempts:
            self._emit("replan", {"attempt": state["replan_attempt"] + 1, "errors": state["errors"]})
            logger.info("触发重规划", attempt=state["replan_attempt"] + 1, errors=state["errors"])
            # 经验回流：将历史 lessons 注入 Planner
            new_plan = self.planner.plan(self._goal_reg.user_goal, self._ws,
                                        self.buffer.lessons_for(self._goal_reg.user_goal))
            if new_plan is not None and new_plan.subtasks:
                new_ids = set(new_plan.subtasks.keys())
                cur_ids = set(self._plan.subtasks.keys())
                if new_ids != cur_ids:
                    # 真正修订计划（如插入修复节点）：替换并登记，保留已完成子任务状态
                    self._plan = new_plan
                    self._plan.register_to(self._goal_reg)
                # 同构计划：保留已完成状态，仅重试阻塞节点
            return {"status": "running", "replan_attempt": state["replan_attempt"] + 1,
                    "current_node": None, "errors": []}
        return {"status": "reflect"}

    def _reflect(self, state: ReasonerState) -> Dict:
        errors = state["errors"] or ["未知失败"]
        layer = self._infer_layer(errors)
        symptom = "; ".join(errors)
        critique = (f"子任务 {state.get('current_node')} 在 {self.max_replan_attempts} 次重规划后仍失败。"
                    f"失败层={layer.value}。")
        lesson = (f"目标「{self._goal_reg.user_goal}」执行失败（层={layer.value}）："
                  f"{symptom}。建议在规划/重规划中显式处理该前置条件或引入兜底子任务。")
        reflection = Reflection(
            failed_subtask=state.get("current_node") or "",
            symptom=symptom, root_cause_layer=layer,
            critique=critique, lessons=lesson, retry_recommended=False,
        )
        self.buffer.add(reflection_to_entry(reflection, self._goal_reg.user_goal))
        logger.warning("Reflexion 已沉淀经验", layer=layer.value)
        self._emit("reflexion", reflection.model_dump(mode="json"))
        return {"status": "failed", "reflection": reflection}

    # ---------------- 工具 ----------------
    def _infer_layer(self, errors: List[str]) -> FailureLayer:
        text = " ".join(errors).lower()
        if "偏离全局目标" in text:
            return FailureLayer.ACTION
        # 前置条件不满足：判断是否有工具能产出该事实 -> ACTION，否则视为计划不可行 -> PLANNING
        import re
        keys = set(re.findall(r"([a-z_]+)\s*(?:==|>=|<=|>|<)", text))
        for k in keys:
            for tool in self.registry._tools.values():
                if any(e.key == k for e in tool.effects):
                    return FailureLayer.ACTION
        return FailureLayer.PLANNING

    # ---------------- 路由 ----------------
    def _route_select(self, state: ReasonerState) -> str:
        if state["status"] == "cancelled":
            return "cancelled"
        if state["status"] == "success":
            return "done"
        if state["status"] == "failed":
            return "failed_end"
        return "reason"

    def _route_post(self, state: ReasonerState) -> str:
        if state["status"] == "cancelled":
            return "cancelled"
        if state["status"] == "success":
            return "done"
        if state["status"] == "blocked":
            return "replan"
        return "continue"

    def _route_replan(self, state: ReasonerState) -> str:
        if state["status"] == "reflect":
            return "reflect"
        return "loop"

    # ---------------- 入口 ----------------
    def run(self, plan: Plan, ws: WorldState, goal_reg) -> Dict:
        self._plan = plan
        self._ws = ws.clone()
        self._goal_reg = goal_reg
        self._counter = 0
        self._decision = None
        # 把初始 step 计数为已执行步数
        init: ReasonerState = {
            "status": "running", "current_node": None, "step": 0,
            "replan_attempt": 0, "errors": [], "trace": [], "reflection": None,
        }
        result = self._graph.invoke(init)
        return result


def reflection_to_entry(reflection: Reflection, goal: str):
    from ..core.task_model import ExperienceEntry
    return ExperienceEntry(goal_pattern=goal, failure_layer=reflection.root_cause_layer,
                           lesson=reflection.lessons)
