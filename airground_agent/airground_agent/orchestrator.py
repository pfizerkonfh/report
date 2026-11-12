"""总装入口 Orchestrator：把目标寄存器 / 世界状态 / Planner / Reasoner / 路由 / 网络 /
经验回流 / 评测体系串成一条可运行的端到端链路。

对外只暴露 run(goal, ws_init) -> RunResult，内部完成：
规划 -> 登记目标 -> ReAct 执行（含前置阻断/重规划/Reflexion）-> 统一评测采集。
"""
from __future__ import annotations

import uuid
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .config.settings import Settings, get_settings
from .core.llm import build_backend
from .core.goal_registry import GoalRegistry
from .core.task_model import Plan, Reflection, TraceStep
from .core.types import RunStatus
from .core.world_state import WorldState
from .planner.planner import Planner
from .reasoner.reasoner import Reasoner
from .reasoner.reflexion import ExperienceBuffer
from .routing.network import NetworkGovernor
from .routing.router import Router
from .tools.adapters import AdapterBundle, CancellationToken, SimConfig, build_adapter_bundle
from .tools.executor import ToolExecutor
from .perception import build_perception_tools, load_scene
from .tools.registry import ToolRegistry
from .eval.metrics import EvalRecord, compute_metrics
from .eval.harness import EvalHarness
from .core.log import get_logger

logger = get_logger("orchestrator")


class RunResult(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run-{uuid.uuid4().hex[:8]}")
    goal: str = ""
    status: RunStatus = RunStatus.FAILED
    plan: Optional[Plan] = None
    trace: List[TraceStep] = Field(default_factory=list)
    reflection: Optional[Reflection] = None
    final_facts: Dict = Field(default_factory=dict)
    metrics: Optional[EvalRecord] = None
    success_criteria_met: bool = False


class Orchestrator:
    def __init__(self, settings: Optional[Settings] = None,
                 buffer_path: Optional[str] = None,
                 on_event: Optional[Callable[[str, Dict], None]] = None,
                 sim: Optional[SimConfig] = None) -> None:
        self.settings = settings or get_settings()
        self._on_event = on_event
        # 执行通道配置：sim 显式传入优先，否则取 settings.sim（默认 legacy）
        self.sim = sim or SimConfig.from_obj(getattr(self.settings, "sim", None))
        # 感知层：加载真实 3DGS 重建场景（找不到则优雅降级为纯逻辑仿真）
        self.perception_cfg = getattr(self.settings, "perception", None)
        self.scene = None
        if self.perception_cfg is not None:
            self.scene = load_scene(
                data_dir=getattr(self.perception_cfg, "data_dir", ""),
                enabled=getattr(self.perception_cfg, "enabled", True),
                max_points=getattr(self.perception_cfg, "max_points", 6000),
            )
        if self.scene is not None:
            logger.info("感知层已加载真实场景", source=self.scene.source)
        perception_tools = build_perception_tools(self.scene) if self.scene is not None else []
        self.registry = ToolRegistry.default(self.settings.vla, extra_tools=perception_tools)
        self.net = NetworkGovernor(self.settings.router)
        self.router = Router(self.net)
        # 执行通道（legacy / sim / playback / ros2）：切换它即可换真机，算法零改动
        self.adapters: AdapterBundle = build_adapter_bundle(
            self.sim, registry=self.registry, vla_settings=self.settings.vla,
            dog_cfg=getattr(self.settings, "dog", None))
        self.token = CancellationToken()
        self.executor = ToolExecutor(self.registry, self.router, self.net,
                                     adapters=self.adapters, sim=self.sim)
        self.buffer = ExperienceBuffer(buffer_path)
        self.backend = build_backend(**self.settings.llm.model_dump())
        self.planner = Planner(self.registry, backend=self.backend,
                               max_retries=self.settings.planner.max_retries)
        self.reasoner = Reasoner(
            self.registry, self.executor, self.planner, self.buffer,
            backend=self.backend,
            max_steps=self.settings.reasoner.max_steps,
            max_replan_attempts=self.settings.reasoner.max_replan_attempts,
            on_event=on_event, token=self.token,
        )
        self.harness = EvalHarness()

    def abort(self, reason: str = "用户中止") -> None:
        """中止当前运行（急停）：令牌生效后长任务会在 ~50ms 内退出。"""
        self.token.cancel(reason)
        self.adapters.abort_all(reason)

    def adapter_health(self) -> Dict:
        """执行通道健康状态（前端展示"当前跑在什么通道上"）。"""
        return self.adapters.health()

    # ------------------------------------------------------------------
    # 机器狗接入：把本轮规划编译成真机指令，并可选择直接下发
    # ------------------------------------------------------------------
    def dog_compile_cfg(self):
        """由 `settings.dog` 构造编译配置（坐标系标定、航点抽稀、地图 ID 等）。"""
        from .dog import CompileConfig
        from .dog.frames import FrameTransform
        d = getattr(self.settings, "dog", None)
        if d is None:
            return CompileConfig()
        data = d.model_dump() if hasattr(d, "model_dump") else dict(d)
        frame = FrameTransform(
            yaw_deg=float(data.get("yaw_deg", 0.0) or 0.0),
            translate=list(data.get("translate") or [0.0, 0.0, 0.0]),
            scale=float(data.get("scale", 1.0) or 1.0),
            flip_x=bool(data.get("flip_x", False)),
            flip_y=bool(data.get("flip_y", False)),
        )
        return CompileConfig(
            map_id=str(data.get("map_id", "") or ""),
            path_id=str(data.get("path_id", "") or ""),
            nav_mode=str(data.get("nav_mode", "auto") or "auto"),
            frame=frame,
            waypoint_stride_m=float(data.get("waypoint_stride_m", 1.0) or 1.0),
            max_waypoints=int(data.get("max_waypoints", 40) or 40),
            heading_mode=str(data.get("heading_mode", "along_path") or "along_path"),
            fixed_yaw_deg=float(data.get("fixed_yaw_deg", 0.0) or 0.0),
            nav_timeout_s=float(data.get("nav_timeout_s", 120.0) or 120.0),
            collect_seconds=float(data.get("collect_seconds", 3.0) or 3.0),
            infer_approach=bool(data.get("infer_approach", False)),
        )

    def to_dog_mission(self, result: "RunResult", cfg=None) -> "DogMission":
        """把一轮规划结果编译成机器狗指令序列（不连接真机，纯编译）。

        只取 platform=DOG 的子任务；航点来自 trace 中 route_plan 的真实 A* 结果，
        缺失时用加载的 3DGS 场景现场现算。返回的 mission 可直接导出 JSON 或下发。
        """
        from .dog import compile_plan
        if result.plan is None:
            raise ValueError("该次运行没有可用 Plan（规划失败），无法编译机器狗任务")
        return compile_plan(result.plan, result.trace, cfg=cfg or self.dog_compile_cfg(),
                            scene=self.scene, goal=result.goal, run_id=result.run_id)

    def dispatch_to_dog(self, result: "RunResult", cfg=None, transport=None,
                        dry_run: Optional[bool] = None, token=None):
        """编译 + 下发：把本轮规划真正送到机器狗上执行，返回执行报告。

        dry_run=True（默认取 settings.dog.dry_run）时只记录不下发，用于联调。
        """
        from .dog import DogMissionRunner, build_transport
        d = getattr(self.settings, "dog", None)
        mission = self.to_dog_mission(result, cfg=cfg)
        if transport is None:
            if dry_run is None:
                dry_run = bool(getattr(d, "dry_run", True) if d is not None else True)
            transport = build_transport(
                uri=str(getattr(d, "ws_uri", "") or "") if d is not None else "",
                exe_path=str(getattr(d, "daemon_exe", "") or "") if d is not None else "",
                dog_ip=str(getattr(d, "dog_ip", "") or "") if d is not None else "",
                send_port=int(getattr(d, "send_port", 8081) or 8081) if d is not None else 8081,
                recv_port=int(getattr(d, "recv_port", 8080) or 8080) if d is not None else 8080,
                dry_run=bool(dry_run),
            )
        runner = DogMissionRunner(transport, token=token or self.token)
        try:
            return runner.run(mission)
        finally:
            if not getattr(transport, "dry_run", True):
                transport.close()

    def scene_info(self) -> Optional[Dict]:
        """真实场景元数据（前端可据此渲染真实占据栅格地图）。"""
        return self.scene.to_dict() if self.scene is not None else None

    def _emit(self, etype: str, payload: Dict) -> None:
        if self._on_event is not None:
            try:
                self._on_event(etype, payload)
            except Exception:  # 观测钩子不应影响主流程
                pass

    # 默认初始世界状态：调用方未显式提供 ws_init 时使用。
    # 这些是「作业开始前已确认的现场条件」——电量满、边侧可用、无人机在线、
    # 安全联锁已放行。真机部署时其中 safety_cleared / battery / uav_online
    # 必须由安全联锁与机载遥测真实写入，此处仅为仿真/演示的初始假设。
    # 缺了它们会导致 camera 等工具被前置校验无条件阻断（Web 控制台曾因此默认全败）。
    DEFAULT_WS: Dict[str, object] = {
        "battery": 100,
        "safety_cleared": True,
        "edge_capable": True,
        "uav_online": True,
        "weather_ok": False,      # 由 weather 工具据实置位
        "footage": False,
        "edited": False,
        "anonymized": False,
    }

    def run(self, goal: str, ws_init: Optional[Dict] = None, csat: Optional[float] = None) -> RunResult:
        facts = dict(self.DEFAULT_WS)
        facts.update(ws_init or {})          # 调用方显式传入优先
        ws = WorldState(facts=facts)
        goal_reg = GoalRegistry()
        logger.info("开始任务", goal=goal)

        # 感知层接入：把真实场景事实注入世界状态，使前置校验建立在真实数据之上
        if self.scene is not None:
            self._emit("scene", self.scene.to_dict(include_grid=False))
            if getattr(self.perception_cfg, "auto_inject_facts", True):
                for k, v in self.scene.to_facts().items():
                    if k not in ws.facts:
                        ws.facts[k] = v

        self._emit("planning", {"goal": goal})
        plan: Optional[Plan] = self.planner.plan(goal, ws)
        if plan is None:
            self._emit("plan_failed", {"goal": goal})
            logger.error("规划失败：无法产出可行 Plan")
            rec = compute_metrics(goal, RunStatus.FAILED, None, [], ws.facts, csat or 2.0)
            self.harness.add(rec)
            return RunResult(goal=goal, status=RunStatus.FAILED, metrics=rec)

        self._emit("plan", plan.model_dump(mode="json"))
        plan.register_to(goal_reg)
        state = self.reasoner.run(plan, ws, goal_reg)
        final_ws = self.reasoner._ws

        success = plan.is_complete()
        status = RunStatus.SUCCESS if success else RunStatus.FAILED
        reflection = state.get("reflection")
        root = reflection.root_cause_layer if reflection else None
        score = csat if csat is not None else (4.5 if success else 2.5)
        rec = compute_metrics(goal, status, plan, state["trace"], final_ws.facts, score, root)
        self.harness.add(rec)

        logger.info("任务结束", status=status.value, steps=len(state["trace"]),
                    p95=rec.latency_p95, csat=score)
        return RunResult(
            goal=goal, status=status, plan=plan, trace=state["trace"],
            reflection=reflection, final_facts=final_ws.facts,
            metrics=rec, success_criteria_met=success,
        )
