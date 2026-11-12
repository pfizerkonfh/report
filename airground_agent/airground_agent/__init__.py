"""空地协同多智能体任务分配与边缘智能决策系统。

包结构：
- core:        领域模型（世界状态 / 目标寄存器 / 任务 DAG / LLM 抽象）
- tools:       工具框架 + 可插拔工具集（含 Precondition/Effect/Rollback 语义）
- planner:     规划层（LangGraph，目标分解 + 可行性校验）
- reasoner:    推理层（LangGraph ReAct 循环 + Reflexion 经验回流）
- routing:     端-边-云任务路由协议与网络治理
- eval:        统一评测体系（四指标 / 离线回放 / 跨层归因 / A/B）
- orchestrator:总装入口
"""
from .orchestrator import Orchestrator, RunResult
from .config.settings import Settings, get_settings

__version__ = "0.1.0"

__all__ = [
    "__version__", "Orchestrator", "RunResult", "Settings", "get_settings",
]
