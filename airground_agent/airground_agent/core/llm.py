"""LLM 后端抽象（可插拔）。

- MockBackend：确定性、基于规则的离线后端，用于单测与 demo，无需联网/API Key。
- LangChainBackend：对接真实大模型（langchain-openai），通过 with_structured_output 返回 schema。

两个后端共享同一接口 `generate(system, user, schema, context)`，保证 Planner/Reasoner
在「离线可跑」与「生产可用」之间无缝切换（对应设计文档的落地诉求）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from .log import get_logger
from .schemas import PlanSchema, ReasonerDecision, SubTaskSpec

logger = get_logger("llm")

T = TypeVar("T", bound=BaseModel)


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system: str, user: str, schema: Type[T], context: Optional[Dict[str, Any]] = None) -> T:
        ...


# ----------------------------------------------------------------------------
# Mock 后端：基于关键词的确定性规划 + 计划对齐的推理
# ----------------------------------------------------------------------------
class MockBackend(LLMBackend):
    def __init__(self) -> None:
        # 可修复前置失败 -> (插入节点, 工具, 设置的事实)
        self._fixable = {
            "battery": ("t_charge", "充电恢复无人机电量", "charge", "battery", 100),
        }

    def generate(self, system, user, schema, context=None):
        if schema is PlanSchema:
            return self._plan(context or {})
        if schema is ReasonerDecision:
            return self._reason(context or {})
        raise NotImplementedError(f"MockBackend 不支持 schema: {schema}")

    # ---- Planner ----
    def _plan(self, ctx: Dict[str, Any]) -> PlanSchema:
        goal: str = ctx.get("goal", "")
        errors: list = ctx.get("errors", [])
        base = self._base_plan(goal)
        if errors:
            return self._try_fix(base, errors, ctx)
        return base

    def _base_plan(self, goal: str) -> PlanSchema:
        g = goal
        # 空地协同：采集类子任务分别判给无人机（空中）与机器狗（地面），计算类判给共享算力
        if any(k in g for k in ("素材", "拍摄", "拍", "短片", "剪辑", "剪", "去人脸", "人脸", "匿名")):
            return PlanSchema(
                root_goal=g,
                success_criteria="完成空地协同采集到去人脸成片的端到端交付",
                subtasks=[
                    SubTaskSpec(id="t_weather", description="查询当日天气与航线可行性", depends_on=[], tool_hint="weather", success_criteria="获取天气与航线许可", platform="shared"),
                    SubTaskSpec(id="t_capture", description="无人机航拍变电站高空设备", depends_on=["t_weather"], tool_hint="camera", success_criteria="生成航拍素材 footage", platform="uav"),
                    SubTaskSpec(id="t_dog", description="机器狗地面巡检拍摄表计与开关柜", depends_on=["t_weather"], tool_hint="camera", success_criteria="生成地面素材 footage", platform="dog"),
                    SubTaskSpec(id="t_edit", description="融合空地素材剪辑为巡检短片", depends_on=["t_capture", "t_dog"], tool_hint="edit", success_criteria="生成剪辑后短片 edited", platform="shared"),
                    SubTaskSpec(id="t_anon", description="对短片中人脸做匿名化", depends_on=["t_edit"], tool_hint="anonymize", success_criteria="输出去人脸成片 anonymized", platform="shared"),
                ],
            )
        if any(k in g for k in ("控制", "导航", "vla", "VLA")):
            return PlanSchema(
                root_goal=g,
                success_criteria="端侧 VLA 推理输出空地协同安全控制指令",
                subtasks=[
                    SubTaskSpec(id="t_vla_uav", description="无人机端侧 VLA 推理生成航迹控制", depends_on=[], tool_hint="vla", success_criteria="输出无人机 control_cmd", platform="uav"),
                    SubTaskSpec(id="t_vla_dog", description="机器狗端侧 VLA 推理生成移动控制", depends_on=[], tool_hint="vla", success_criteria="输出机器狗 control_cmd", platform="dog"),
                ],
            )
        return PlanSchema(
            root_goal=g,
            success_criteria="完成用户目标",
            subtasks=[
                SubTaskSpec(id="t_uav", description="无人机航拍变电站高空设备", depends_on=[], tool_hint="camera", success_criteria="生成航拍巡检素材", platform="uav"),
                SubTaskSpec(id="t_dog", description="机器狗地面巡检设备与表计", depends_on=[], tool_hint="vla", success_criteria="输出地面巡检结果", platform="dog"),
                SubTaskSpec(id="t_cloud", description="云端融合空地数据生成报告", depends_on=["t_uav", "t_dog"], tool_hint="cloud_llm", success_criteria="输出分析报告", platform="shared"),
            ],
        )

    def _try_fix(self, base: PlanSchema, errors: list, ctx: Dict[str, Any]) -> PlanSchema:
        joined = " ".join(errors).lower()
        for kw, (nid, desc, tool, fact, val) in self._fixable.items():
            if kw in joined:
                # 在第一个失败节点的前置插入修复节点
                subs = list(base.subtasks)
                subs.insert(0, SubTaskSpec(id=nid, description=desc, depends_on=[], tool_hint=tool,
                                           success_criteria=f"设置 {fact}>={val}", platform="shared"))
                # 让原首节点依赖修复节点
                if subs[1].depends_on == []:
                    subs[1].depends_on = [nid]
                return PlanSchema(root_goal=base.root_goal, success_criteria=base.success_criteria, subtasks=subs)
        # 无法修复：返回原计划（将触发 Reflexion）
        return base

    # ---- Reasoner ----
    def _reason(self, ctx: Dict[str, Any]) -> ReasonerDecision:
        node = ctx.get("node")
        ws: Dict[str, Any] = ctx.get("ws", {})
        tool = node.tool_hint
        args = self._build_args(tool, ws, node)
        thought = f"子任务[{node.id}]：{node.description}；依据计划选用工具 `{tool}`，服务于最终目标。"
        return ReasonerDecision(thought=thought, subtask_id=node.id, tool=tool, args=args, aligned=True, rationale="遵循 GoalRegistry 登记计划")

    @staticmethod
    def _build_args(tool: str, ws: Dict[str, Any], node) -> Dict:
        if tool == "camera":
            return {"scene": "substation", "target": "defect"}
        if tool in ("edit", "anonymize"):
            return {"source": ws.get("footage_path", "/data/footage.raw")}
        if tool == "weather":
            return {"site": "substation"}
        if tool == "vla":
            return {"observation": ws.get("obs", "rgb+depth")}
        if tool == "charge":
            return {}
        return {"prompt": node.description}


# ----------------------------------------------------------------------------
# LangChain 后端（生产可用，可选依赖 langchain-openai）
# ----------------------------------------------------------------------------
class LangChainBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: str, base_url: str = "", temperature: float = 0.2):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("使用 LangChainBackend 需安装 langchain-openai") from e
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key or None,
            base_url=base_url or None,
            temperature=temperature,
        )

    def generate(self, system, user, schema, context=None):
        from langchain_core.messages import HumanMessage, SystemMessage
        chain = self.llm.with_structured_output(schema)
        return chain.invoke([SystemMessage(content=system), HumanMessage(content=user)])


# ----------------------------------------------------------------------------
# OpenAI 兼容后端（中转站 / 自建网关 / vLLM / Ollama 等 OpenAI 风格接口）
# ----------------------------------------------------------------------------
class OpenAICompatibleBackend(LLMBackend):
    def __init__(self, model_name: str, api_key: str, base_url: str = "",
                 temperature: float = 0.2, max_tokens: int = 1024, timeout: float = 60.0,
                 max_retries: int = 4):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("使用 OpenAICompatibleBackend 需安装 openai") from e
        # 规范化 base_url：OpenAI SDK 需要在末尾带 /v1
        if base_url:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = base_url + "/v1"
        self.client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url or None, timeout=timeout)
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    @staticmethod
    def _schema_hint(schema: Type[T]) -> str:
        try:
            js = schema.model_json_schema()
        except Exception:
            return ""
        props = js.get("properties", {})
        lines = []
        for name, meta in props.items():
            t = meta.get("type", "any")
            desc = meta.get("description", "")
            lines.append(f"  - {name} ({t}): {desc}")
        return "\n".join(lines)

    def _uses_max_completion_tokens(self) -> bool:
        """GPT-5/6 与 O 系列弃用 max_tokens，改用 max_completion_tokens；其余模型沿用 max_tokens。"""
        m = (self.model_name or "").lower()
        return any(t in m for t in ("gpt-5", "gpt-6", "o1", "o2", "o3", "o4", "o5"))

    # 部分中转站对 GPT-5/6 的 max_completion_tokens 有硬上限（实测 gpt-5.5 在 256 可用、320+ 必 500）
    _GPT5_MAX_COMPLETION_CAP = 256

    def _tok_kwargs(self) -> Dict[str, int]:
        if self._uses_max_completion_tokens():
            cap = min(self.max_tokens, self._GPT5_MAX_COMPLETION_CAP)
            return {"max_completion_tokens": cap}
        return {"max_tokens": self.max_tokens}

    @staticmethod
    def _strip_fences(text: str) -> str:
        """去掉 ```json ... ``` 代码围栏等包裹。"""
        import re
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
        if m:
            return m.group(1).strip()
        return text.strip()

    @staticmethod
    def _repair_truncated(text: str) -> str:
        """对可能被 token 上限截断的 JSON 做最小修复：补齐未闭合的括号/引号。"""
        s = text.rstrip()
        # 去掉尾部不完整的字符串（最后一个 " 之后没有闭合）
        if s.count('"') % 2 == 1:
            last_q = s.rfind('"')
            s = s[:last_q]
        depth = {"{": 0, "[": 0}
        for ch in s:
            if ch in depth:
                depth[ch] += 1
            elif ch in ("}", "]"):
                key = "{" if ch == "}" else "["
                if depth[key] > 0:
                    depth[key] -= 1
        need = "}" * depth["{"] + "]" * depth["["]
        return s + need

    def _parse(self, schema, content: str):
        raw = self._strip_fences(content)
        # 1) 直接校验
        try:
            return schema.model_validate_json(raw)
        except Exception:
            pass
        # 2) 数组形式（模型有时直接返回 subtasks 数组）：包装为 PlanSchema
        if raw.lstrip().startswith("["):
            try:
                from typing import List as _List
                from pydantic import TypeAdapter
                from .schemas import SubTaskSpec
                subs = TypeAdapter(_List[SubTaskSpec]).validate_json(raw)
                from .schemas import PlanSchema
                return PlanSchema(subtasks=subs)
            except Exception:
                pass
        # 3) 抠出对象并修复截断
        import re
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return schema.model_validate_json(self._repair_truncated(m.group(0)))
            except Exception:
                pass
        raise RuntimeError(
            f"模型返回无法解析为 {schema.__name__}，原始内容: {content[:500]}"
        )

    def generate(self, system, user, schema, context=None):
        import time
        from openai import APIConnectionError, APIStatusError, InternalServerError, RateLimitError

        hint = self._schema_hint(schema)
        sys_msg = (
            system
            + "\n\n[输出格式要求] 你必须且只返回一个合法的 JSON 对象，严格匹配以下字段结构"
              "（不要包含任何额外说明文字，也不要使用 Markdown 代码块）：\n"
            + hint
        )
        # GPT-5/6 与 O 系列在中转站常不支持 response_format=json_object（会 500），
        # 这类模型改用纯提示词约束 + 下方正则兜底解析。
        kwargs: Dict[str, object] = dict(
            model=self.model_name,
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
            temperature=self.temperature,
            **self._tok_kwargs(),
        )
        if not self._uses_max_completion_tokens():
            kwargs["response_format"] = {"type": "json_object"}

        # 部分中转站上游偶发 500/限流，做指数退避重试（仅对可重试的 5xx/429/连接错误）。
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or "{}"
                return self._parse(schema, content)
            except (InternalServerError, RateLimitError, APIConnectionError, APIStatusError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    backoff = 2 ** (attempt + 1)
                    logger.warning("LLM 调用暂失败，退避重试", attempt=attempt + 1,
                                   backoff_s=backoff, err=str(e)[:120])
                    time.sleep(backoff)
                    continue
                break
        raise RuntimeError(f"真实模型调用在 {self.max_retries} 次重试后仍失败：{last_err}") from last_err

    def ping(self) -> None:
        """最小连通性自检：发一个无 schema 的小请求，失败则抛出带诊断的 RuntimeError。"""
        import time
        from openai import APIConnectionError, AuthenticationError, InternalServerError, NotFoundError, RateLimitError
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "ping"}],
                    **({"max_completion_tokens": 4} if self._uses_max_completion_tokens() else {"max_tokens": 4}),
                )
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                if isinstance(e, AuthenticationError):
                    raise RuntimeError(
                        "鉴权失败(401)：API Key 无效或未授权，请检查 config/real_model.yaml 的 llm.api_key"
                    ) from e
                if isinstance(e, NotFoundError):
                    raise RuntimeError(
                        f"模型或路径不存在(404)：请确认 model_name={self.model_name!r} 在中转站可用，"
                        f"且 base_url 正确（代码已自动补 /v1）"
                    ) from e
                # 其余（含 5xx/限流/连接错误）视为可重试或通用错误
                if attempt == 0:
                    time.sleep(2)
                    continue
                break
        # 兜底：若循环因可重试错误退出，给出通用诊断
        if last_err is not None:
            if isinstance(last_err, APIConnectionError):
                raise RuntimeError(
                    f"连接失败：无法到达 {self.base_url or 'OpenAI'}，请检查网络或中转站地址"
                ) from last_err
            raise RuntimeError(f"中转站请求异常：{last_err}") from last_err


_BACKEND: Optional[LLMBackend] = None


def get_backend(backend: str = "mock", **kwargs) -> LLMBackend:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    if backend == "mock":
        _BACKEND = MockBackend()
    elif backend == "langchain":
        _BACKEND = LangChainBackend(**kwargs)
    elif backend == "openai":
        _BACKEND = OpenAICompatibleBackend(**kwargs)
    else:
        raise ValueError(f"未知 LLM backend: {backend}")
    return _BACKEND


def build_backend(backend: str = "mock", model_name: str = "", api_key: str = "",
                  base_url: str = "", temperature: float = 0.2, max_tokens: int = 1024,
                  timeout: float = 60.0, max_retries: int = 4) -> LLMBackend:
    """按配置构建一个 LLM 后端实例（不依赖全局缓存，便于在 Orchestrator 中显式注入）。"""
    if backend == "mock":
        return MockBackend()
    if backend == "langchain":
        return LangChainBackend(model_name=model_name, api_key=api_key, base_url=base_url,
                                temperature=temperature)
    if backend == "openai":
        return OpenAICompatibleBackend(model_name=model_name, api_key=api_key, base_url=base_url,
                                       temperature=temperature, max_tokens=max_tokens,
                                       timeout=timeout, max_retries=max_retries)
    raise ValueError(f"未知 LLM backend: {backend}")


def reset_backend() -> None:
    global _BACKEND
    _BACKEND = None
