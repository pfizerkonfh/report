"""ROS2 真机适配器（骨架）—— 把 ToolAdapter 协议映射到真实机器人通道。

定位
----
这是**真机通道的接入骨架**，不是可运行的真机驱动：真机的 action 类型、话题名、
坐标系取决于你的机器人栈（DJI / 宇树 / 自研），必须由你在 `ACTION_MAP` 里补全。
本文件负责把"难且通用"的部分做好：

1. **调用语义**：goal 下发 → 等待受理 → 等待结果（真实阻塞）；
2. **超时**：超过 timeout 真实判 TIMEOUT，不做"假装成功"；
3. **取消**：调用 goal.cancel_goal_async() 并等待确认，而非直接抛异常；
4. **遥测**：从 ROS2 话题（电池、里程计、链路）读取真机状态回传；
5. **降级**：rclpy 未安装时给出明确指引，而不是 ImportError 崩溃。

接入步骤（真机落地时）
----------------------
1. `pip install rclpy`（或使用已装 ROS2 的 conda/系统环境）；
2. 按你的机器人栈补全 `ACTION_MAP`：tool -> (action_name, action_type, goal_fields)；
3. 补全 `_goal_from_args()`：把 Planner 给的参数翻译成 ROS2 goal 消息；
4. 补全 `_read_telemetry()`：从 /battery、/odom、/link_quality 等话题取真机状态。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ...core.types import Effect
from .base import AdapterContext, AdapterOutcome, AdapterStatus, ToolAdapter

try:  # ROS2 在 Windows / 无 ROS 环境下通常不可用，这里做可选依赖
    import rclpy  # type: ignore
    from rclpy.action import ActionClient  # type: ignore
    from rclpy.node import Node  # type: ignore

    _RCLPY_AVAILABLE = True
except Exception:  # pragma: no cover - 环境相关
    rclpy = None  # type: ignore
    ActionClient = None  # type: ignore
    Node = object  # type: ignore
    _RCLPY_AVAILABLE = False


#: tool -> (action 名, action 类型名, 说明)。类型名需按你的机器人栈替换。
ACTION_MAP: Dict[str, Dict[str, str]] = {
    "camera": {"action": "/uav/capture", "type": "airground_msgs/action/Capture",
               "note": "无人机航拍 / 机器狗近距采集（按 platform 选择命名空间）"},
    "vla": {"action": "/edge/vla_infer", "type": "airground_msgs/action/VLAInfer",
            "note": "端侧 VLA 推理，输出控制指令"},
    "weather": {"action": "/cloud/weather_query", "type": "airground_msgs/action/WeatherQuery",
                "note": "气象与航线查询（云端）"},
    "edit": {"action": "/cloud/video_edit", "type": "airground_msgs/action/VideoEdit",
             "note": "云端剪辑渲染"},
    "anonymize": {"action": "/edge/anonymize", "type": "airground_msgs/action/Anonymize",
                  "note": "边侧人脸脱敏"},
    "charge": {"action": "/fleet/auto_charge", "type": "airground_msgs/action/AutoCharge",
               "note": "自主充电"},
    "cloud_llm": {"action": "/cloud/llm_analyze", "type": "airground_msgs/action/LLMAnalyze",
                  "note": "云端大模型分析"},
}

#: 平台 -> ROS2 命名空间（空地协同：同一工具在不同平台上走不同节点）
PLATFORM_NS: Dict[str, str] = {"uav": "/uav", "dog": "/dog", "shared": ""}


class Ros2Adapter(ToolAdapter):
    """ROS2 真机适配器。

    未安装 rclpy 时 `invoke()` 返回 FAILED 并给出接入指引 —— 这样上层算法、
    前端、评测都能照常跑起来，不会因为缺依赖而整体崩溃。
    """

    name = "ros2"
    real_hardware = True

    def __init__(self, node: Any = None, namespace: str = "") -> None:
        self._node = node
        self.namespace = namespace
        self._clients: Dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return _RCLPY_AVAILABLE and self._node is not None

    def health(self) -> Dict[str, Any]:
        return {
            "ok": self.available,
            "adapter": self.name,
            "real_hardware": True,
            "rclpy_available": _RCLPY_AVAILABLE,
            "node_bound": self._node is not None,
            "hint": "" if self.available else (
                "未检测到 rclpy 或未绑定 Node。请在 ROS2 环境中运行并传入 node，"
                "或先用 AdapterMode.SIM（硬件在环仿真）验证调度逻辑。"),
        }

    def invoke(self, ctx: AdapterContext) -> AdapterOutcome:
        t0 = time.perf_counter()
        if not self.available:
            return AdapterOutcome(
                status=AdapterStatus.FAILED, success=False,
                observation="ROS2 通道不可用", latency_ms=0.0,
                telemetry={"adapter": self.name, "rclpy_available": _RCLPY_AVAILABLE},
                error="rclpy_missing_or_node_unbound",
            )

        spec = ACTION_MAP.get(ctx.tool_name)
        if spec is None:
            return AdapterOutcome(
                status=AdapterStatus.FAILED, success=False,
                observation=f"工具 {ctx.tool_name} 未映射 ROS2 action",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                telemetry={"adapter": self.name}, error="action_not_mapped",
            )

        ns = PLATFORM_NS.get(ctx.platform.value, "")
        action_name = f"{ns}{spec['action'].lstrip('/')}" if ns else spec["action"]

        # ---- 真机调用：下发 goal 并真实等待结果 ----
        try:
            outcome_status, observation, telemetry = self._send_goal(
                action_name=action_name,
                goal_msg=self._goal_from_args(ctx),
                timeout_ms=ctx.timeout_ms,
                token=ctx.token,
                progress_cb=ctx.progress_cb,
            )
        except Exception as e:  # 真机异常必须被捕获，否则上层无法进入重规划
            return AdapterOutcome(
                status=AdapterStatus.FAILED, success=False,
                observation=f"ROS2 调用异常: {e}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                telemetry={"adapter": self.name, "action": action_name},
                error=str(e),
            )

        wall = (time.perf_counter() - t0) * 1000.0
        telemetry.update({"adapter": self.name, "action": action_name,
                          "namespace": ns, "real_hardware": True})
        success = outcome_status == AdapterStatus.SUCCEEDED
        return AdapterOutcome(
            status=outcome_status, success=success, observation=observation,
            latency_ms=wall, sim_duration_ms=wall, telemetry=telemetry,
            effects=self._effects_from_telemetry(telemetry) if success else None,
            error="" if success else outcome_status.value,
        )

    # ------------------------------------------------------------------
    # 需要按机器人栈补全的三处
    # ------------------------------------------------------------------
    def _goal_from_args(self, ctx: AdapterContext) -> Any:
        """把 Planner 给的参数翻译成 ROS2 goal 消息。**需按你的 action 类型补全。**"""
        return dict(ctx.args)

    def _send_goal(self, action_name: str, goal_msg: Any, timeout_ms: float,
                   token: Any, progress_cb: Any) -> tuple:
        """下发 goal → 等待受理 → 等待结果（真实阻塞，支持超时与取消）。

        骨架已按 ROS2 标准语义写好，接入时只需把 `ActionClient` 的类型参数
        换成你的真实 action 类型。
        """
        client = self._clients.get(action_name)
        if client is None:
            raise RuntimeError(
                f"未创建 ActionClient: {action_name}。请在 _get_client() 中按真实 action 类型创建。")

        future = client.send_goal_async(goal_msg)
        deadline = time.perf_counter() + timeout_ms / 1000.0
        while not future.done():
            if token is not None and token.is_cancelled():
                return AdapterStatus.CANCELLED, "任务被中止", {"phase": "cancelled"}
            if time.perf_counter() > deadline:
                return AdapterStatus.TIMEOUT, f"goal 未在 {timeout_ms:.0f}ms 内被受理", {}
            time.sleep(0.02)

        goal_handle = future.result()
        if not goal_handle.accepted:
            return AdapterStatus.FAILED, "goal 被真机拒绝", {"accepted": False}

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            if token is not None and token.is_cancelled():
                # 真机取消：必须显式下发 cancel 并等待确认，不能只丢弃 future
                cancel_future = goal_handle.cancel_goal_async()
                while not cancel_future.done():
                    time.sleep(0.02)
                return AdapterStatus.CANCELLED, "已下发急停并确认", {"cancelled": True}
            if time.perf_counter() > deadline:
                goal_handle.cancel_goal_async()
                return AdapterStatus.TIMEOUT, f"goal 执行超时（>{timeout_ms:.0f}ms）", {}
            if progress_cb:
                progress_cb(-1.0, "running")  # -1 = 真机未上报进度
            time.sleep(0.05)

        res = result_future.result()
        telemetry = self._read_telemetry(action_name)
        telemetry["result_code"] = getattr(res, "code", 0)
        ok = getattr(res, "success", True)
        return (AdapterStatus.SUCCEEDED if ok else AdapterStatus.FAILED,
                getattr(res, "message", "真机执行完成"), telemetry)

    def _read_telemetry(self, action_name: str) -> Dict[str, Any]:
        """从 ROS2 话题读取真机状态（电量 / 里程 / 链路）。**需按你的话题名补全。**"""
        return {"battery": None, "odometer_m": None, "link_rssi_dbm": None,
                "source": "ros2-topics"}

    @staticmethod
    def _effects_from_telemetry(telemetry: Dict[str, Any]) -> Optional[list]:
        """把真机遥测回写为世界状态副作用（电量以真机上报为准）。"""
        batt = telemetry.get("battery")
        if batt is None:
            return None
        try:
            return [Effect(key="battery", op="set", value=round(float(batt), 1))]
        except (TypeError, ValueError):
            return None

    def abort(self, reason: str = "急停") -> None:
        """真机急停：对所有在飞的 goal 下发取消。"""
        for name, client in self._clients.items():
            try:
                client._cancel_all_goals()  # noqa: SLF001 - ROS2 未提供公开批量接口
            except Exception:
                pass
