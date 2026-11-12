"""全局配置（pydantic-settings）。

优先级：环境变量 AIRGROUND_* > 配置文件（--config 或默认 default.yaml）> 代码默认值。
生产环境下推荐用环境变量注入敏感项（如 LLM_API_KEY）与部署相关项（如路由阈值）。
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    backend: str = "mock"          # mock | langchain | openai（openai 指向 OpenAI 兼容中转站）
    model_name: str = "gpt-4o-mini"
    api_key: str = ""              # 可用环境变量 AIRGROUND_LLM__API_KEY 注入，避免硬编码
    base_url: str = ""             # OpenAI 兼容中转站地址，如 http://host:3000
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 60.0          # 单次请求超时（秒）
    max_retries: int = 4            # 真实端点偶发 5xx/限流时的指数退避重试次数


class PlannerSettings(BaseModel):
    max_retries: int = 3           # 分解/校验失败后的最大重试次数


class ReasonerSettings(BaseModel):
    max_steps: int = 32            # 单轮任务推理最大步数（防 runaway）
    max_replan_attempts: int = 2   # 单节点阻塞后的重规划次数，超过则触发 Reflexion


class RouterSettings(BaseModel):
    edge_latency_budget_ms: float = 60.0     # 端侧时延预算上限（VLA 20Hz→50ms）
    cloud_latency_budget_ms: float = 300.0   # 云端任务可接受时延
    degrade_rtt_ms: float = 200.0            # 网络 RTT 超过该值视为降级
    backoff_base_ms: float = 200.0
    backoff_max_ms: float = 5000.0


class VLASettings(BaseModel):
    """端侧 VLA 推理时延优化开关（对应设计文档问题②）。

    通过逐项开启优化，度量端到端时延收益。
    """
    baseline_latency_ms: float = 65.0
    enable_moe: bool = True        # MoE 稀疏激活
    enable_distill: bool = True    # 知识蒸馏（更激进量化）
    quant_mode: str = "awq_int4"   # awq_int4 | smoothquant_w8a8 | hybrid
    enable_flash_attn: bool = True
    enable_paged_attn: bool = True
    target_frequency_hz: float = 20.0


class EvalSettings(BaseModel):
    csat_default: float = 4.5      # 无真实用户反馈时的占位 CSAT（1-5）
    ab_guardrail_latency_drop_ms: float = 20.0  # 治疗组端到端时延恶化超过该值则回滚


class SimSettings(BaseModel):
    """执行通道（ToolAdapter）配置 —— 决定"事情由谁真正做出来"。

    mode:
      legacy   —— 原 Mock：零耗时、必然成功。仅用于算法回归，不代表真机。
      sim      —— 硬件在环仿真：真实时序 + 真机行为模型（电量/里程/链路/故障）。
      playback —— 录播回放：按现场记录的真实时序复现（确定性，便于 A/B）。
      ros2     —— ROS2 真机通道（需 rclpy 与节点绑定）。
      zsibot   —— ZSIBOT 机器狗真机通道（导航 WebSocket + SDK daemon），
                  机器狗子任务会真实下发；坐标系标定见 `dog` 段。
    """

    mode: str = "legacy"            # legacy | sim | playback | ros2
    time_scale: float = 10.0        # 仿真加速比：10 表示仿真 60s 压缩为真实 6s（1 = 实时）
    failure_rate: float = 0.0       # 作业失败注入概率（0~1）
    comm_loss_rate: float = 0.0     # 链路中断注入概率（0~1）
    timeout_ms: float = 120000.0    # 单次调用超时（仿真时长口径；一次无人机架次约 49s）
    seed: Optional[int] = None      # 随机种子（固定后仿真可复现）
    playback_file: str = ""         # 录播文件（留空则用 examples/sample_playback.jsonl）
    uav_speed_mps: float = 8.0      # 无人机巡航速度
    dog_speed_mps: float = 1.2      # 机器狗地面速度
    min_battery_pct: float = 15.0   # 低于该电量拒绝起飞


class DogBridgeSettings(BaseModel):
    """机器狗接入配置（ZSIBOT 轮式四足）—— 决策平台到真机的桥。

    两条物理通道：
      - `ws_uri`：导航通道，WebSocket `ws://<ip>:10010`，发 `req_func` 帧
        （见厂家 demo nav-05.py）。
      - `daemon_exe`：动作通道，`main_sdk.cpp` 编译出的 `zsibot_control`，
        以 `--daemon` 模式由 Python 通过 stdin/stdout 驱动。

    `dry_run=True`（默认）时只生成/记录指令，不下发真机 —— 联调阶段的安全阀。

    坐标：决策平台用 3DGS 世界系（X-Z 水平、Y 竖直），机器狗用 ROS 地图系
    （X-Y 水平、Z 竖直）。`yaw_deg / translate / scale` 是现场标定参数，
    未标定时只有轴映射，真机下发前必须标定。
    """

    #: 导航 WebSocket 地址
    ws_uri: str = "ws://192.168.234.234:10010"
    #: zsibot_control 可执行程序路径（留空 = 不启用动作通道）
    daemon_exe: str = ""
    #: 机器狗本体 IP（传给 daemon --dog-ip）
    dog_ip: str = "192.168.234.1"
    send_port: int = 8081
    recv_port: int = 8080
    #: 安全阀：True 时只生成指令不下发
    dry_run: bool = True
    #: 狗端定位地图名（留空 = 用狗当前地图）
    map_id: str = ""
    #: 狗端已录制路径 ID（给了走多点导航）
    path_id: str = ""
    #: 导航模式：auto / waypoints / multi
    nav_mode: str = "auto"
    #: 采集窗口（秒）：到位后停留采集时间
    collect_seconds: float = 3.0
    #: 单条导航指令超时（秒）
    nav_timeout_s: float = 120.0
    #: 航点抽稀间距（米）
    waypoint_stride_m: float = 1.0
    max_waypoints: int = 40
    #: 航点朝向：along_path / fixed / none
    heading_mode: str = "along_path"
    fixed_yaw_deg: float = 0.0
    #: 兜底：机器狗子任务未显式要求导航时，用真实场景现算一段前往作业区的路径
    infer_approach: bool = False
    #: ---- 坐标系标定 ----
    yaw_deg: float = 0.0
    translate: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: float = 1.0
    flip_x: bool = False
    flip_y: bool = False


class PerceptionSettings(BaseModel):
    """感知层配置 —— 接入真实 3DGS 重建产物。

    data_dir 留空时自动向上查找 `exp/viewer/showcase_assets/nav_data.json`。
    """

    enabled: bool = True              # 关闭则回退为纯逻辑仿真（不加载真实场景）
    data_dir: str = ""                # 3DGS 产物目录（含 nav_data.json）
    allow_unknown: bool = True        # 规划时是否允许穿越未知格
    auto_inject_facts: bool = True    # 是否把真实场景统计注入初始 WorldState
    max_points: int = 6000            # 点云加载上限（防止内存膨胀）


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIRGROUND_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    planner: PlannerSettings = Field(default_factory=PlannerSettings)
    reasoner: ReasonerSettings = Field(default_factory=ReasonerSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    vla: VLASettings = Field(default_factory=VLASettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    sim: SimSettings = Field(default_factory=SimSettings)
    dog: DogBridgeSettings = Field(default_factory=DogBridgeSettings)
    perception: PerceptionSettings = Field(default_factory=PerceptionSettings)
    log_level: str = "INFO"

    @classmethod
    def from_file(cls, path: str) -> "Settings":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


@lru_cache(maxsize=1)
def get_settings(config_path: Optional[str] = None) -> Settings:
    if config_path and os.path.exists(config_path):
        return Settings.from_file(config_path)
    # 默认尝试读取包同级的 default.yaml
    default = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "default.yaml")
    if os.path.exists(default):
        return Settings.from_file(default)
    return Settings()
