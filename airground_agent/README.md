# 空地协同 · 多智能体任务分配与边缘智能决策系统

本项目是「国网变电站空地协同无人巡检」科技项目中 **第（4）项「多智能体任务分配与边缘智能决策系统」** 的落地实现，对应设计文档四大工程问题：

| 设计文档问题 | 本仓库落点 |
| --- | --- |
| ① Agent 长程规划失败 | `core/goal_registry`（目标寄存器）、`tools/base`（工具 Precondition/Effect 语义）、`planner`（Planner/Reasoner 分层）、`reasoner`（ReAct + Reflexion） |
| ② 端侧 VLA 模型延迟 | `tools/latency.py`（MoE/蒸馏 + AWQ/SmoothQuant + Flash/PagedAttention 三层收益仿真） |
| ③ 端-边-云任务路由 | `routing/`（RouteDecision 协议 + 判定树 + 失败回退 + 网络治理） |
| ④ 统一评测体系 | `eval/`（规划成功率 / 动作合理度 / 端到端时延 / CSAT，离线回放 + 跨层归因 + A/B 护栏） |

技术栈：**Python + LangGraph**（Planner、Reasoner 均为 StateGraph 实现），LLM 后端可插拔（离线 `mock` / 生产 `langchain`）。

---

## 架构总览

```
                         ┌─────────────────────────────┐
   用户中文长程目标 ──▶   │       Orchestrator          │
                         │  (run(goal) -> RunResult)   │
                         └───────┬───────────┬─────────┘
               ┌─────────────────┘           └─────────────────┐
               ▼                                               ▼
        ┌──────────────┐                                 ┌──────────────┐
        │   Planner    │  LangGraph: 分解->结构化校验->  │   Router /   │
        │ (分解+排程)   │             重规划循环          │  NetworkGov  │
        └──────┬───────┘                                 └──────┬───────┘
               │ Plan (DAG)                                    │ RouteDecision
               ▼                                               │
        ┌──────────────┐    GoalRegistry / WorldState          │
        │   Reasoner   │  LangGraph ReAct:                    ▼
        │ (执行+反思)   │  select→reason→execute→post          ┌──────────────┐
        │              │  阻断→replan→(超限)→Reflexion         │  ToolExecutor│
        └──────┬───────┘                                      │ (前置校验+执行)│
               │ TraceStep                                      └──────┬───────┘
               ▼                                                    │
        ┌──────────────┐                                            ▼
        │ EvalHarness  │◀──── ExperienceBuffer (Reflexion 经验回流)◀─┘
        │ 四指标+归因   │
        └──────────────┘
```

---

## 快速开始

```bash
# 1) 准备环境（Python >= 3.10）
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
# 可选：对接真实大模型  pip install langchain-openai

# 2) 端到端演示（mock 后端，无需联网/API Key）
python -m examples.demo

# 3) 运行测试
pytest tests/ -q
```

最小调用：

```python
from airground_agent import Orchestrator

orch = Orchestrator()
ws = {"battery": 100, "safety_cleared": True, "edge_capable": True, "uav_online": True}
res = orch.run("帮我把这一天的素材拍好剪成短片并去除人脸", ws)
print(res.status, res.metrics.model_dump())
```

---

## 核心机制

### ① 目标寄存器 + 工具语义，根治「长程丢失/工具误调用」
- `GoalRegistry` 固化最终目标与每个子任务成功判据，提供 `is_aligned(tool, subtask)`：任何未绑定到「仍待完成子任务」的动作都会被 Reasoner 拦截。
- `ToolSpec` 显式声明 `preconditions / effects / rollback / route`。执行前由 `WorldState` 引擎做可判定校验，不满足则**阻断并升级重规划**，而非擅自替换工具。

### ② 端侧 VLA 时延优化（可度量）
`simulate_vla_latency()` 按 **结构层(MoE+蒸馏) → 量化层(AWQ/SmoothQuant) → 算子层(Flash/PagedAttention)** 三层依次叠加收益，返回时延分解，纪律：每次只动一层，p99<50ms 且 ≥20Hz 才合入。Demo 中 65ms → 28.5ms（≈35Hz）。

### ③ 端-边-云路由协议
`RouteDecision` 消息贯穿边/云；判定树：隐私高→边侧；时延预算紧且可边侧→边侧；算力高/需大模型→云端；其余默认边侧。失败回退：`escalate_cloud / degrade_edge / hold_stale / retry_backoff / edge_cache`。`NetworkGovernor` 提供 RTT/丢包注入、降级判定、stale-state hold、指数退避。

### ④ 统一评测体系
四指标：`planning_success / action_reasonableness(=1−前置违例率) / 端到端时延(p50/p95/p99) / CSAT`。`EvalHarness` 支持离线回放（逐层二分定位根因）与批量套件；`attribution` 矩阵把 badcase 现象映射到根因层；`ABTest` 提供影子模式 + 护栏自动回滚。

---

### ⑤ 机器狗接入：任务规划 -> 真机指令
`airground_agent/dog/` 把决策平台的规划产物翻译成 **ZSIBOT 机器狗**能吃的数据，两条物理通道：
- **导航**：WebSocket `ws://<ip>:10010`，一帧一个 `req_func`（厂家 demo `nav-05.py` 协议）；
- **动作**：`zsibot_control --daemon` 的 stdin 文本行（`main_sdk.cpp` 编译产物）。

```python
from airground_agent import Orchestrator

orch = Orchestrator()
res = orch.run("巡检变电站开关柜与设备区")

mission = orch.to_dog_mission(res)      # 纯编译：Plan+真实A*航点 -> 机器狗指令
print(mission.summary())                # 指令数 / 航点数 / 通道分布
print(mission.nav_frames())             # 可直接喂给导航 WebSocket 的帧
print(mission.daemon_lines())           # 可直接写入 daemon stdin 的命令
open("mission.json","w").write(mission.to_json())

rep = orch.dispatch_to_dog(res, dry_run=True)   # 干跑联调；dry_run=False 才真发
```

命令行：`python -m airground_agent.dog.cli --goal "..." --emit mission.json`
（加 `--send` 真机下发，`--infer-approach` 兜底现算导航段，详见 `--help`）。

坐标系：决策平台用 3DGS 世界系（**X-Z 水平、Y 竖直**），机器狗用 ROS 地图系（**X-Y 水平、Z 竖直**）。
`dog.yaw_deg / translate / scale` 是现场标定参数，`FrameTransform.fit_from_point_pairs()` 可用
≥2 组同名点求解。**未标定直接下发会导致航点整体偏移**。

映射纪律：只编译 `platform=DOG` 的子任务；航点必须来自真实 A* 计算（trace 或现场现算）；
映射不了的工具写入 `warnings` 并产出占位指令，绝不静默丢弃。

#### ⑤.1 本地虚拟狗联调台（接真机前必做）
`dog/devkit/` 在没有真机的开发机上，用"虚拟狗"把整条下发链路跑通——**真实 WebSocket 连接 + 真实子进程 stdin/stdout**，唯一区别是"狗"是本地模拟的：
- `virtual_dog_ws.VirtualDogWsServer`：模仿狗端 `alg_control_node` 的导航 WebSocket（`app_req`/`app_resp` 帧协议、`StandBy`/`Navigating` 状态机）；
- `virtual_zsibot_control.py` + `fake_zsibot_control.{bat,sh}`：模仿 `zsibot_control --daemon`（stdin 命令 / stdout JSON，响应形状对齐 `main_sdk.cpp` 的 `send_json_response`）。

```bash
# 一键联调：起虚拟狗 -> 连双通道 -> 跑完整 mission -> 打印执行报告
python -m airground_agent.dog.devkit.run_virtual
# 仅跑 pytest 覆盖（runner 全链路 / 适配器直调 / 急停）
pytest tests/test_dog_virtual.py -q
```
验证通过后，接真狗只需把 `ws_uri` / `daemon_exe` 指向真机地址（见 ⑤.2）。

#### ⑤.2 接真机步骤（安全分级）
⚠️ 关键前提：`zsibot_control` 是 Ubuntu 上 `make` 出的 C++ 二进制，`SdkDaemonClient` 把它当**本地子进程**拉起，因此**动作通道必须在装了 SDK 的那台 Ubuntu 上跑**；导航 WebSocket 可任意联网机器发起。最干净的部署是把整个 `airground_agent` 跑在狗所在的 Ubuntu 网关上。

1. 狗开机入网，导航节点监听 `:10010`，RTSP 在线；Ubuntu 侧 `make` 出 `zsibot_control`，`pip install websockets`。
2. 改 `config/dog.example.yaml`：`ws_uri` / `daemon_exe`(绝对路径) / `dog_ip` / `map_id` / `dry_run=false`。
3. **坐标标定（安全红线，不可跳过）**：用 ≥2 个现场同名点求变换，
   `FrameTransform.fit_from_point_pairs(world_pts, map_pts)` 后填 `yaw_deg/translate/scale`。
   未标定直接导航会让狗跑到莫名其妙的地方。
4. 分级试车：① 仅连通不动作（`transport.connect()` + `get_status` 读电量）→ ② 系留急停（`sdk_mode`→`stand_up`→立刻`emergency_stop`，空旷区）→ ③ <1m 极短航点（人待命急停，验证标定）→ ④ 完整 mission。

`Orchestrator.to_dog_mission(res)` 产出 mission；真机执行用 `sim.mode=zsibot`（`ZsibotDogAdapter` 直连）或 `orch.dispatch_to_dog(res, dry_run=False)`。

---

## 配置

默认读取 `airground_agent/config/default.yaml`，可用环境变量 `AIRGROUND_*` 覆盖（如 `AIRGROUND_LLM__BACKEND=langchain`、`AIRGROUND_LLM__API_KEY=...`）。生产环境建议用 `langchain` 后端并注入真实 API Key。

---

## 生产化对接点
- **LLM 后端**：实现 `core/llm.py:LLMBackend` 即可替换 `mock`；`LangChainBackend` 已提供。
- **真实工具**：继承 `tools/base.py:ToolSpec`，实现 `run(ws, **kwargs)` 并声明 Precondition/Effect；把 ROS / 大模型 / 算子调用塞进 `run`。
- **真机链路**：`Orchestrator` 已是纯 Python，可嵌入 C++/ROS 节点（通过进程间通信或 pybind 暴露 `run()`）。
- **机器狗**：`sim.mode=zsibot` 走 `ZsibotDogAdapter`（机器狗子任务真实下发）；离线编译用
  `Orchestrator.to_dog_mission()`，配置见 `settings.DogBridgeSettings` 与 `config/dog.example.yaml`。

## 已知限制（原型范围）
- 重规划目前整体替换计划（同构计划会保留已完成状态，仅重试阻塞节点）；真实系统需更细粒度的增量重规划与幂等执行。
- mock 工具时延为仿真值（不真实 sleep）；真实端侧 VLA 需接入实际推理引擎以采集实测时延。
- 经验回流为关键词匹配；生产可替换为向量检索 / 微调记忆。
