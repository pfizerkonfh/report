# 空地协同无人巡检系统 — 长期项目笔记

## 项目定位
- 国网变电站「空地协同」无人巡检系统。本仓库聚焦第(4)子项：**多智能体任务分配与边缘智能决策系统**。
- 「空地协同」的核心含义：任务规划完成后，**部分工作交给无人机（UAV），部分工作交给机器狗（四足机器人）**，端-边-云协同执行。物理平台维度（无人机 / 机器狗）与计算层级维度（端侧 VLA / 边缘 / 云端）是两个需要同时建模的维度。

## 核心技术栈
- Python + LangGraph；端侧 3B VLA 模型（INT4 量化，延迟目标 ≤65ms）；统一评测体系。
- 运行环境：`D:\anaconda_envs\envs\agent`（conda `agent` 虚拟环境），日常 mock 回归 18/18 通过。
- 真实模型走 OpenAI 兼容中转站 `http://www.yjsylink.cn:3000`，仅 GPT-5.x 渠道（如 gpt-5.5），需指数退避重试。
  **跑真实模型必须绕过沙箱**：沙箱会切断该域名（WinError 10054 远程主机强迫关闭连接），直连时约 50% 返回 500，
  单次全链路约 3~4 分钟。

## 关键设计决策
- 长程规划：ReAct + Reflexion + Precondition 语义校验，GoalRegistry 对齐全局目标，Action 偏离即拦截。
- 端-边-云路由：NetworkGovernor + Router 按 tier_hint 选路；VLA 端侧优先，云端兜底。
- 统一评测：planning_success / action_rationality / end_to_end_latency / p95 / CSAT / 成功判据达成。

## 平台分配维度（已实现）
- `Platform`（UAV 无人机 / DOG 机器狗 / SHARED 协同算力）与 `RouteTier`（edge/cloud）**正交**，前者物理平台、后者计算层级。
- 每个 SubTask 由 Planner 判给一个物理平台；TraceStep 记录该步的执行平台；前端 DAG/轨迹按平台着色。
- 归一化走 `core/types.py::normalize_platform()`，兼容中英文，未知回落 SHARED。

## 执行通道抽象层（ToolAdapter，2026-10-02 落地）
- **核心解耦**：`ToolSpec` = 语义契约（Precondition/Effect/Rollback/RouteHint）；`ToolAdapter` = 执行通道。
  换真机只换 Adapter，Planner/Reasoner/Router/评测零改动。目录 `tools/adapters/`。
- 四种模式：`legacy`（原 Mock，零耗时，默认，保证回归）/ `sim`（硬件在环仿真，真实时序+状态演化+故障注入）/
  `playback`（JSONL 录播回放，确定性）/ `ros2`（真机骨架，rclpy 可选导入，缺依赖优雅降级）。
- 配置入口：`SimConfig`（`tools/adapters/factory.py`）+ `settings.SimSettings` + `config/default.yaml` 的 `sim` 段。
- 真机语义四要素：**真实墙钟耗时**（time_scale 加速）/ **真实超时** / **可取消**（CancellationToken，~50ms 响应）/
  **遥测回传**（电量/里程/位置/链路/相位）。`AdapterOutcome.latency_ms`（实测）与 `sim_duration_ms`（仿真时长）分离。
- **关键坑**：真机适配器必须把工具声明的语义 Effect（`footage`/`edited`…）与物理遥测**合并回写**，
  否则后续步骤前置校验全线阻断并误触发 Reflexion。见 `AdapterContext.declared_effects`。
- **急停语义**：取消后必须立即停机（状态图 `cancelled` 终止分支），不得继续重规划/反思。

## 感知层（2026-10-02 落地，真实 3DGS 数据接入）
- 数据源 `exp/`：`showcase_assets/nav_data.json`（53×125 占据栅格 @0.2m + 85 相机轨迹 + A* 路径）、
  `nav_points.json`（6000 点）、`viewer/metrics.json`（PSNR_test 28.06、高斯 387299、98 位姿）。
- **坐标系铁律**：X-Z 水平、**Y 竖直**；`occ[ix][iz]`（ix↔X、iz↔Z）；格→世界用 `xmin/zmin + (i+0.5)*res`。
- 模块 `perception/`：`scene.py`（SceneModel + load_scene 自动发现）、`tools.py`
  （scene_inspect / map_query / route_plan / height_probe —— 全是真实计算，非 Mock）。
- 配置 `PerceptionSettings`；orchestrator 用 `scene.to_facts()` 把真实事实注入 WorldState，
  使 Precondition 可判定 `scene_known_ratio >= 0.5` 这类**真实条件**。
- 实测：真实栅格 A* 得 58 格 / 11.73m，与 build_nav.py 参考路径一致；已知率 53.4%。
- **限制**：Mock LLM 计划固定，不会主动选感知工具；真实 LLM 才会调用。
- **已验证（真实 gpt-5.5 跑测）**：Planner 会自主把 `map_query` / `route_plan` 编入计划并真正调用，
  感知层确实参与了决策，不再是摆设。
- **纯计算工具语义 `ToolSpec.compute_only`**（默认 False；`SceneBackedTool` 自动置 True）：
  这类工具无物理动作，其 observation / effects **必须来自真实计算**。执行器走
  `executor._execute_compute_only()`：先跑通道取时序/遥测（传 `declared_effects=[]` 禁止静态回写），
  再跑真实 `run()` 覆盖。**否则仿真通道会用模板 observation + 静态声明默认值**
  （如 `route_len_m=0.0`）吞掉真实 A* 结果——这是踩过的真实坑。

## 初始世界状态约定
- `Orchestrator.DEFAULT_WS`：`battery=100 / safety_cleared=True / edge_capable=True / uav_online=True /
  weather_ok=False / footage,edited,anonymized=False`。`run()` 中默认值打底、`ws_init` 覆盖。
- 这些是「作业开始前已确认的现场条件」，不是每次调用该重复手传的东西；缺了会导致 camera 等被
  前置校验无条件阻断（Web 控制台曾因此默认全败）。
- **真机部署时 `safety_cleared / battery / uav_online` 必须由安全联锁与机载遥测真实写入**，不可沿用默认。

## 对外材料（交付物清单）
- `技术报告_空地协同多智能体系统_v2.html`：技术向完整报告，每项指标带「实测 / 推演 / 占位」来源标注。
- `面试版项目介绍_空地协同多智能体系统.html`：秋招面试向，含电梯演讲、架构图、7 个技术点、
  12 题 Q&A、简历可粘贴版、3/8 分钟讲述节奏。**两份材料统一口径：无真机实测数据，推演数字不作性能论证。**
- `软考系统分析师论文_空地协同项目.html`：5 篇论文（需求建模 / 分层正交架构 / 契约与可替换构件 /
  云边端协同 / 硬件在环仿真测试），摘要+正文六段式。角色统一写为「系统分析师」，
  项目周期按 2025-03—2026-01 撰写（可替换），推演指标一律标注「估算/尚未实测」。
- `output/报告章节/`（2026-11-09）：`02-技术报告.doc` 第三/四章与 `01-工作报告.docx` 对应章节的 Markdown 成稿、
  4 张浅色 SVG 流程图、`图集预览.html` 与 `撰写说明_章节对照与数据口径.md`（含 before/after 对照）。
  2026-11-10 第八轮：两章统一「减表增段落」，正文各只保留 2 张索引型表（3-1/3-11、4-2/4-10），
  其余 15 张表与多数列表改写为长段；编号保留不复用，内容不删减。
  2026-11-11 第九轮：行文规范化——正文箭头与中文引号全部清除（仅保留 9 处 A* 算法标记）、
  清理 10 处历史残留重复段、口语词改书面语、并列斜杠改顿号；SVG 与预览页注释同步。
  **行文口径铁律（后续沿用）**：技术报告两章只述问题/方案/方法，不列实验数据；需求规格未给数值的条款不写指标；
  正文以长段论述为主，不堆叠箭头、引号与行内代码标记，图表保持精简。
  **口径铁律**：项目级样机装置/控制软件有第三方实测（W02605200142 建图 0.47cm、W02605730012 4 项符合），
  与本仓库第(4)子项原型（无真机硬件实测、电量/里程/VLA 时延为推演）必须分开表述；
  本次实跑全量回归 95 项通过（旧材料 65/65、101 为历史口径）。

## 待办 / 风险
- 系统当前仍是 **POC，不能直接上真实业务**（工具全 Mock、VLA 是公式仿真、无感知模型/安全联锁/持久化/鉴权）。
  落地路径：A 硬件在环仿真 ✅ → B 接单台真机（先机器狗）→ C 空地并行协同 + 真实端侧 VLA。
- Reasoner 目前**串行**执行（`ready_nodes[0]`），未利用无人机与机器狗可并行的物理条件 —— 阶段 C 重点。
- 中转站稳定性差（500 频发），真实模式偶发超时属中继侧问题；生产需换直连企业级模型服务。
- 端口 8777 可能残留旧服务进程（Windows 允许多进程同端口监听），重启前先 `netstat -ano | grep :8777 | grep -i listen` 确认只有一个监听。
