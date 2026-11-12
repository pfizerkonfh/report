"""命令行入口：决策平台跑一轮规划 -> 导出/下发机器狗任务。

典型用法（先离线校验，再上真机）
--------------------------------
```bash
# 1) 离线编译：不连狗，只产出机器狗能吃的数据
python -m airground_agent.dog.cli --goal "巡检变电站开关柜" --emit mission.json

# 2) 干跑联调：走完整下发流程但只记录不真发（dry-run）
python -m airground_agent.dog.cli --goal "巡检变电站开关柜" --dry-run-send

# 3) 真机下发（Ubuntu 现场，连着机器狗 WiFi）
python -m airground_agent.dog.cli --goal "巡检变电站开关柜" \
    --send --ws-uri ws://192.168.234.234:10010 --daemon-exe ./zsibot_control
```

坐标系：默认只有轴映射（3DGS X-Z 水平 -> 狗地图 X-Y 水平）。
真机下发前必须用现场同名点标定 `--yaw-deg / --translate / --scale`。
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from ..config.settings import DogBridgeSettings, Settings, SimSettings
from ..orchestrator import Orchestrator


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dog-bridge",
        description="把决策平台的任务规划编译为机器狗指令序列，并可选下发真机",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--goal", default="巡检变电站开关柜与设备区，采集近距影像并回传",
                   help="任务目标（交给决策平台规划）")
    p.add_argument("--mode", default="legacy", choices=["legacy", "sim", "playback"],
                   help="决策平台的执行通道模式（默认 legacy：Mock 规划，秒级）")
    p.add_argument("--config", default="", help="配置文件（yaml）")
    p.add_argument("--scene-dir", default="", help="3DGS 场景产物目录（含 nav_data.json）")
    p.add_argument("--emit", default="", help="把机器狗 mission 导出为 JSON 文件")
    p.add_argument("--send", action="store_true", help="真机下发（需 --ws-uri / --daemon-exe）")
    p.add_argument("--dry-run-send", action="store_true",
                   help="走完整下发流程但只记录不真发（安全阀）")
    # ---- 机器狗通道 ----
    p.add_argument("--ws-uri", default="", help="导航 WebSocket 地址")
    p.add_argument("--daemon-exe", default="", help="zsibot_control 可执行程序路径")
    p.add_argument("--dog-ip", default="", help="机器狗本体 IP")
    p.add_argument("--map-id", default="", help="狗端定位地图名")
    p.add_argument("--path-id", default="", help="狗端已录制路径 ID（给了走多点导航）")
    # ---- 坐标系标定 ----
    p.add_argument("--yaw-deg", type=float, default=0.0, help="世界系到地图系的旋转角（度）")
    p.add_argument("--translate", default="", help="地图系平移，逗号分隔：tx,ty,tz")
    p.add_argument("--scale", type=float, default=1.0, help="尺度")
    p.add_argument("--infer-approach", action="store_true",
                   help="兜底：机器狗子任务没有显式导航时，用真实场景现算一段前往作业区的路径")
    return p


def _settings_from_args(args) -> Settings:
    if args.config:
        st = Settings.from_file(args.config)
    else:
        st = Settings()
    st.sim = SimSettings(mode=args.mode)
    if args.scene_dir:
        st.perception.data_dir = args.scene_dir
    dog = DogBridgeSettings(
        ws_uri=args.ws_uri or st.dog.ws_uri,
        daemon_exe=args.daemon_exe or st.dog.daemon_exe,
        dog_ip=args.dog_ip or st.dog.dog_ip,
        map_id=args.map_id or st.dog.map_id,
        path_id=args.path_id or st.dog.path_id,
        dry_run=not args.send,
        yaw_deg=args.yaw_deg,
        scale=args.scale,
    )
    if args.translate:
        try:
            dog.translate = [float(v) for v in args.translate.split(",")]
        except ValueError:
            print(f"[warn] --translate 解析失败，忽略：{args.translate}", file=sys.stderr)
    st.dog = dog
    return st


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    st = _settings_from_args(args)

    orch = Orchestrator(st)
    print(f"[1/3] 决策平台规划中（mode={st.sim.mode}）... goal={args.goal}")
    result = orch.run(args.goal)
    plan_n = len(result.plan.subtasks) if result.plan else 0
    dog_n = sum(1 for s in (result.plan.subtasks.values() if result.plan else [])
                if s.platform.value == "dog")
    print(f"      规划完成：子任务 {plan_n} 个，其中机器狗平台 {dog_n} 个，"
          f"状态={result.status.value}")

    print("[2/3] 编译机器狗指令序列 ...")
    cfg = orch.dog_compile_cfg()
    if args.infer_approach:
        cfg.infer_approach = True
    try:
        mission = orch.to_dog_mission(result, cfg=cfg)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    summary = mission.summary()
    print(f"      mission={summary['mission_id']} 指令 {summary['commands']} 条 "
          f"{summary['by_channel']} 航点 {summary['waypoints']} 个")
    print(f"      航点来源：{mission.waypoint_source or '（无导航段）'}")
    for w in mission.warnings:
        print(f"      [warn] {w}")

    problems = mission.validate()
    if problems:
        print("[error] 指令校验未通过，拒绝下发：", file=sys.stderr)
        for p_ in problems:
            print(f"      - {p_}", file=sys.stderr)
        return 3

    if args.emit:
        with open(args.emit, "w", encoding="utf-8") as f:
            f.write(mission.to_json())
        print(f"[3/3] 已导出 -> {args.emit}")
    else:
        print("[3/3] 指令预览：")
        for c in mission.commands:
            tag = f"  {c.id:<6}{c.describe()}"
            print(tag + (f"    # {c.subtask_id}" if c.subtask_id else ""))

    if args.send or args.dry_run_send:
        dry = not args.send
        print(f"\n[下发] {'DRY-RUN（不真发）' if dry else '真机下发'}")
        report = orch.dispatch_to_dog(result, dry_run=dry)
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.success else 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
