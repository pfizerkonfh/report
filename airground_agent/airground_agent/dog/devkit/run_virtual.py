"""一键联调：起"虚拟狗" -> 连真机通道 -> 跑完整 mission -> 打印执行报告。

这是接真狗前的本地验证。它用真实的 WebSocket 连接 + 真实的子进程 stdin/stdout
驱动 `DogMissionRunner` 与 `ZsibotDogAdapter`，唯一区别是"狗"是本地模拟的。

用法（在项目根目录）：
    python -m airground_agent.dog.devkit.run_virtual
    python -m airground_agent.dog.devkit.run_virtual --port 14567 --emit mission_out.json
    python -m airground_agent.dog.devkit.run_virtual --no-adapter   # 只跑 runner

换真狗：把 --ws-uri / 包装脚本换成真机地址与真 zsibot_control 即可（见 README）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ...dog import runner as _runner_mod  # noqa: F401  (ensure package importable)
from ..protocol import DogCommand
from ..transport import DogTransport, build_transport
from ..adapter import ZsibotDogAdapter
from ..frames import default_transform
from ...tools.adapters.base import AdapterContext, CancellationToken, Platform
from .virtual_dog_ws import VirtualDogWsServer
from .sample_mission import make_sample_mission


HERE = os.path.dirname(os.path.abspath(__file__))


def fake_daemon_exe() -> str:
    if os.name == "nt":
        return os.path.join(HERE, "fake_zsibot_control.bat")
    return os.path.join(HERE, "fake_zsibot_control.sh")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="机器狗虚拟联调台")
    p.add_argument("--port", type=int, default=14567, help="虚拟狗导航 WS 端口")
    p.add_argument("--ws-uri", default="", help="覆盖 WS 地址（接真狗时用）")
    p.add_argument("--sim-duration", type=float, default=0.6,
                   help="虚拟狗单段导航仿真耗时（秒）")
    p.add_argument("--goal", default="", help="任务目标描述（仅展示用）")
    p.add_argument("--emit", default="", help="把 mission 导出为 JSON 文件")
    p.add_argument("--no-adapter", action="store_true",
                   help="跳过 ZsibotDogAdapter 直调测试（只跑 runner）")
    p.add_argument("--dog-ip", default="127.0.0.1", help="动作通道狗 IP（虚拟用本地）")
    args = p.parse_args(argv)

    uri = args.ws_uri or f"ws://127.0.0.1:{args.port}"
    print(f"[devkit] 启动虚拟狗导航服务 @ {uri} (sim_duration={args.sim_duration}s)")
    srv = VirtualDogWsServer(port=args.port, sim_duration=args.sim_duration)
    try:
        srv.start()
    except RuntimeError as e:
        print(f"[devkit] 虚拟狗启动失败: {e}")
        return 2

    exe = fake_daemon_exe()
    transport = build_transport(uri=uri, exe_path=exe, dog_ip=args.dog_ip,
                               send_port=8081, recv_port=8080, dry_run=False)
    print(f"[devkit] 连接双通道（nav WS + sdk daemon @ {exe}）...")
    conn = transport.connect()
    print(f"          nav : {conn.get('nav')}")
    print(f"          sdk : {conn.get('sdk')}")
    if not (conn.get("nav", {}).get("ok") and conn.get("sdk", {}).get("ok")):
        print("[devkit] 通道连接失败，退出")
        srv.stop()
        return 3

    mission = make_sample_mission(goal=args.goal or "虚拟狗联调巡检")
    if args.emit:
        with open(args.emit, "w", encoding="utf-8") as f:
            f.write(mission.to_json())
        print(f"[devkit] mission 已导出 -> {args.emit}")

    print(f"\n[devkit] 运行 mission（{len(mission.commands)} 条指令）...")
    report = _runner_mod.DogMissionRunner(transport).run(mission)
    rep = report.to_dict()
    print(f"          成功={rep['success']} 指令={rep['commands']} "
          f"成功={rep['succeeded']} 失败={rep['failed']} 耗时={rep['total_ms']:.0f}ms")
    for r in rep["results"]:
        flag = "OK " if r["ok"] else "ERR"
        extra = f" nav={r['nav_state']}" if r["nav_state"] else ""
        print(f"   [{flag}] {r['op']:<14} {r['channel']:<4} {r['latency_ms']:.0f}ms{extra}")
    if rep.get("telemetry"):
        print(f"           末态遥测: {json.dumps(rep['telemetry'], ensure_ascii=False)}")

    if not args.no_adapter:
        print("\n[devkit] 直调 ZsibotDogAdapter（route_plan，世界系航点 -> 地图系导航）...")
        adapter = ZsibotDogAdapter(transport=transport, frame=default_transform(),
                                   collect_seconds=0.5, nav_timeout_s=10)
        ctx = AdapterContext(tool_name="route_plan", platform=Platform.DOG,
                             args={"waypoints": [[3.0, 0.3, 6.0], [4.0, 0.3, 8.0]]},
                             token=CancellationToken())
        outcome = adapter.invoke(ctx)
        od = outcome.to_dict()
        print(f"           status={od['status']} success={od['success']} "
              f"obs={od['observation']}")
        batt = od["telemetry"].get("battery")
        print(f"           回传电量={batt} 导航末态={od['telemetry'].get('nav_states')}")
        assert outcome.success, "ZsibotDogAdapter 直调应成功"
        assert batt is not None, "应回传电量遥测"

    print("\n[devkit] 急停测试（先停导航再软急停）...")
    estop = transport.emergency_stop()
    print(f"           stop_nav.ok={estop['stop_nav'].get('ok')} "
          f"emergency_stop.ok={estop['emergency_stop'].get('ok')}")

    # 断言指令确实发出：导航走 WS（srv.received），动作走 daemon（transport.sent）
    recv_ops = [next(iter(d.keys())) for d in srv.received if isinstance(d, dict)]
    print(f"\n[devkit] 虚拟狗(导航WS)共收到 {len(recv_ops)} 条 req_func：{recv_ops}")
    assert any(o == "start_nav" for o in recv_ops), "虚拟狗应收到 start_nav"
    sdk_sent = {e["payload"] for e in transport.sent if e["channel"] == "sdk"}
    print(f"[devkit] 动作通道(daemon)发出命令：{sorted(sdk_sent)}")
    assert "stand_up" in sdk_sent, "动作通道应发出 stand_up"
    assert "sdk_mode" in sdk_sent, "动作通道应发出 sdk_mode"

    print("\n[devkit] 全部通过 ✅  —— 整条下发链路（协议/传输/执行/适配）已本地跑通。")
    srv.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
