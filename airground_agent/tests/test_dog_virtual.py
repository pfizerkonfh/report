"""虚拟狗联调测试：在没有真机的机器上，用真实 WS + 真实子进程跑通整条下发链路。

覆盖：
- test_runner_e2e         : DogMissionRunner 跑样例 mission，全成功 + 遥测回传 + 协议断言
- test_adapter_navigate   : ZsibotDogAdapter 直调 route_plan，世界系航点 -> 地图系导航 + 电量回传
- test_emergency_stop     : transport.emergency_stop 双通道都 ok

这些测试证明：当把 uri / daemon_exe 换成真狗地址时，代码路径与真机一致。
"""
from __future__ import annotations

import os
import socket

import pytest

from airground_agent.dog.adapter import ZsibotDogAdapter
from airground_agent.dog.devkit.sample_mission import make_sample_mission
from airground_agent.dog.devkit.virtual_dog_ws import VirtualDogWsServer
from airground_agent.dog.frames import default_transform
from airground_agent.dog.runner import DogMissionRunner
from airground_agent.dog.transport import build_transport
from airground_agent.tools.adapters.base import AdapterContext, CancellationToken, Platform


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_exe() -> str:
    devkit = os.path.join(_root(), "airground_agent", "dog", "devkit")
    if os.name == "nt":
        return os.path.join(devkit, "fake_zsibot_control.bat")
    return os.path.join(devkit, "fake_zsibot_control.sh")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def dog_server():
    srv = VirtualDogWsServer(port=_free_port(), sim_duration=0.3)
    srv.start()
    yield srv
    srv.stop()


@pytest.fixture
def transport(dog_server):
    t = build_transport(uri=dog_server.uri, exe_path=_fake_exe(), dog_ip="127.0.0.1",
                        send_port=8081, recv_port=8080, dry_run=False)
    conn = t.connect()
    assert conn["nav"]["ok"], conn
    assert conn["sdk"]["ok"], conn
    yield t
    t.close()


def _recv_ops(srv: VirtualDogWsServer):
    return [next(iter(d.keys())) for d in srv.received if isinstance(d, dict)]


def _sent_ops(transport):
    return [(e["channel"], e["payload"]) for e in transport.sent]


def test_runner_e2e(transport, dog_server):
    mission = make_sample_mission()
    report = DogMissionRunner(transport).run(mission)
    rep = report.to_dict()
    assert rep["success"], rep
    assert rep["failed"] == 0, rep
    # 末端 get_status 遥测回传
    assert report.telemetry.get("robot_status"), "应回传真机状态"
    # 导航指令（走 WS）经虚拟狗收到；动作指令（走 daemon）记录在 transport.sent
    nav_ops = _recv_ops(dog_server)
    assert "start_nav" in nav_ops
    sdk_sent = {p for ch, p in _sent_ops(transport) if ch == "sdk"}
    assert "stand_up" in sdk_sent
    assert "sdk_mode" in sdk_sent
    # 坐标已换算到地图系（世界 [1,*,2] -> 地图 [1,-2,0]）
    navs = [d["start_nav"]["position"] for d in dog_server.received if "start_nav" in d]
    assert {"x": 1.0, "y": -2.0, "z": 0.0} in navs, navs
    assert {"x": 2.0, "y": -4.0, "z": 0.0} in navs, navs


def test_adapter_navigate(transport, dog_server):
    adapter = ZsibotDogAdapter(transport=transport, frame=default_transform(),
                               collect_seconds=0.3, nav_timeout_s=10)
    ctx = AdapterContext(tool_name="route_plan", platform=Platform.DOG,
                         args={"waypoints": [[3.0, 0.3, 6.0], [4.0, 0.3, 8.0]]},
                         token=CancellationToken())
    outcome = adapter.invoke(ctx)
    assert outcome.success, outcome.to_dict()
    # 世界 [3,*,6] -> 地图 [3,-6,0]；世界 [4,*,8] -> 地图 [4,-8,0]
    navs = [d["start_nav"]["position"] for d in dog_server.received if "start_nav" in d]
    assert {"x": 3.0, "y": -6.0, "z": 0.0} in navs, navs
    assert {"x": 4.0, "y": -8.0, "z": 0.0} in navs, navs
    # 真机遥测（电量）应被合并回写
    assert outcome.telemetry.get("battery") is not None


def test_emergency_stop(transport):
    estop = transport.emergency_stop()
    assert estop["stop_nav"]["ok"], estop
    assert estop["emergency_stop"]["ok"], estop
