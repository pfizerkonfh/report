"""机器狗本地测试台（devkit）：在没有真机的机器上，用"虚拟狗"把整条下发链路跑通。

包含：
- `virtual_dog_ws.VirtualDogWsServer`：模仿狗端导航 WS 服务（app_req/app_resp 帧协议）。
- `virtual_zsibot_control.py` + 包装脚本：模仿 zsibot_control --daemon（stdin/stdout JSON）。
- `sample_mission.make_sample_mission()`：构造一份"已编译完成"的 DogMission 样例。
- `run_virtual.py`：一键起虚拟狗、连真机通道、跑完整 mission 并打印执行报告。

这些只用于开发期联调；上线接真狗时只需把 uri / daemon_exe 指向真机地址即可。
"""
from .virtual_dog_ws import VirtualDogWsServer
from .sample_mission import make_sample_mission

__all__ = ["VirtualDogWsServer", "make_sample_mission"]
