机器狗综合控制面板

运行环境
- Ubuntu 22.04
- Python 3
- 厂家 SDK 安装在 `/usr/local/include` 和 `/usr/local/lib`

构建
```bash
make
```

启动
```bash
python3 robot_controller.py
```

说明
- `main_sdk.cpp` 已改为 `--daemon` 模式，GUI 通过 stdin/stdout 调用
- 机器狗 IP / 发送端口 / 接收端口会真实传入 SDK 构造器
- 日志同时写入界面和 `robot_controller.log`
- 界面只保留轮式机器狗可用的动作按钮
- 顶部可填写 RTSP 地址，默认值为 `rtsp://192.168.234.1:8554/test`
- 勾选“显示视频流”后会在右侧显示机器狗画面
- 视频流只依赖 WiFi 与 RTSP 地址，不依赖控制程序是否启动
- 视频流功能依赖 `opencv-python` 和 `Pillow`
- 日志输出区独立在底部，默认更宽更高
- 导航页保留地图、路径、单点 / 多点导航和自定义请求入口
