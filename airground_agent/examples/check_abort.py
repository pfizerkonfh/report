"""急停自检：发起一次实时（time_scale=1）的空地协同任务，运行 2s 后下发中止，
验证长任务能被取消且连接能正常收尾。

用法：python examples/check_abort.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8777"


def post(path: str, payload: dict, timeout: int = 120):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def main() -> int:
    body = {
        "goal": "无人机航拍变电站高空设备，机器狗巡检地面表计",
        "mode": "mock",
        # time_scale=1 即实时：无人机架次真实要跑 ~48s
        "exec": {"mode": "sim", "time_scale": 1.0, "timeout_ms": 120000},
        "ws_init": {"battery": 100, "safety_cleared": True, "edge_capable": True},
    }
    resp = post("/api/run", body)
    run_id = None
    events: list = []
    buf = b""
    aborted = False
    t0 = time.time()

    for raw in resp:
        buf += raw
        while b"\n\n" in buf:
            chunk, buf = buf.split(b"\n\n", 1)
            ev, data = None, ""
            for line in chunk.decode("utf-8", "ignore").split("\n"):
                if line.startswith("event:"):
                    ev = line[6:].strip()
                elif line.startswith("data:"):
                    data += line[5:].strip()
            if ev is None:
                continue
            events.append(ev)
            if ev == "start":
                try:
                    run_id = json.loads(data).get("run_id")
                    print(f"  运行已建立：run_id={run_id}")
                except Exception:
                    pass
        # 实时跑 2.5s 后下发急停
        if (not aborted) and run_id and (time.time() - t0) > 2.5:
            aborted = True
            print(f"  [t={time.time() - t0:.1f}s] 下发中止指令…")
            try:
                r = post("/api/abort", {"run_id": run_id}, timeout=10)
                print("  中止响应:", json.load(r))
            except Exception as e:  # noqa: BLE001
                print("  中止失败:", e)
        if "end" in events:
            break

    resp.close()
    dur = time.time() - t0
    print(f"  事件序列: {' -> '.join(events)}")
    print(f"  总耗时  : {dur:.1f}s（未中止时该任务需 ~95s）")
    ok = aborted and dur < 20 and "end" in events
    print("  结论    :", "✅ 急停生效，任务被及时取消" if ok else "❌ 急停未生效")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
