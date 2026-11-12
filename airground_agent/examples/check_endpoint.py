"""真实模型连通性自检：仅发一个最小请求，验证 Key / 地址 / 模型是否可用。

运行：  python -m examples.check_endpoint
或：    AIRGROUND_CONFIG=airground_agent/config/real_model.yaml python -m examples.check_endpoint

返回：
  0 = 连通正常
  1 = 配置缺失
  2 = 连通异常（已打印诊断，按提示修改 config/real_model.yaml 或环境变量）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console

from airground_agent.config.settings import Settings
from airground_agent.core.llm import build_backend

console = Console()
CONFIG = Path(__file__).resolve().parent.parent / "config" / "real_model.yaml"


def main() -> int:
    cfg_path = os.environ.get("AIRGROUND_CONFIG")
    path = Path(cfg_path) if cfg_path else CONFIG
    if not path.exists():
        console.print(f"[red]未找到真实模型配置：{path}[/red]")
        console.print("请新建 config/real_model.yaml，填入中转站地址与 API Key。")
        return 1

    settings = Settings.from_file(str(path))
    llm = settings.llm
    console.print(f"backend={llm.backend} model={llm.model_name} base_url={llm.base_url}")

    if llm.backend == "mock" or not llm.base_url:
        console.print("[yellow]llm.backend 不是 openai 或未设置 base_url，跳过真实连通性检查。[/yellow]")
        return 1

    backend = build_backend(**llm.model_dump())
    console.print("[cyan]正在发送自检请求（ping）...[/cyan]")
    try:
        backend.ping()
    except RuntimeError as e:
        console.print(f"[red]连通异常：{e}[/red]")
        return 2

    console.print("[green]✅ 连通正常：中转站已接受该 Key 与模型，可运行 python -m examples.run_real[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
