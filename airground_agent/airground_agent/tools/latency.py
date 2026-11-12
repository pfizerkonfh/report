"""端侧 VLA 推理时延仿真（对应设计文档问题②）。

逐项开启优化并度量端到端时延收益，纪律：每次只动一层，p99<50ms 且 ≥20Hz 才合入。
返回 (latency_ms, breakdown) 供评测体系度量。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..config.settings import VLASettings


@dataclass
class LatencyBreakdown:
    base: float
    after_structure: float = 0.0
    after_quant: float = 0.0
    after_kernel: float = 0.0
    final: float = 0.0
    detail: Dict[str, float] = field(default_factory=dict)


def simulate_vla_latency(cfg: VLASettings) -> LatencyBreakdown:
    """按 结构 -> 量化 -> 算子 三层依次叠加收益，返回时延分解。"""
    base = cfg.baseline_latency_ms
    cur = base

    # 1) 结构层：MoE 稀疏 + 蒸馏
    if cfg.enable_moe:
        cur *= 0.72          # top-k 专家激活，FLOPs 显著下降
    if cfg.enable_distill:
        cur *= 0.88          # 蒸馏学生模型更紧凑
    after_structure = cur

    # 2) 量化层：AWQ(INT4) / SmoothQuant(W8A8) / hybrid
    if cfg.quant_mode == "awq_int4":
        cur *= 0.85          # 4-bit 权重 + 显著权重保护
    elif cfg.quant_mode == "smoothquant_w8a8":
        cur *= 0.90          # 消除反量化开销
    elif cfg.quant_mode == "hybrid":
        cur *= 0.80          # 注意力 W8A8 + 其余 INT4
    after_quant = cur

    # 3) 算子层：FlashAttention + PagedAttention
    if cfg.enable_flash_attn:
        cur *= 0.92          # 减少 HBM 读写
    if cfg.enable_paged_attn:
        cur *= 0.94          # KV 分页 + 连续批处理降尾延迟
    after_kernel = cur

    final = max(cur, 8.0)    # 物理下限
    return LatencyBreakdown(
        base=base, after_structure=after_structure, after_quant=after_quant,
        after_kernel=after_kernel, final=final,
        detail={
            "moe": 0.72 if cfg.enable_moe else 1.0,
            "distill": 0.88 if cfg.enable_distill else 1.0,
            "quant": {"awq_int4": 0.85, "smoothquant_w8a8": 0.90, "hybrid": 0.80}.get(cfg.quant_mode, 1.0),
            "flash_attn": 0.92 if cfg.enable_flash_attn else 1.0,
            "paged_attn": 0.94 if cfg.enable_paged_attn else 1.0,
        },
    )
