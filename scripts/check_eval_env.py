#!/usr/bin/env python3
"""检测本机是否适合跑「合并后模型」的 GSM8K / MATH / BBH 本地评测（无需 CUDA）。"""
from __future__ import annotations

import sys


def _fmt_gib(n: float) -> str:
    return f"{n:.1f} GiB" if n >= 1.0 else f"{n*1024:.0f} MiB"


def main() -> int:
    try:
        import torch
    except ImportError as e:
        print("未安装 torch，无法评测。请先: pip install torch", file=sys.stderr)
        return 1

    print("PyTorch", torch.__version__)
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    print("CUDA :", "可用" if has_cuda else "不可用", f"（{torch.cuda.get_device_name(0)}）" if has_cuda else "")

    if has_mps:
        print("MPS  (Apple GPU): 可用")
    else:
        print("MPS  (Apple GPU): 不可用 / 非 macOS 或未启用")

    mem_ok = None
    try:
        import psutil

        v = psutil.virtual_memory()
        mem_ok = v.available / (1024**3)
        print("可用系统内存(约):", _fmt_gib(v.available / (1024**3)))
    except ImportError:
        print("可用系统内存: （安装 psutil 可显示）  pip install psutil")

    # 4bit 走 bitsandbytes，仅 NVIDIA+CUDA 上有意义；勿在 Mac 上 import bnb（会报警/异常）
    bnb_4bit_eval = bool(has_cuda)
    print(
        "4bit 评测（--load-in-4bit）:",
        "可用（需 CUDA + bitsandbytes）" if bnb_4bit_eval else "本机不可用（仅 CUDA；Apple 请用默认半精度+MPS）",
    )

    # 结论
    print()
    if has_cuda:
        print("结论: 有 NVIDIA GPU，半精度与可选 4bit 均可，一般可本地评测 1.5B。")
    elif has_mps:
        print("结论: 可用 Apple MPS + float16 跑 1.5B 评测（已写进 eval/model_loader.py）；")
        print("     不要用 --load-in-4bit。全量仍耗时，可先 bash run_eval_local.sh --quick。")
    else:
        print("结论: 无 GPU，仅 CPU 可跑，但 1.5B 全量评测（尤其 MATH+BBH）会非常慢；")
        if mem_ok is not None and mem_ok < 12:
            print("     本机可用内存 < ~12 GiB 时，不建议在 CPU 上跑全量，建议用 --quick 或云端/GPU 评。")
        else:
            print("     若内存较充裕，可用较小 --eval_n 试跑，或装 GPU 再评。")

    from pathlib import Path

    for label, p in (
        ("SFT 合并", Path("outputs/sft_merged/config.json")),
        ("最终合并", Path("outputs/merged/config.json")),
    ):
        if p.is_file():
            print(f"已找到 {label} 模型: {p.parent}/")
        else:
            print(f"未找到 {label} 模型（需先同步训练产物）: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
