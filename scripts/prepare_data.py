#!/usr/bin/env python3
"""
数据集下载与预处理脚本

从 HuggingFace Hub 拉取三个数据集，转换为本地 JSON 缓存，并将路径写回 config。
已存在的缓存文件会自动跳过，不重复下载。

用法：
    python scripts/prepare_data.py                        # 默认各 50000 条
    python scripts/prepare_data.py --sft_n 10000 --dpo_n 10000
    python scripts/prepare_data.py --quick                # 快速测试，各 500 条
"""
import argparse
import json
import os
import sys
import yaml
from datasets import load_dataset


DEFAULT_INSTRUCTION = "请先进行清晰的逐步推理，再给出最终答案。"


def load_hf_token(config_path: str) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    token = cfg.get("hf_token", "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("✅ HuggingFace Token 已加载")
    else:
        print("⚠️  未配置 hf_token，如访问受限数据集请先填写 config/sft_config.yaml")
    return token


def prepare_sft(output_path: str, numina_n: int, magpie_n: int) -> None:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"⏭️  SFT 缓存已存在，跳过下载: {output_path}")
        return

    sft_rows = []

    print(f"📥 下载 NuminaMath-CoT（取 {numina_n} 条）...")
    numina = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    for x in numina.select(range(min(numina_n, len(numina)))):
        sft_rows.append({
            "instruction": DEFAULT_INSTRUCTION,
            "input":  x["problem"].strip(),
            "output": x["solution"].strip(),
        })
    print(f"   ✓ NuminaMath {len(sft_rows)} 条")

    print(f"📥 下载 Magpie-Reasoning-150K（取 {magpie_n} 条）...")
    magpie = load_dataset("Magpie-Align/Magpie-Reasoning-150K", split="train")
    before = len(sft_rows)
    for x in magpie.select(range(min(magpie_n, len(magpie)))):
        sft_rows.append({
            "instruction": x["instruction"].strip(),
            "input":  "",
            "output": x["response"].strip(),
        })
    print(f"   ✓ Magpie {len(sft_rows) - before} 条")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_rows, f, ensure_ascii=False, indent=2)
    print(f"✅ SFT 数据已保存：共 {len(sft_rows)} 条 → {output_path}")


def prepare_dpo(output_path: str, orca_n: int) -> None:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"⏭️  DPO 缓存已存在，跳过下载: {output_path}")
        return

    print(f"📥 下载 Intel/orca_dpo_pairs（取 {orca_n} 条）...")
    orca = load_dataset("Intel/orca_dpo_pairs", split="train")
    dpo_rows = []
    for x in orca.select(range(min(orca_n, len(orca)))):
        prompt   = x.get("question", "").strip()
        chosen   = x.get("chosen", "").strip()
        rejected = x.get("rejected", "").strip()
        if not prompt or not chosen or not rejected:
            continue
        dpo_rows.append({
            "prompt":   prompt,
            "chosen":   chosen,
            "rejected": rejected,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_rows, f, ensure_ascii=False, indent=2)
    print(f"✅ DPO 数据已保存：共 {len(dpo_rows)} 条 → {output_path}")


def update_configs(sft_out: str, dpo_out: str) -> None:
    """将本地数据集路径写回两个 config，并清除在线 datasets 字段。"""
    with open("config/sft_config.yaml", "r", encoding="utf-8") as f:
        sft_cfg = yaml.safe_load(f)
    sft_cfg.pop("datasets", None)
    sft_cfg["dataset_path"] = sft_out
    with open("config/sft_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(sft_cfg, f, allow_unicode=True, sort_keys=False)

    with open("config/dpo_config.yaml", "r", encoding="utf-8") as f:
        dpo_cfg = yaml.safe_load(f)
    dpo_cfg.pop("dataset", None)
    dpo_cfg["dataset_path"] = dpo_out
    with open("config/dpo_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dpo_cfg, f, allow_unicode=True, sort_keys=False)

    print("✅ config 已切换为本地缓存模式")


def main():
    parser = argparse.ArgumentParser(description="下载并预处理训练数据集")
    parser.add_argument("--config",      default="config/sft_config.yaml", help="SFT config 路径（用于读取 hf_token）")
    parser.add_argument("--output_dir",  default="data/processed",         help="本地缓存目录")
    parser.add_argument("--sft_n",       type=int, default=50000,          help="SFT 每个数据集采样条数（默认 50000）")
    parser.add_argument("--dpo_n",       type=int, default=50000,          help="DPO 数据集采样条数（默认 50000）")
    parser.add_argument("--quick",       action="store_true",              help="快速测试模式，各取 500 条")
    args = parser.parse_args()

    if args.quick:
        args.sft_n = 500
        args.dpo_n = 500
        print("⚡ 快速测试模式：每个数据集仅取 500 条")

    os.makedirs(args.output_dir, exist_ok=True)

    load_hf_token(args.config)

    sft_out = os.path.join(args.output_dir, "sft_train.json")
    dpo_out = os.path.join(args.output_dir, "dpo_train.json")

    prepare_sft(sft_out, numina_n=args.sft_n, magpie_n=args.sft_n)
    prepare_dpo(dpo_out, orca_n=args.dpo_n)
    update_configs(sft_out, dpo_out)

    print("\n📦 数据集摘要：")
    print(f"   SFT : {sft_out}")
    print(f"   DPO : {dpo_out}")


if __name__ == "__main__":
    main()
