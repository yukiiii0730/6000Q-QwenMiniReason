#!/usr/bin/env python3
"""BBH 27 子任务全量评测器（修复"只评 1/27 子任务"的根本错误）。

【为什么必要】
论文/榜单的 "BBH-X%" 都是 27 任务的均值。直接对单子任务（boolean_expressions）
报告 86% 是不能代表 BBH 真实水平的。

【两种模式】
- local：调 eval/bbh_eval.py，对每个子任务跑一次本地模型
- api  ：调 eval/bbh_api_eval.py，跑 API 模型（如 qwen2.5-7b/14b）

【输出】
- 每个子任务的明细 JSON：results/bbh_full/<tag>/<subset>.json
- 汇总 JSON：results/bbh_full/<tag>_summary.json
  包含 27 task 的 acc + 总均值 + 95% CI

用法：
    # 本地模型 (SFT 后 / DPO 后)
    python eval/bbh_full_eval.py \
        --mode local --model_path outputs/sft_merged \
        --max_samples 100 --output_dir results/bbh_full/sft

    # API 模型（baseline）
    python eval/bbh_full_eval.py \
        --mode api --api_base_url $API_URL --api_key $API_KEY \
        --model qwen2.5-7b-instruct --max_samples 100 \
        --output_dir results/bbh_full/qwen25_7b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# BBH 27 子任务标准列表
BBH_SUBSETS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


def wilson_ci(p: float, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def run_one(args: argparse.Namespace, subset: str, output_dir: Path) -> dict:
    out_path = output_dir / f"{subset}.json"
    if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
        try:
            with out_path.open() as f:
                return json.load(f)
        except Exception:
            pass

    if args.mode == "local":
        cmd = [
            sys.executable, "eval/bbh_eval.py",
            "--model_path", args.model_path,
            "--subset", subset,
            "--max_samples", str(args.max_samples),
            "--sampling_mode", args.sampling_mode,
            "--seed", str(args.seed),
            "--max_new_tokens", str(args.max_new_tokens),
            "--output", str(out_path),
        ]
        if args.load_in_4bit:
            cmd.append("--load_in_4bit")
    else:
        cmd = [
            sys.executable, "eval/bbh_api_eval.py",
            "--api_base_url", args.api_base_url,
            "--api_key", args.api_key,
            "--model", args.model,
            "--subset", subset,
            "--max_samples", str(args.max_samples),
            "--sampling_mode", args.sampling_mode,
            "--seed", str(args.seed),
            "--max_new_tokens", str(args.max_new_tokens),
            "--output", str(out_path),
        ]
        if args.enable_thinking:
            cmd.append("--enable_thinking")
            if args.thinking_budget:
                cmd += ["--thinking_budget", str(args.thinking_budget)]

    print(f"\n▶ [{subset}] {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, timeout=args.task_timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        print(f"⚠️  子任务 {subset} 超时（>{args.task_timeout}s），跳过")
        return {"subset": subset, "accuracy": None, "total": 0, "error": "timeout"}

    if rc != 0:
        print(f"⚠️  子任务 {subset} 失败 rc={rc}")
        return {"subset": subset, "accuracy": None, "total": 0, "error": f"rc={rc}"}

    if not out_path.exists():
        return {"subset": subset, "accuracy": None, "total": 0, "error": "no_output"}
    with out_path.open() as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "api"], required=True)
    ap.add_argument("--output_dir", required=True, help="单次评测的根目录")
    ap.add_argument("--max_samples", type=int, default=100, help="每个子任务的样本数（27 个 task ×100 = 2700 题）")
    ap.add_argument("--sampling_mode", choices=["first", "random", "stratified"], default="stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--task_timeout", type=int, default=900, help="单子任务最长允许时间（秒）")
    ap.add_argument("--force", action="store_true", help="强制重跑已有缓存")
    # local
    ap.add_argument("--model_path")
    ap.add_argument("--load_in_4bit", action="store_true")
    # api
    ap.add_argument("--api_base_url")
    ap.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--thinking_budget", type=int, default=0)
    # 子集选择（实验用）
    ap.add_argument("--subsets", nargs="*", default=None,
                    help="只跑这些子集；默认跑全部 27 个。可用于省钱模式（235B 只跑 5 个代表）")
    args = ap.parse_args()

    if args.mode == "local" and not args.model_path:
        raise SystemExit("--mode local 需要 --model_path")
    if args.mode == "api" and not (args.api_base_url and args.model and args.api_key):
        raise SystemExit("--mode api 需要 --api_base_url, --model 和 API key")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = args.subsets or BBH_SUBSETS
    print(f"📋 评测计划：{len(subsets)} 个 BBH 子任务，每子集 {args.max_samples} 题")

    per_task = {}
    total_correct = 0
    total_n = 0
    failed = []

    for i, subset in enumerate(subsets, 1):
        print(f"\n=== [{i}/{len(subsets)}] {subset} ===")
        res = run_one(args, subset, output_dir)
        per_task[subset] = {
            "accuracy": res.get("accuracy"),
            "total": res.get("total", 0),
            "correct": res.get("correct", 0),
        }
        if res.get("accuracy") is None:
            failed.append(subset)
            continue
        total_correct += res.get("correct", 0)
        total_n += res.get("total", 0)

    valid_accs = [v["accuracy"] for v in per_task.values() if v["accuracy"] is not None]
    macro_avg = sum(valid_accs) / len(valid_accs) if valid_accs else 0.0
    micro_avg = (total_correct / total_n) if total_n else 0.0
    ci_lo, ci_hi = wilson_ci(micro_avg, total_n)

    summary = {
        "mode": args.mode,
        "model": args.model_path or args.model,
        "n_subsets": len(subsets),
        "n_total": total_n,
        "macro_avg_accuracy": round(macro_avg, 4),
        "micro_avg_accuracy": round(micro_avg, 4),
        "micro_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "per_task": per_task,
        "failed_subsets": failed,
    }
    summary_path = Path(str(output_dir) + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"📊 BBH 全量评测汇总（{summary['model']}）")
    print(f"   子任务数: {len(subsets)}（成功 {len(subsets) - len(failed)}，失败 {len(failed)}）")
    print(f"   样本总数: {total_n}")
    print(f"   Macro Avg: {macro_avg*100:.2f}%   Micro Avg: {micro_avg*100:.2f}%   95% CI [{ci_lo*100:.1f}, {ci_hi*100:.1f}]")
    print(f"   汇总 → {summary_path}")
    if failed:
        print(f"⚠️  失败子任务: {failed}")


if __name__ == "__main__":
    main()
