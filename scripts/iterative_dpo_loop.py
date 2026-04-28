#!/usr/bin/env python3
"""Iterative DPO 循环编排器（GPU 环境运行）

每一轮 (round k) 流程：
  1. 用当前 student 在 GSM8K (subset) 上推理 → 收集 badcases JSONL
  2. （本地 / 远端）调 Teacher 生成 chosen → 构造 dpo_teacher_round_k.json
  3. 用该数据继续 DPO 训练（在前一轮 adapter 上微调小步数）
  4. 评测，写日志

由于 chosen 生成需要 API 而本地无 GPU，可以选择：
  - "online"  ：本轮 chosen 在 GPU 服务器上现场调 API（默认，最方便）
  - "offline" ：本轮 badcases 上传回本地、本地生成 chosen，再 rsync 回 GPU 继续训练

【输入】
  --rounds N                   迭代轮数（推荐 2~3）
  --steps_per_round 200        每轮 DPO 训练步数
  --base_dpo_config            起始 DPO config（首轮基于此训练）
  --eval_subset gsm8k|bbh      用哪个评测集采集 badcase（默认 gsm8k）
  --eval_n 200                 评测样本数

【输出】
  outputs/dpo_iter/round_{k}/  每轮 adapter
  data/processed/dpo_teacher_round_{k}.json  每轮 DPO 数据
  logs/iterative_dpo/summary.json  汇总指标曲线

注意：本脚本依赖 unsloth/transformers，必须在 GPU 服务器或 Colab 上运行。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def run(cmd: list[str], cwd: str | None = None) -> int:
    print(f"\n▶ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd)
    return proc.returncode


def write_yaml(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Iterative DPO 循环编排器（GPU 环境）")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps_per_round", type=int, default=200)
    parser.add_argument("--base_dpo_config", default="config/dpo_config.yaml")
    parser.add_argument("--base_adapter", default="outputs/sft", help="第 1 轮的起始 adapter")
    parser.add_argument("--root_output", default="outputs/dpo_iter")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--eval_subset", choices=["gsm8k", "bbh"], default="gsm8k")
    parser.add_argument("--eval_n", type=int, default=200)
    parser.add_argument("--bbh_subset", default="boolean_expressions")
    parser.add_argument("--teacher_workers", type=int, default=4)
    parser.add_argument("--samples_per_round", type=int, default=1000)
    parser.add_argument("--mode", choices=["online", "offline"], default="online",
                        help="online=GPU 上直接调 Teacher API；offline=只到 badcase 阶段，提示用户本地生成")
    args = parser.parse_args()

    Path(args.root_output).mkdir(parents=True, exist_ok=True)
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    Path("logs/iterative_dpo").mkdir(parents=True, exist_ok=True)

    summary = {"rounds": [], "config": vars(args)}

    base_cfg = yaml.safe_load(Path(args.base_dpo_config).read_text(encoding="utf-8"))

    current_adapter = args.base_adapter

    for k in range(1, args.rounds + 1):
        print(f"\n========== Iterative DPO Round {k}/{args.rounds} ==========")
        round_dir = Path(args.root_output) / f"round_{k}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_data = Path(args.data_dir) / f"dpo_teacher_round_{k}.json"

        # === 0. 先把 adapter merge 成完整模型（eval 脚本只接受 model_path） ===
        merged_path = round_dir / "merged_for_eval"
        if not (merged_path / "config.json").exists():
            print(f"🔧 合并 adapter → {merged_path}")
            mrc = run([
                sys.executable, "scripts/merge_lora.py",
                "--adapter_path", current_adapter,
                "--output_path", str(merged_path),
            ])
            if mrc != 0:
                print(f"❌ adapter 合并失败 (round {k})")
                break

        # === 1. 用合并后的模型评测 → 收集 badcases ===
        bad_path = round_dir / f"{args.eval_subset}_badcases.jsonl"
        result_path = round_dir / f"{args.eval_subset}_eval.json"

        if args.eval_subset == "gsm8k":
            cmd = [
                sys.executable, "eval/gsm8k_eval.py",
                "--model_path", str(merged_path),
                "--max_samples", str(args.eval_n),
                "--output", str(result_path),
                "--badcase_output", str(bad_path),
            ]
        else:
            cmd = [
                sys.executable, "eval/bbh_eval.py",
                "--model_path", str(merged_path),
                "--subset", args.bbh_subset,
                "--max_samples", str(args.eval_n),
                "--output", str(result_path),
                "--badcase_output", str(bad_path),
            ]
        if run(cmd) != 0:
            print(f"⚠️  评测失败，跳过本轮 (round {k})")
            break

        if not bad_path.exists() or bad_path.stat().st_size == 0:
            print(f"⚠️  无 badcase（模型已答对全部题），提前结束")
            break

        # === 2. 调 Teacher 生成 chosen ===
        if args.mode == "offline":
            print(f"💡 离线模式：请将以下文件下载到本地，运行 build_teacher_dpo 后回传 → {bad_path}")
            print(f"   预期产物路径: {round_data}")
            print("   完成后再次运行本脚本会跳过本步")
            if not round_data.exists():
                break
        else:
            cmd = [
                sys.executable, "scripts/build_teacher_dpo.py",
                "--rejected_jsonl", str(bad_path),
                "--rejected_kind", "badcase",
                "--output", str(round_data),
                "--max_samples", str(args.samples_per_round),
                "--workers", str(args.teacher_workers),
            ]
            if run(cmd) != 0:
                print(f"❌ Teacher 生成失败，停止迭代")
                break

        # === 3. 在前一轮 adapter 基础上继续 DPO（base 用合并后的完整模型） ===
        round_cfg = copy.deepcopy(base_cfg)
        round_cfg["base_adapter_path"] = str(merged_path)
        round_cfg["output_dir"] = str(round_dir / "dpo")
        round_cfg["dataset_path"] = str(round_data)
        round_cfg.pop("dataset", None)  # 强制走 dataset_path
        round_cfg["train"]["max_steps"] = int(args.steps_per_round)
        round_cfg["train"]["warmup_steps"] = max(20, int(args.steps_per_round * 0.1))

        cfg_path = round_dir / "dpo_config.yaml"
        write_yaml(str(cfg_path), round_cfg)

        cmd = [sys.executable, "scripts/dpo_train.py", "--config", str(cfg_path)]
        if run(cmd) != 0:
            print(f"❌ DPO 训练失败，停止迭代")
            break

        next_adapter = str(round_dir / "dpo")
        current_adapter = next_adapter

        # 记录摘要
        try:
            metrics = json.loads(result_path.read_text(encoding="utf-8"))
            acc = metrics.get("accuracy", None)
        except Exception:
            acc = None
        summary["rounds"].append({
            "round": k,
            "input_adapter": current_adapter if k == 1 else summary["rounds"][-1]["output_adapter"],
            "output_adapter": next_adapter,
            "eval_acc_before_train": acc,
            "data_path": str(round_data),
        })

        with open("logs/iterative_dpo/summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n📑 Iterative DPO 汇总:")
    for r in summary["rounds"]:
        print(f"   Round {r['round']}: acc(before)={r['eval_acc_before_train']} → adapter={r['output_adapter']}")
    print("\n✅ 全部轮次完成。最新 adapter:", current_adapter)


if __name__ == "__main__":
    main()
