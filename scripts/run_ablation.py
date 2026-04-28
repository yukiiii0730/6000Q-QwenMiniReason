#!/usr/bin/env python3
"""Ablation 实验编排器（创新点 B + 数据策略消融）

【4 组对照实验】
A. lora_baseline           — LoRA + 单段 SFT (mixed)         + Vanilla DPO (distilabel)
B. dora_curriculum         — DoRA + 课程 SFT (Stage1a→1b)    + Vanilla DPO (distilabel)
C. dora_curriculum_teacher — DoRA + 课程 SFT                  + Teacher-Guided DPO (1 类合并)
D. dora_curriculum_targeted— DoRA + 课程 SFT                  + Error-Type-Targeted DPO（创新核心）

【输出】
results/ablation/<group>/
  ├── sft_eval.json    (GSM8K + BBH-27 全量)
  ├── dpo_eval.json
  └── delta.json       相对 baseline 的 +pp 差值

results/ablation/summary_table.json  / .md   汇总表（写报告直接拷贝）

用法：
    # 跑全部 4 组
    python scripts/run_ablation.py

    # 只跑 D 组（最贵、最核心）
    python scripts/run_ablation.py --groups D

    # 快速验证（少量步数 + 少量样本）
    python scripts/run_ablation.py --quick

依赖：scripts/sft_train.py、scripts/dpo_train.py、eval/* 全部可运行
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

# ================== v4: 6 组实验配置 ==================
# A/B 验证 PEFT + 课程的基础贡献；C/D 验证 Teacher 信号 vs Targeted 信号；
# E/F 验证 DPO loss 算法改进（v4 新增创新点）。
GROUPS = {
    "A": {
        "name": "lora_baseline",
        "desc": "LoRA + 单段 SFT(mixed) + Vanilla DPO(distilabel, sigmoid)",
        "use_dora": False,
        "use_curriculum": False,
        "dpo_data": "distilabel",
        "loss_type": "sigmoid",
        "error_type_weights": None,
    },
    "B": {
        "name": "dora_curriculum",
        "desc": "DoRA + 五段课程 + Vanilla DPO(distilabel, sigmoid)",
        "use_dora": True,
        "use_curriculum": True,
        "dpo_data": "distilabel",
        "loss_type": "sigmoid",
        "error_type_weights": None,
    },
    "C": {
        "name": "dora_curriculum_teacher",
        "desc": "DoRA + 五段课程 + Teacher-Guided DPO（统一 prompt, sigmoid）",
        "use_dora": True,
        "use_curriculum": True,
        "dpo_data": "teacher_merged",
        "loss_type": "sigmoid",
        "error_type_weights": None,
    },
    "D": {
        "name": "dora_curriculum_targeted",
        "desc": "DoRA + 五段课程 + Error-Type-Targeted DPO（创新核心）",
        "use_dora": True,
        "use_curriculum": True,
        "dpo_data": "targeted",
        "loss_type": "sigmoid",
        "error_type_weights": None,
    },
    "E": {
        "name": "dora_curriculum_targeted_ipo",
        "desc": "DoRA + 五段课程 + Targeted DPO + IPO loss（防 reward hacking）",
        "use_dora": True,
        "use_curriculum": True,
        "dpo_data": "targeted",
        "loss_type": "ipo",
        "error_type_weights": None,
    },
    "F": {
        "name": "dora_curriculum_targeted_weighted",
        "desc": "DoRA + 五段课程 + Targeted DPO + Error-Type Weighted（v4 核心）",
        "use_dora": True,
        "use_curriculum": True,
        "dpo_data": "targeted",
        "loss_type": "sigmoid",
        "error_type_weights": {
            "arithmetic": 1.0,
            "reasoning_skip": 2.0,
            "setup_error": 1.5,
            "unit_or_format": 1.0,
            "extraction_error": 1.0,
            "_default": 1.0,
        },
    },
}


def run(cmd: list[str], log: Path | None = None, timeout: int | None = None) -> int:
    print(f"\n▶ {' '.join(cmd)}")
    if log:
        with log.open("w", encoding="utf-8") as f:
            try:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
                return proc.returncode
            except subprocess.TimeoutExpired:
                return -1
    try:
        proc = subprocess.run(cmd, timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        return -1


def write_yaml(path: str, data: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False)


def patch_sft_cfg(base_cfg: dict, group: dict, output_dir: str, quick: bool) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["output_dir"] = output_dir
    cfg["lora"]["use_dora"] = bool(group["use_dora"])
    if not group["use_curriculum"]:
        cfg.pop("stages", None)
        cfg.setdefault("datasets", [{"path": "data/processed/sft_train.json"}])
    if quick:
        cfg.pop("stages", None)
        cfg["datasets"] = [{"path": "data/processed/sft_train.json", "max_samples": 500}]
        cfg["train"]["max_steps"] = 50
    return cfg


def patch_dpo_cfg(base_cfg: dict, group: dict, sft_merged: str, output_dir: str,
                  data_path: str, quick: bool) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["base_adapter_path"] = sft_merged
    cfg["output_dir"] = output_dir
    cfg["lora"]["use_dora"] = bool(group["use_dora"])
    cfg["dataset_path"] = data_path
    cfg.pop("dataset", None)  # 强制走 dataset_path
    # v4: loss_type + error_type_weights
    cfg["loss_type"] = group.get("loss_type", "sigmoid")
    if group.get("error_type_weights"):
        cfg["error_type_weights"] = group["error_type_weights"]
    else:
        cfg.pop("error_type_weights", None)
    if quick:
        cfg["train"]["max_steps"] = 30
    return cfg


def eval_model(model_path: str, group_dir: Path, eval_n: int, math_n: int,
               bbh_n: int, quick: bool, tag: str) -> dict:
    """跑 GSM8K + MATH-500 + BBH 27 子任务。"""
    gsm = group_dir / f"gsm8k_{tag}.json"
    if not gsm.exists():
        rc = run([
            sys.executable, "eval/gsm8k_eval.py",
            "--model_path", model_path,
            "--max_samples", str(eval_n),
            "--output", str(gsm),
        ], timeout=3600)
        if rc != 0:
            print(f"⚠️  gsm8k 评测失败（{tag}）")

    math_path = group_dir / f"math_{tag}.json"
    if not math_path.exists():
        rc = run([
            sys.executable, "eval/math_eval.py",
            "--model_path", model_path,
            "--max_samples", str(math_n),
            "--output", str(math_path),
        ], timeout=7200)
        if rc != 0:
            print(f"⚠️  MATH-500 评测失败（{tag}）")

    bbh_dir = group_dir / f"bbh_{tag}"
    if not (Path(str(bbh_dir) + "_summary.json")).exists():
        run([
            sys.executable, "eval/bbh_full_eval.py",
            "--mode", "local",
            "--model_path", model_path,
            "--max_samples", str(bbh_n),
            "--output_dir", str(bbh_dir),
        ], timeout=7200)

    out = {}
    for key, path in [("gsm8k", gsm), ("math500", math_path)]:
        if path.exists():
            try:
                out[key] = json.load(open(path))["accuracy"]
            except Exception:
                out[key] = None
    sum_path = Path(str(bbh_dir) + "_summary.json")
    if sum_path.exists():
        try:
            out["bbh_macro"] = json.load(open(sum_path))["macro_avg_accuracy"]
        except Exception:
            out["bbh_macro"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=list(GROUPS.keys()),
                    help="跑哪几组（A/B/C/D），默认全部")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--eval_n", type=int, default=200, help="GSM8K 样本数")
    ap.add_argument("--math_n", type=int, default=500, help="MATH-500 样本数")
    ap.add_argument("--bbh_n", type=int, default=100, help="BBH 每子任务样本数")
    ap.add_argument("--root", default="results/ablation")
    ap.add_argument("--sft_cfg", default="config/sft_config.yaml")
    ap.add_argument("--dpo_cfg", default="config/dpo_config.yaml")
    ap.add_argument("--targeted_dpo_data", default="data/processed/dpo_targeted_round1.json",
                    help="D 组用：build_targeted_dpo.py 的产物")
    ap.add_argument("--teacher_dpo_data", default="data/processed/dpo_teacher_round_1.json",
                    help="C 组用：build_teacher_dpo.py 的产物")
    ap.add_argument("--distilabel_data", default="data/processed/dpo_train.json",
                    help="A/B 组用：prepare_data.py 的产物（distilabel-math-preference）")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    base_sft = yaml.safe_load(Path(args.sft_cfg).read_text())
    base_dpo = yaml.safe_load(Path(args.dpo_cfg).read_text())

    summary = {"groups": {}, "config": vars(args)}
    for g_key in args.groups:
        if g_key not in GROUPS:
            print(f"⚠️  未知组 {g_key}, 跳过")
            continue
        group = GROUPS[g_key]
        gdir = root / group["name"]
        gdir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 70)
        print(f"🧪 实验组 {g_key}: {group['name']}")
        print(f"   {group['desc']}")
        print("=" * 70)

        # 1. SFT
        sft_dir = str(gdir / "sft")
        sft_merged = str(gdir / "sft_merged")
        sft_cfg = patch_sft_cfg(base_sft, group, sft_dir, args.quick)
        cfg_path = gdir / "sft_config.yaml"
        write_yaml(str(cfg_path), sft_cfg)

        if not (Path(sft_dir) / "adapter_config.json").exists():
            rc = run([sys.executable, "scripts/sft_train.py", "--config", str(cfg_path)],
                     log=gdir / "sft.log", timeout=24 * 3600)
            if rc != 0:
                print(f"❌ {group['name']} SFT 失败")
                continue

        # merge SFT
        if not (Path(sft_merged) / "config.json").exists():
            run([sys.executable, "scripts/merge_lora.py",
                 "--adapter_path", sft_dir, "--output_path", sft_merged])

        sft_eval = eval_model(sft_merged, gdir, args.eval_n, args.math_n, args.bbh_n, args.quick, "sft")
        with (gdir / "sft_eval.json").open("w") as f:
            json.dump(sft_eval, f, ensure_ascii=False, indent=2)
        print(f"📊 [{g_key}] SFT 评测: {sft_eval}")

        # 2. DPO
        if group["dpo_data"] == "distilabel":
            dpo_data = args.distilabel_data
        elif group["dpo_data"] == "teacher_merged":
            dpo_data = args.teacher_dpo_data
        else:  # targeted
            dpo_data = args.targeted_dpo_data
        if not Path(dpo_data).exists():
            print(f"⚠️  DPO 数据不存在: {dpo_data}，跳过 DPO 阶段")
            summary["groups"][g_key] = {"name": group["name"], "sft": sft_eval, "dpo": None,
                                         "note": f"missing dpo_data: {dpo_data}"}
            continue

        dpo_dir = str(gdir / "dpo")
        dpo_merged = str(gdir / "dpo_merged")
        dpo_cfg = patch_dpo_cfg(base_dpo, group, sft_merged, dpo_dir, dpo_data, args.quick)
        cfg_path = gdir / "dpo_config.yaml"
        write_yaml(str(cfg_path), dpo_cfg)

        if not (Path(dpo_dir) / "adapter_config.json").exists():
            rc = run([sys.executable, "scripts/dpo_train.py", "--config", str(cfg_path)],
                     log=gdir / "dpo.log", timeout=24 * 3600)
            if rc != 0:
                print(f"❌ {group['name']} DPO 失败")
                summary["groups"][g_key] = {"name": group["name"], "sft": sft_eval, "dpo": None}
                continue

        if not (Path(dpo_merged) / "config.json").exists():
            run([sys.executable, "scripts/merge_lora.py",
                 "--adapter_path", dpo_dir, "--output_path", dpo_merged])

        dpo_eval = eval_model(dpo_merged, gdir, args.eval_n, args.math_n, args.bbh_n, args.quick, "dpo")
        with (gdir / "dpo_eval.json").open("w") as f:
            json.dump(dpo_eval, f, ensure_ascii=False, indent=2)
        print(f"📊 [{g_key}] DPO 评测: {dpo_eval}")

        def _delta(k):
            if not dpo_eval or not sft_eval:
                return None
            a = dpo_eval.get(k); b = sft_eval.get(k)
            if a is None or b is None:
                return None
            return a - b

        summary["groups"][g_key] = {
            "name": group["name"],
            "desc": group["desc"],
            "sft": sft_eval,
            "dpo": dpo_eval,
            "delta_gsm8k": _delta("gsm8k"),
            "delta_math": _delta("math500"),
            "delta_bbh": _delta("bbh_macro"),
        }
        with (root / "summary.json").open("w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    # ===== 写 markdown 汇总表 =====
    md_lines = ["# Ablation Summary\n",
                "| Group | Description | SFT GSM8K | SFT MATH | SFT BBH | DPO GSM8K | DPO MATH | DPO BBH | Δ GSM8K | Δ MATH | Δ BBH |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g_key in args.groups:
        if g_key not in summary["groups"]:
            continue
        gs = summary["groups"][g_key]
        sft = gs.get("sft") or {}
        dpo = gs.get("dpo") or {}

        def fmt(v):
            return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "—"

        md_lines.append(
            f"| {g_key} | {gs.get('desc', gs.get('name', ''))} | "
            f"{fmt(sft.get('gsm8k'))} | {fmt(sft.get('math500'))} | {fmt(sft.get('bbh_macro'))} | "
            f"{fmt(dpo.get('gsm8k'))} | {fmt(dpo.get('math500'))} | {fmt(dpo.get('bbh_macro'))} | "
            f"{fmt(gs.get('delta_gsm8k'))} | {fmt(gs.get('delta_math'))} | {fmt(gs.get('delta_bbh'))} |"
        )
    (root / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n✅ Ablation 汇总: {root / 'summary.json'} 与 {root / 'summary.md'}")


if __name__ == "__main__":
    main()
