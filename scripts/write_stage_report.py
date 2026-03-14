#!/usr/bin/env python3
"""生成阶段训练/评测报告（JSON + Markdown）。"""

from __future__ import annotations

import json
import importlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import argparse


def load_yaml(path: str) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    try:
        yaml = importlib.import_module("yaml")

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def parse_training_log(path: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "train_runtime": None,
        "train_steps_per_second": None,
        "train_samples_per_second": None,
        "train_loss": None,
        "last_progress": None,
    }
    p = Path(path)
    if not p.exists():
        return out
    text = p.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")

    m = re.search(
        r"\{'train_runtime':\s*'([^']+)',\s*'train_samples_per_second':\s*'([^']+)',\s*'train_steps_per_second':\s*'([^']+)',\s*'train_loss':\s*'([^']+)'",
        text,
    )
    if m:
        out["train_runtime"] = m.group(1)
        out["train_samples_per_second"] = m.group(2)
        out["train_steps_per_second"] = m.group(3)
        out["train_loss"] = m.group(4)

    prog = re.findall(r"\b(\d+/\d+)\b", text)
    if prog:
        out["last_progress"] = prog[-1]
    return out


def summarize_trainer_metrics(output_dir: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "exists": False,
        "path": None,
        "rows": 0,
        "first_loss": None,
        "last_loss": None,
        "min_loss": None,
        "last_learning_rate": None,
        "last_rewards_accuracies": None,
    }
    if not output_dir:
        return out

    p = Path(output_dir) / "trainer_log_history.jsonl"
    if not p.exists():
        return out

    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)

    out["exists"] = True
    out["path"] = str(p)
    out["rows"] = len(rows)

    def _to_float(x: Any):
        try:
            return float(x)
        except Exception:
            return None

    losses = [_to_float(r.get("loss")) for r in rows]
    losses = [x for x in losses if x is not None]
    if losses:
        out["first_loss"] = losses[0]
        out["last_loss"] = losses[-1]
        out["min_loss"] = min(losses)

    lrs = [_to_float(r.get("learning_rate")) for r in rows]
    lrs = [x for x in lrs if x is not None]
    if lrs:
        out["last_learning_rate"] = lrs[-1]

    ras = [_to_float(r.get("rewards/accuracies")) for r in rows]
    ras = [x for x in ras if x is not None]
    if ras:
        out["last_rewards_accuracies"] = ras[-1]

    return out


def summarize_result_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        return {"exists": True, "parse_error": str(e)}

    summary: dict[str, Any] = {"exists": True}
    if isinstance(data, dict):
        common_keys = [
            "accuracy",
            "acc",
            "exact_match",
            "score",
            "pass_rate",
            "overall",
            "overall_accuracy",
            "final_score",
            "correct",
            "total",
        ]
        for k in common_keys:
            v = data.get(k)
            if isinstance(v, (int, float)):
                summary[k] = v
        if "summary" in data and isinstance(data["summary"], dict):
            for k, v in data["summary"].items():
                if isinstance(v, (int, float, str)):
                    summary[f"summary.{k}"] = v
    return summary


def infer_data_plan(cfg: dict[str, Any], stage: str) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {"mode": "unknown", "sources": [], "note": "配置文件缺失或解析失败"}

    if stage in ("sft", "sft_eval") and isinstance(cfg.get("datasets"), list):
        ds = cfg.get("datasets") or []
        return {
            "mode": "hf_datasets",
            "sources": [
                {
                    "name": x.get("name"),
                    "split": x.get("split", "train"),
                    "max_samples": x.get("max_samples"),
                }
                for x in ds
                if isinstance(x, dict)
            ],
            "note": "SFT 使用 HuggingFace 在线数据集",
        }

    if stage in ("dpo", "final_eval") and isinstance(cfg.get("dataset"), dict):
        ds = cfg.get("dataset") or {}
        return {
            "mode": "hf_dataset",
            "sources": [
                {
                    "name": ds.get("name"),
                    "split": ds.get("split", "train"),
                    "max_samples": ds.get("max_samples"),
                }
            ],
            "note": "DPO 使用 HuggingFace 在线偏好数据集",
        }

    ds_path = cfg.get("dataset_path")
    if ds_path:
        return {
            "mode": "local_json",
            "sources": [{"dataset_path": ds_path}],
            "note": "使用本地缓存数据集",
        }

    return {"mode": "unknown", "sources": [], "note": "未识别到 dataset/datasets/dataset_path"}


def build_experiment_plan(cfg: dict[str, Any], stage: str, result_files: list[str]) -> dict[str, Any]:
    stage_map = {
        "sft": "SFT 监督微调",
        "dpo": "DPO 偏好优化",
        "sft_eval": "SFT 模型评测",
        "final_eval": "SFT+DPO 最终评测",
    }
    plan: dict[str, Any] = {
        "stage_goal": stage_map.get(stage, stage),
        "model_name": cfg.get("model_name") if isinstance(cfg, dict) else None,
        "base_adapter_path": cfg.get("base_adapter_path") if isinstance(cfg, dict) else None,
        "data_plan": infer_data_plan(cfg, stage),
        "notes": [],
        "result_files": [rf for rf in result_files if rf],
    }

    if stage in ("sft_eval", "final_eval"):
        plan["notes"].append("GSM8K 评测详情已记录 pred_raw 字段，可用于回看模型思考过程。")

    if isinstance(cfg, dict) and cfg.get("max_seq_length"):
        plan["notes"].append(f"最大序列长度: {cfg.get('max_seq_length')}")
    if isinstance(cfg, dict) and "load_in_4bit" in cfg:
        plan["notes"].append(f"量化加载: {cfg.get('load_in_4bit')}")
    return plan


def write_report(stage: str, config_path: str, stage_log: str, run_id: str, run_log_dir: str, result_files: list[str]) -> None:
    report_dir = Path(run_log_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(config_path)
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    log_stats = parse_training_log(stage_log)
    metrics_stats = summarize_trainer_metrics(cfg.get("output_dir") if isinstance(cfg, dict) else "")

    result_summaries: dict[str, dict[str, Any]] = {}
    for rf in result_files:
        if rf:
            result_summaries[rf] = summarize_result_json(rf)

    report: dict[str, Any] = {
        "stage": stage,
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "paths": {
            "run_log_dir": run_log_dir,
            "stage_log": stage_log,
            "config": config_path,
        },
        "config": {
            "model_name": cfg.get("model_name") if isinstance(cfg, dict) else None,
            "base_adapter_path": cfg.get("base_adapter_path") if isinstance(cfg, dict) else None,
            "dataset": cfg.get("dataset") if isinstance(cfg, dict) else None,
            "datasets": cfg.get("datasets") if isinstance(cfg, dict) else None,
            "dataset_path": cfg.get("dataset_path") if isinstance(cfg, dict) else None,
            "max_seq_length": cfg.get("max_seq_length") if isinstance(cfg, dict) else None,
            "load_in_4bit": cfg.get("load_in_4bit") if isinstance(cfg, dict) else None,
            "beta": cfg.get("beta") if isinstance(cfg, dict) else None,
            "lora": cfg.get("lora") if isinstance(cfg, dict) else None,
            "train": train_cfg,
        },
        "experiment_plan": build_experiment_plan(cfg if isinstance(cfg, dict) else {}, stage, result_files),
        "training_log_summary": log_stats,
        "training_metrics_summary": metrics_stats,
        "result_files": result_summaries,
    }

    json_path = report_dir / f"{stage}_summary.json"
    md_path = report_dir / f"{stage}_summary.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# {stage.upper()} 实验记录",
        "",
        f"- 生成时间: {report['generated_at']}",
        f"- Run ID: {run_id}",
        f"- 阶段日志: {stage_log}",
        f"- 配置文件: {config_path}",
        "",
        "## 训练配置",
        f"- model_name: {report['config']['model_name']}",
        f"- base_adapter_path: {report['config']['base_adapter_path']}",
        f"- dataset_path: {report['config']['dataset_path']}",
        f"- max_seq_length: {report['config']['max_seq_length']}",
        f"- load_in_4bit: {report['config']['load_in_4bit']}",
    ]
    if report["config"]["beta"] is not None:
        lines.append(f"- beta: {report['config']['beta']}")

    for k, v in (report["config"]["train"] or {}).items():
        lines.append(f"- train.{k}: {v}")

    lines.extend(
        [
            "",
            "## 本次实验方案",
            f"- stage_goal: {report['experiment_plan']['stage_goal']}",
            f"- data.mode: {report['experiment_plan']['data_plan']['mode']}",
            f"- data.note: {report['experiment_plan']['data_plan']['note']}",
        ]
    )

    for i, src in enumerate(report["experiment_plan"]["data_plan"]["sources"], start=1):
        lines.append(f"- data.source_{i}: {src}")

    for i, note in enumerate(report["experiment_plan"].get("notes", []), start=1):
        lines.append(f"- note_{i}: {note}")

    lines.extend(
        [
            "",
            "## 训练日志摘要",
            f"- last_progress: {log_stats.get('last_progress')}",
            f"- train_runtime: {log_stats.get('train_runtime')}",
            f"- train_steps_per_second: {log_stats.get('train_steps_per_second')}",
            f"- train_samples_per_second: {log_stats.get('train_samples_per_second')}",
            f"- train_loss: {log_stats.get('train_loss')}",
        ]
    )

    lines.extend(
        [
            "",
            "## 训练指标曲线摘要",
            f"- metrics.exists: {metrics_stats.get('exists')}",
            f"- metrics.path: {metrics_stats.get('path')}",
            f"- metrics.rows: {metrics_stats.get('rows')}",
            f"- metrics.first_loss: {metrics_stats.get('first_loss')}",
            f"- metrics.last_loss: {metrics_stats.get('last_loss')}",
            f"- metrics.min_loss: {metrics_stats.get('min_loss')}",
            f"- metrics.last_learning_rate: {metrics_stats.get('last_learning_rate')}",
            f"- metrics.last_rewards_accuracies: {metrics_stats.get('last_rewards_accuracies')}",
        ]
    )

    if result_summaries:
        lines.append("")
        lines.append("## 结果摘要")
        for rf, rs in result_summaries.items():
            lines.append(f"- {rf}:")
            for k, v in rs.items():
                lines.append(f"  - {k}: {v}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"📝 已生成阶段报告: {md_path}")
    print(f"📝 已生成阶段报告: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成阶段报告")
    parser.add_argument("stage")
    parser.add_argument("config_path")
    parser.add_argument("stage_log")
    parser.add_argument("run_id")
    parser.add_argument("run_log_dir")
    parser.add_argument("result_files", nargs="*")
    args = parser.parse_args()

    write_report(
        stage=args.stage,
        config_path=args.config_path,
        stage_log=args.stage_log,
        run_id=args.run_id,
        run_log_dir=args.run_log_dir,
        result_files=args.result_files,
    )


if __name__ == "__main__":
    main()
