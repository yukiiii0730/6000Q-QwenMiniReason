#!/usr/bin/env python3
"""lm-evaluation-harness 评测封装 — 与 Qwen 官方评测方式一致。

用法：
    python3 eval/lm_eval_run.py --model_path outputs/sft_merged --tasks gsm8k_mmlu --limit 100
    python3 eval/lm_eval_run.py --model_path outputs/sft_merged --tasks math --limit 100 --output logs/lm_sft_math.json

支持的 tasks：
    gsm8k  → gsm8k_cot_zeroshot（Qwen2.5 官方 8-shot CoT）
    math   → hendrycks_math（MATH-500 标准评测）

输出格式与原有 eval/*.py 兼容：{accuracy, total, correct, details[], ...}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def ensure_lm_eval():
    """确保 lm-eval 已安装。"""
    try:
        import lm_eval  # noqa: F401
        return True
    except ImportError:
        print("📦 安装 lm-eval ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "lm-eval[api]"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True


TASK_MAP = {
    "gsm8k": "gsm8k_cot",          # 8-shot，与 Qwen 官方技术报告一致
    "math": "hendrycks_math",       # MATH-500 via --limit 71 per subtask
}


def run_lm_eval(model_path: str, task: str, limit: int, batch_size: str = "auto",
                output_dir: str = "logs/lm_eval", extra_args: list | None = None) -> dict:
    """运行 lm-eval 并返回结果 dict。"""
    lm_task = TASK_MAP.get(task, task)
    out_dir = Path(output_dir) / f"{task}_{Path(model_path).name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float16,trust_remote_code=True,ignore_mismatched_sizes=True",
        "--tasks", lm_task,
        "--batch_size", batch_size,
        "--output_path", str(out_dir.resolve()),
        "--log_samples",
        "--trust_remote_code",
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    if extra_args:
        cmd += extra_args

    print(f"  执行: {' '.join(cmd[-8:])}")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    stdout_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_lines.append(line)
    proc.wait()

    if proc.returncode != 0:
        print(f"  ❌ lm_eval 失败 (exit {proc.returncode})")
        return {}

    # 解析结果：找 results 目录下的 JSON
    result = parse_lm_eval_output(str(out_dir.resolve()), task)

    # 如果文件解析失败，从 stdout 中提取 accuracy 作为 fallback
    if not result or (not result.get("details") and result.get("accuracy", 0) == 0):
        stdout_acc = _parse_accuracy_from_stdout("".join(stdout_lines), task)
        if stdout_acc is not None:
            print(f"  ℹ️ 从 stdout 提取 accuracy={stdout_acc:.4f}（结果文件未找到）")
            result = {
                "accuracy": round(stdout_acc, 4),
                "total": limit,
                "correct": int(stdout_acc * limit),
                "lm_eval_accuracy": stdout_acc,
                "details": result.get("details", []) if result else [],
            }
    return result


def _parse_accuracy_from_stdout(stdout: str, task: str) -> float | None:
    """从 lm-eval stdout 的结果表格中解析 accuracy（fallback）。
    策略：先找表头确定 Value 列的索引，再从数据行提取对应值。
    lm-eval 表格格式（用 | 分隔）:
      | Tasks |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
      |-------|------:|------|-----:|--------|---|-----:|---|-----:|
      |task   |    ver|filter|     n|metric  |   |value |   |stderr|
    """
    lm_task = TASK_MAP.get(task, task)
    lines = stdout.splitlines()
    value_col = -1  # Value 列在 parts[] 中的索引

    for line in lines:
        # 第一步：找表头，确定 Value 列索引
        if "Metric" in line and "Value" in line:
            parts = line.split("|")
            for i, p in enumerate(parts):
                if "Value" in p.strip():
                    value_col = i
                    break
            continue

        if value_col < 0:
            continue  # 还没找到表头

        # 跳过分隔行
        if "---" in line and "|" in line:
            continue

        # 必须包含 metric 关键字
        if "exact_match" not in line and "acc,none" not in line and "acc_norm" not in line:
            continue

        # 必须包含 task 名
        if lm_task not in line and task not in line:
            continue

        # 从 value_col 提取 accuracy
        parts = line.split("|")
        if value_col < len(parts):
            p = parts[value_col].strip()
            try:
                val = float(p)
                if 0 <= val <= 1:
                    return val
            except ValueError:
                pass
    return None


def parse_lm_eval_output(output_dir: str, task: str) -> dict:
    """解析 lm-eval 输出目录，提取 accuracy 和 sample-level details。"""
    import glob as _glob

    # 使用绝对路径避免 CWD 不一致导致 glob 找不到文件
    abs_dir = str(Path(output_dir).resolve())

    # lm-eval 输出结构: output_dir/<model_name>/results_*.json 或 results.json
    result_files = _glob.glob(f"{abs_dir}/**/results_*.json", recursive=True)
    if not result_files:
        result_files = _glob.glob(f"{abs_dir}/**/results.json", recursive=True)
    if not result_files:
        result_files = _glob.glob(f"{abs_dir}/*.json")

    if not result_files:
        print(f"  ⚠️ 未找到 lm-eval 结果文件 (在 {abs_dir})")
        # 列出目录内容用于调试
        try:
            for p in Path(abs_dir).rglob("*.json"):
                print(f"    发现: {p}")
        except Exception:
            pass
        return {}

    with open(result_files[0], encoding="utf-8") as f:
        raw = json.load(f)

    # 提取主指标
    results = raw.get("results", {})
    lm_task = TASK_MAP.get(task, task)

    acc = None
    for key in [lm_task, task, f"{task}_cot_zeroshot"]:
        if key in results:
            task_result = results[key]
            # 尝试各种 metric key
            for metric in ["acc,none", "acc", "exact_match,strict-match",
                           "exact_match,flexible-extract", "acc_norm,none"]:
                if metric in task_result:
                    acc = task_result[metric]
                    break
            if acc is not None:
                break

    # 提取 sample-level details
    sample_files = _glob.glob(f"{abs_dir}/**/samples/*.jsonl", recursive=True)
    details = []
    if sample_files:
        with open(sample_files[0], encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    doc = sample.get("doc", {})
                    filtered_resps = sample.get("filtered_resps", [])
                    target = sample.get("target", "")

                    # 提取预测和 ground truth
                    pred = ""
                    if filtered_resps:
                        pred = filtered_resps[0] if isinstance(filtered_resps[0], str) else str(filtered_resps[0])

                    # 原始模型回复（用于错误分类）
                    resps = sample.get("resps", [])
                    pred_raw = ""
                    if resps:
                        pred_raw = resps[0] if isinstance(resps[0], str) else str(resps[0])

                    correct = sample.get("acc", None)
                    if correct is None:
                        # 从 metrics 推断
                        exact_match = sample.get("exact_match", None)
                        if exact_match is not None:
                            correct = bool(exact_match)

                    detail = {
                        "pred": pred,
                        "pred_raw": pred_raw,
                        "gt": target if isinstance(target, str) else str(target),
                        "correct": correct,
                    }
                    # 保留原始问题
                    if "question" in doc:
                        detail["question"] = doc["question"]
                    elif "problem" in doc:
                        detail["problem"] = doc["problem"]
                    if "answer" in doc:
                        detail["gt"] = str(doc["answer"])
                    if "level" in doc:
                        detail["level"] = doc["level"]
                    if "subject" in doc:
                        detail["subject"] = doc["subject"]

                    details.append(detail)
                except json.JSONDecodeError:
                    continue

    n = len(details) if details else 0
    correct = sum(1 for d in details if d.get("correct")) if details else 0
    accuracy = correct / n if n > 0 else (acc if acc else 0)

    return {
        "accuracy": round(accuracy, 4),
        "total": n,
        "correct": correct,
        "lm_eval_accuracy": acc,
        "details": details,
    }


def save_result(result: dict, output_path: str, model_path: str, task: str, limit: int):
    """保存结果到 JSON 文件（兼容原有格式）。"""
    result["model"] = model_path
    result["benchmark"] = task
    result["eval_method"] = "lm-evaluation-harness"
    result["limit"] = limit

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  💾 结果已保存: {output_path}")


def save_badcases(result: dict, badcase_path: str):
    """从 lm-eval 结果中提取 badcase（答错的题），保存为 JSONL。
    格式兼容 classify_errors.py 的输入要求。"""
    details = result.get("details", [])
    badcases = []
    for d in details:
        if not d.get("correct"):
            question = d.get("question") or d.get("problem") or d.get("input", "")
            badcases.append({
                "question": question,
                "pred_raw": d.get("pred_raw", d.get("pred", "")),
                "pred": d.get("pred", ""),
                "gt_raw": d.get("gt", ""),
                "gt": d.get("gt", ""),
            })

    if not badcases:
        print(f"  ℹ️ 无 badcase（全部正确），跳过保存")
        return

    Path(badcase_path).parent.mkdir(parents=True, exist_ok=True)
    with open(badcase_path, "w", encoding="utf-8") as f:
        for bc in badcases:
            f.write(json.dumps(bc, ensure_ascii=False) + "\n")
    print(f"  💾 Badcase: {len(badcases)} 条 → {badcase_path}")


def main():
    parser = argparse.ArgumentParser(description="lm-evaluation-harness 评测封装")
    parser.add_argument("--model_path", required=True, help="模型路径（本地或 HF Hub）")
    parser.add_argument("--tasks", nargs="+", required=True,
                        choices=list(TASK_MAP.keys()), help="评测任务")
    parser.add_argument("--limit", type=int, default=0, help="每个 task 评测样本数（0=全量）")
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--output_dir", default="logs/lm_eval")
    parser.add_argument("--output", default="", help="单个 task 的输出路径（覆盖默认）")
    parser.add_argument("--badcase_output", default="", help="badcase JSONL 输出路径（从 lm-eval 结果中提取错题）")
    parser.add_argument("--extra_args", nargs="*", default=[], help="额外 lm-eval 参数")
    args = parser.parse_args()

    ensure_lm_eval()

    for task in args.tasks:
        print(f"\n{'='*60}\n  lm-eval: {task} | model={args.model_path}\n{'='*60}")

        result = run_lm_eval(
            args.model_path, task, args.limit,
            args.batch_size, args.output_dir, args.extra_args,
        )

        if result:
            if args.output and len(args.tasks) == 1:
                out_path = args.output
            else:
                model_name = Path(args.model_path).name
                out_path = f"logs/lm_{task}_{model_name}.json"

            save_result(result, out_path, args.model_path, task, args.limit)

            # 保存 badcase（用于下游错误分类）
            if args.badcase_output and task == "gsm8k":
                save_badcases(result, args.badcase_output)

            acc = result.get("accuracy", 0)
            n = result.get("total", 0)
            correct = result.get("correct", 0)
            lm_acc = result.get("lm_eval_accuracy")
            print(f"  📊 {task}: {acc:.1%} ({correct}/{n})" +
                  (f"  [lm-eval原始={lm_acc:.1%}]" if lm_acc else ""))


if __name__ == "__main__":
    main()
