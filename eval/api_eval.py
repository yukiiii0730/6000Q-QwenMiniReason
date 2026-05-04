#!/usr/bin/env python3
"""DashScope API 评测脚本 — 用 Qwen 官方 API 评测 1.5B / 7B 模型。

用法：
    python3 eval/api_eval.py                     # 跑全部（1.5B + 7B × GSM8K + MATH，n=200）
    python3 eval/api_eval.py --models 1.5b       # 只跑 1.5B
    python3 eval/api_eval.py --benchmarks gsm8k   # 只跑 GSM8K
    python3 eval/api_eval.py --max_samples 50     # 调整样本数

特性：
    - 断点续跑：每 10 条自动保存，中断后重跑自动跳过已完成
    - 不覆盖已有结果：如果输出文件已有完整结果则跳过
    - 实时进度打印
    - API key 从 .env 读取
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import List

import requests
from tqdm import tqdm

# ── .env 加载 ─────────────────────────────────────────────────
def load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

load_dotenv()

DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
CHECKPOINT_EVERY = 10

MODEL_MAP = {
    "1.5b": "qwen2.5-1.5b-instruct",
    "7b": "qwen2.5-7b-instruct",
}


# ── 答案抽取（复用 gsm8k_eval.py / math_eval.py 逻辑）────────
def extract_number(text: str) -> str:
    if not text:
        return ""
    s = text.replace(",", "")
    m = re.findall(r"\\boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}", s)
    if m:
        return m[-1]
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return m[-1]
    for pat in [
        r"答案\s*[:：是为等于]+\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"final answer\s*(?:is|:)?\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"the answer is\s*\$?\s*(-?\d+(?:\.\d+)?)",
    ]:
        m = re.findall(pat, s, flags=re.IGNORECASE)
        if m:
            return m[-1]
    tail = s[-300:]
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
    if nums:
        return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else ""


_BOXED_RE = re.compile(r"\\boxed\s*{")


def extract_boxed(text: str) -> str:
    if not text:
        return ""
    last = ""
    for m in _BOXED_RE.finditer(text):
        i = m.end()
        depth = 1
        out = []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
                out.append(c)
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
                out.append(c)
            else:
                out.append(c)
            i += 1
        last = "".join(out).strip()
    return last


def extract_math_answer(text: str) -> str:
    if not text:
        return ""
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    for pat in [
        r"答案\s*[:：是为等于]+\s*\$?\s*([^\n。.,，]+?)(?:[\.。\n]|$)",
        r"final answer\s*(?:is|:)?\s*\$?\s*([^\n.]+?)(?:[\.\n]|$)",
        r"the answer is\s*\$?\s*([^\n.]+?)(?:[\.\n]|$)",
    ]:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return m[-1].strip(" .。,，:：$\\")
    tail = text[-200:]
    m = re.findall(r"\$([^$]+)\$", tail)
    if m:
        return m[-1].strip()
    return ""


def _strip_string(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace("$", "").replace(" ", "").replace("\\\\", "\\")
    s = s.replace("\\!", "").replace("\n", "").replace("\\,", "")
    if "=" in s and len(s.split("=")[-1]) > 0:
        s = s.split("=")[-1]
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\textbf\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{(.*?)\}", r"\1", s)
    s = s.rstrip(".")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\frac([0-9])([0-9])", r"\1/\2", s)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt([0-9a-zA-Z])", r"sqrt(\1)", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\circ", "")
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\&", "&")
    return s


def _to_float(s: str):
    if s is None:
        return None
    s = str(s).strip()
    try:
        return float(s)
    except Exception:
        pass
    m = re.match(r"^(-?)(\d+)\s*/\s*(\d+)$", s)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        num, den = int(m.group(2)), int(m.group(3))
        if den != 0:
            return sign * num / den
    return None


def is_equiv(pred: str, gt: str) -> bool:
    if pred is None or gt is None:
        return False
    p = _strip_string(pred)
    g = _strip_string(gt)
    if not p or not g:
        return False
    if p == g:
        return True
    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf is not None:
        if abs(pf - gf) < 1e-4 or (gf != 0 and abs((pf - gf) / gf) < 1e-4):
            return True
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
        ep = parse_expr(p.replace("^", "**"), evaluate=True)
        eg = parse_expr(g.replace("^", "**"), evaluate=True)
        if sympy.simplify(ep - eg) == 0:
            return True
    except Exception:
        pass
    return False


# ── 数据集加载 & 抽样 ─────────────────────────────────────────
def load_gsm8k():
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split="test")


def load_math500():
    from datasets import load_dataset
    try:
        return load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        return load_dataset("hendrycks/competition_math", split="test")


def select_eval_subset(ds, max_samples: int, seed: int = 42):
    n = len(ds)
    if max_samples <= 0 or max_samples >= n:
        return ds, list(range(n))
    rng = random.Random(seed)
    levels_field = "level" if "level" in ds.column_names else None
    if not levels_field:
        # GSM8K：按题目长度分层
        questions = ds["question"] if "question" in ds.column_names else ds["input"]
        lengths = [len(str(q)) for q in questions]
        sorted_indices = sorted(range(n), key=lambda i: lengths[i])
        bins = 5
        groups = [sorted_indices[i * n // bins:(i + 1) * n // bins] for i in range(bins)]
        base = max_samples // bins
        rem = max_samples % bins
        picked = []
        for i, g in enumerate(groups):
            k = base + (1 if i < rem else 0)
            if k > 0 and g:
                picked.extend(rng.sample(g, min(k, len(g))))
        if len(picked) < max_samples:
            remaining = [i for i in range(n) if i not in set(picked)]
            picked.extend(rng.sample(remaining, max_samples - len(picked)))
        indices = sorted(picked[:max_samples])
        return ds.select(indices), indices
    # MATH：按 level 分层
    by_level: dict = {}
    for i, lv in enumerate(ds[levels_field]):
        by_level.setdefault(str(lv), []).append(i)
    n_levels = len(by_level)
    base = max_samples // n_levels
    rem = max_samples % n_levels
    picked: list[int] = []
    for j, (lv, idxs) in enumerate(sorted(by_level.items())):
        k = base + (1 if j < rem else 0)
        if k > 0 and idxs:
            picked.extend(rng.sample(idxs, min(k, len(idxs))))
    if len(picked) < max_samples:
        remaining = [i for i in range(n) if i not in set(picked)]
        picked.extend(rng.sample(remaining, max_samples - len(picked)))
    indices = sorted(picked[:max_samples])
    return ds.select(indices), indices


# ── Prompt 构建 ───────────────────────────────────────────────
def build_gsm8k_prompt(question: str) -> str:
    return (
        "请一步步推理后给出最终答案，并把最终答案放在 \\boxed{} 中。\n\n"
        "题目：" + question
    )


def build_math_prompt(problem: str) -> str:
    return (
        "请一步步推理后给出最终答案，并把最终答案放在 \\boxed{} 中。\n\n"
        "题目：" + problem
    )


# ── API 调用 ──────────────────────────────────────────────────
def call_api(api_key: str, model: str, prompt: str, max_retries: int = 3) -> str:
    """调用 DashScope OpenAI-compatible API。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    url = f"{DASHSCOPE_BASE}/chat/completions"

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 30)
                print(f"  [rate limit] 等待 {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 3
                print(f"  [retry {attempt+1}] {e}, 等待 {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [ERROR] API 调用失败: {e}")
                return ""
    return ""


# ── 断点续跑 ──────────────────────────────────────────────────
def load_checkpoint(output_path: str) -> tuple[list, set]:
    """加载已有断点，返回 (details, done_keys)。"""
    p = Path(output_path)
    if not p.exists():
        return [], set()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        details = data.get("details", [])
        # 用 problem/question 的前 100 字符作为去重 key
        done_keys = set()
        for d in details:
            key = (d.get("problem") or d.get("question") or d.get("input", ""))[:100]
            done_keys.add(key)
        return details, done_keys
    except Exception:
        return [], set()


def save_checkpoint(output_path: str, details: list, model_name: str,
                    benchmark: str, sample_indices: list):
    n = len(details)
    correct = sum(1 for d in details if d.get("correct"))
    acc = correct / max(n, 1)
    result = {
        "model": model_name,
        "benchmark": benchmark,
        "accuracy": round(acc, 4),
        "total": n,
        "correct": correct,
        "sample_indices": sample_indices,
        "details": details,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                  encoding="utf-8")


# ── 单个 benchmark 评测 ──────────────────────────────────────
def run_benchmark(api_key: str, model_id: str, model_label: str,
                  benchmark: str, max_samples: int, output_dir: str):
    output_path = str(Path(output_dir) / f"api_{model_label}_{benchmark}.json")

    # 检查已有完整结果
    if Path(output_path).exists():
        try:
            existing = json.loads(Path(output_path).read_text(encoding="utf-8"))
            if existing.get("total", 0) >= max_samples:
                acc = existing.get("accuracy", 0)
                print(f"  ✅ 已有完整结果 ({existing['total']}题, acc={acc:.1%})，跳过")
                return
        except Exception:
            pass

    # 加载数据集
    if benchmark == "gsm8k":
        ds = load_gsm8k()
    else:
        ds = load_math500()

    ds, sample_indices = select_eval_subset(ds, max_samples)

    # 加载断点
    details, done_keys = load_checkpoint(output_path)
    if details:
        print(f"  [续跑] 已有 {len(details)} 条结果，跳过已完成")

    # 评测
    total = len(ds)
    correct = sum(1 for d in details if d.get("correct"))
    t0 = time.time()

    for i, ex in enumerate(tqdm(ds, desc=f"{model_label} {benchmark}",
                                 initial=len(details), total=total)):
        if benchmark == "gsm8k":
            question = ex.get("question", "")
            key = question[:100]
            if key in done_keys:
                continue
            prompt = build_gsm8k_prompt(question)
            gt = ex.get("answer", "")
            # 提取最终数字
            gt_num = gt.split("####")[-1].strip() if "####" in gt else gt
            pred_raw = call_api(api_key, model_id, prompt)
            pred_answer = extract_number(pred_raw)
            ok = pred_answer == gt_num.strip()
            details.append({
                "question": question,
                "pred": pred_answer,
                "pred_raw": pred_raw,
                "gt": gt_num.strip(),
                "correct": ok,
            })
        else:
            problem = ex.get("problem", "")
            key = problem[:100]
            if key in done_keys:
                continue
            prompt = build_math_prompt(problem)
            gt_answer = ex.get("answer") or extract_boxed(ex.get("solution", ""))
            pred_raw = call_api(api_key, model_id, prompt)
            pred_answer = extract_math_answer(pred_raw)
            ok = is_equiv(pred_answer, gt_answer)
            details.append({
                "problem": problem,
                "pred": pred_answer,
                "pred_raw": pred_raw,
                "gt": gt_answer,
                "level": ex.get("level"),
                "subject": ex.get("subject"),
                "correct": ok,
            })

        done_keys.add(key)
        correct += int(ok)

        # 进度打印
        if len(details) % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            acc_so_far = correct / max(len(details), 1)
            speed = len(details) - (total - len(ds) + i + 1 - (i + 1 - len(details)))
            done_n = len(details)
            speed_per_sec = done_n / max(elapsed, 1e-6)
            eta = (total - done_n) / max(speed_per_sec, 1e-6)
            print(f"  [{done_n}/{total}] acc={acc_so_far:.1%}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
            save_checkpoint(output_path, details, model_id, benchmark, sample_indices)

    # 最终保存
    save_checkpoint(output_path, details, model_id, benchmark, sample_indices)
    n = len(details)
    acc = correct / max(n, 1)
    print(f"  📊 {model_label} {benchmark}: {acc:.1%} ({correct}/{n})")


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DashScope API 评测")
    parser.add_argument("--models", nargs="+", default=["1.5b", "7b"],
                        choices=["1.5b", "7b"], help="要评测的模型")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math"],
                        choices=["gsm8k", "math"], help="要跑的 benchmark")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("❌ DASHSCOPE_API_KEY 未设置，请在 .env 中配置")
        sys.exit(1)

    print(f"🔑 DashScope API Key: {api_key[:8]}...{api_key[-4:]}")
    print(f"📋 模型: {args.models}")
    print(f"📋 Benchmarks: {args.benchmarks}")
    print(f"📋 样本数: {args.max_samples}")
    print(f"📋 输出目录: {args.output_dir}")
    print()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    total_tasks = len(args.models) * len(args.benchmarks)
    done_tasks = 0

    for model_size in args.models:
        model_id = MODEL_MAP[model_size]
        model_label = f"qwen25_{model_size}"
        print(f"\n{'='*60}")
        print(f"  模型: {model_id}")
        print(f"{'='*60}")

        for bench in args.benchmarks:
            done_tasks += 1
            print(f"\n  [{done_tasks}/{total_tasks}] {bench.upper()}")
            run_benchmark(api_key, model_id, model_label, bench,
                          args.max_samples, args.output_dir)

    print(f"\n{'='*60}")
    print("  全部完成！")
    print(f"{'='*60}")

    # 汇总
    print("\n📊 结果汇总:")
    for model_size in args.models:
        model_label = f"qwen25_{model_size}"
        for bench in args.benchmarks:
            path = Path(args.output_dir) / f"api_{model_label}_{bench}.json"
            if path.exists():
                d = json.loads(path.read_text(encoding="utf-8"))
                print(f"  {model_label} {bench:6s}: {d['accuracy']:.1%} ({d['correct']}/{d['total']})")


if __name__ == "__main__":
    main()
