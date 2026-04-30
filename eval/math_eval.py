#!/usr/bin/env python3
"""MATH-500 本地评测器（v4 新增——真正的数学推理标尺）

【为什么必加】
GSM8K 是小学应用题，已被 7B 以上模型刷到 90%+ 接近天花板。
MATH（Hendrycks et al. 2021）是高中竞赛数学，分 5 个难度级别：
  Level 1: Counting & Probability (introductory)
  ...
  Level 5: 最难（AMC/AIME 改编）
是 1B-5B 推理模型的真正标尺，DeepSeek-Math/R1-Distill/MetaMath 全部用 MATH 报告主指标。
我们用 HuggingFaceH4/MATH-500（标准 500 题子集，OpenAI prm800k 代表样本）。

【MATH 答案匹配的核心难点】
GSM8K 答案是纯数字（"42"），MATH 答案可能是：
  - 整数 / 小数：42, 0.5, -3
  - 分数：\frac{1}{2}, 1/2, \dfrac{3}{4}
  - 根号：\sqrt{2}, 2\sqrt{3}
  - LaTeX 表达式：(2,3), \pi/4, x^2+1
  - 集合：\{1,2,3\}, [0,1)
  - 字符串："yes", "no", "(0, 1, -3)"

本评测器用三层匹配策略：
  1. 字面相等（normalize 后）—— 最严格
  2. 数值相等（能解析成数字时）—— 处理 0.5 vs 1/2
  3. SymPy 等价（可选，需 sympy 包）—— 处理 \frac{1}{2} vs 0.5 vs 50/100

【输出统计】
- 整体 accuracy + 95% CI
- 按 5 个 level 分层 accuracy（这是论文标准报告方式）
- 按 subject 分层（algebra / geometry / number_theory / ...）

用法：
    python eval/math_eval.py \
        --model_path outputs/sft_merged \
        --max_samples 500 \
        --output results/math_sft.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from tqdm import tqdm

from model_loader import load_model_and_tokenizer


# =============================================================================
# 答案抽取与匹配（MATH 核心难点）
# =============================================================================
_BOXED_RE = re.compile(r"\\boxed\s*{")


def extract_boxed(text: str) -> str:
    """提取 \\boxed{...} 的内容，正确处理嵌套大括号（LaTeX 常见）。"""
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


def extract_answer(text: str) -> str:
    """从模型输出抽取最终答案。
    优先级：\\boxed{...} > "答案：..." / "answer is ..." > 末段 LaTeX 表达式。"""
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

    # 末段最后一个 LaTeX 内容
    tail = text[-200:]
    m = re.findall(r"\$([^$]+)\$", tail)
    if m:
        return m[-1].strip()
    return ""


# === 答案 normalize（论文级 robust matching，参考 hendrycks/math 官方实现 + minerva）===
def _strip_string(s: str) -> str:
    """规范化 LaTeX 数学表达式（参考 Lewkowycz et al. Minerva 2022 的实现）。"""
    if s is None:
        return ""
    s = str(s).strip()
    # 去掉 $$ 和 $
    s = s.replace("$", "")
    # 去掉空白
    s = s.replace(" ", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("\\!", "")
    s = s.replace("\n", "")
    s = s.replace("\\,", "")
    # 等号左侧（如 x = 5 → 5）
    if "=" in s and len(s.split("=")[-1]) > 0:
        s = s.split("=")[-1]
    # 度数 / 单位
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\textbf\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{(.*?)\}", r"\1", s)
    # 移除尾部句点
    s = s.rstrip(".")
    # 处理分数：\\frac{a}{b} 与 \\dfrac, \\tfrac
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    s = re.sub(r"\\frac([0-9])([0-9])", r"\1/\2", s)
    # 处理根号
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt([0-9a-zA-Z])", r"sqrt(\1)", s)
    # 度数符号
    s = s.replace("^\\circ", "")
    s = s.replace("^{\\circ}", "")
    s = s.replace("\\circ", "")
    # 千位逗号
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    # 移除外层多余括号 \\left( \\right)
    s = s.replace("\\left", "").replace("\\right", "")
    # +符号 / 空格
    s = s.replace("+\\", "+\\")  # noqa: just keep
    s = s.replace("\\&", "&")
    return s


def _to_float(s: str):
    """尝试把 normalize 后的字符串转 float（支持 a/b 分数）。"""
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
    """三层等价判断。"""
    if pred is None or gt is None:
        return False
    p = _strip_string(pred)
    g = _strip_string(gt)
    if not p or not g:
        return False

    # Layer 1: 字面相等
    if p == g:
        return True

    # Layer 2: 数值相等（容差 1e-4）
    pf = _to_float(p)
    gf = _to_float(g)
    if pf is not None and gf is not None:
        if abs(pf - gf) < 1e-4 or (gf != 0 and abs((pf - gf) / gf) < 1e-4):
            return True

    # Layer 3: SymPy 等价（可选，处理 \frac{1}{2} vs 0.5）
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr  # type: ignore
        # 尝试解析；失败就跳过
        ep = parse_expr(p.replace("sqrt(", "sqrt(").replace("^", "**"),
                        evaluate=True)
        eg = parse_expr(g.replace("sqrt(", "sqrt(").replace("^", "**"),
                        evaluate=True)
        if sympy.simplify(ep - eg) == 0:
            return True
    except Exception:
        pass
    return False


# =============================================================================
# Sampling & Prompt
# =============================================================================
def select_eval_subset(ds, max_samples: int, sampling_mode: str, seed: int):
    n = len(ds)
    if max_samples <= 0 or max_samples >= n:
        return ds, list(range(n))
    rng = random.Random(seed)
    if sampling_mode == "first":
        indices = list(range(max_samples))
        return ds.select(indices), indices
    if sampling_mode == "random":
        indices = sorted(rng.sample(range(n), max_samples))
        return ds.select(indices), indices
    # stratified by level
    levels_field = "level" if "level" in ds.column_names else None
    if not levels_field:
        indices = sorted(rng.sample(range(n), max_samples))
        return ds.select(indices), indices
    by_level: dict = defaultdict(list)
    for i, lv in enumerate(ds[levels_field]):
        by_level[str(lv)].append(i)
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


def build_prompt(tokenizer, problem: str, system: str | None = None) -> str:
    user_msg = (
        "请一步步推理后给出最终答案，并把最终答案放在 \\boxed{} 中。\n\n"
        "题目：" + problem
    )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_msg})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prefix = (system + "\n") if system else ""
        return prefix + user_msg + "\n"


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# =============================================================================
# 统计 & badcase
# =============================================================================
def wilson_ci(p: float, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def resolve_badcase_output(output_path: str, badcase_output: str) -> str:
    if badcase_output:
        return badcase_output
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}_badcases.jsonl"))


def write_badcases(details: List[dict], output_path: str) -> int:
    badcases = [d for d in details if not d.get("correct", False)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in badcases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(badcases)


def aggregate_by_level(details: List[dict]) -> dict:
    by_lv = defaultdict(lambda: {"total": 0, "correct": 0})
    for d in details:
        lv = str(d.get("level", "unknown"))
        by_lv[lv]["total"] += 1
        by_lv[lv]["correct"] += int(d.get("correct", False))
    return {
        lv: {"accuracy": round(v["correct"] / max(1, v["total"]), 4),
             "correct": v["correct"], "total": v["total"]}
        for lv, v in sorted(by_lv.items())
    }


def aggregate_by_subject(details: List[dict]) -> dict:
    by_s = defaultdict(lambda: {"total": 0, "correct": 0})
    for d in details:
        s = str(d.get("subject", "unknown"))
        by_s[s]["total"] += 1
        by_s[s]["correct"] += int(d.get("correct", False))
    return {
        s: {"accuracy": round(v["correct"] / max(1, v["total"]), 4),
            "correct": v["correct"], "total": v["total"]}
        for s, v in sorted(by_s.items(), key=lambda x: -x[1]["total"])
    }


# =============================================================================
# Main
# =============================================================================
def load_math500(split: str = "test"):
    """加载 HuggingFaceH4/MATH-500（标准 500 题子集）。
    字段：problem, solution, answer, subject, level, unique_id"""
    try:
        return load_dataset("HuggingFaceH4/MATH-500", split=split)
    except Exception:
        # 兜底：直接加载 hendrycks/competition_math 的 test split
        print("⚠️  HuggingFaceH4/MATH-500 加载失败，回退 hendrycks/competition_math")
        ds = load_dataset("hendrycks/competition_math", split="test")
        return ds


def main():
    ap = argparse.ArgumentParser(description="MATH-500 自动评测")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=500, help="0 表示全量 500 题")
    ap.add_argument("--sampling_mode", choices=["first", "random", "stratified"], default="stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="eval/math_result.json")
    ap.add_argument("--badcase_output", default="")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    args = ap.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, load_in_4bit=args.load_in_4bit)

    ds = load_math500(args.split)
    ds, sample_indices = select_eval_subset(
        ds, max_samples=args.max_samples, sampling_mode=args.sampling_mode, seed=args.seed
    )

    correct = 0
    details: List[dict] = []
    for ex in tqdm(ds, desc="MATH-500"):
        problem = ex.get("problem") or ex.get("question") or ""
        gt_full = ex.get("solution") or ex.get("answer") or ""
        gt_answer = ex.get("answer") or extract_boxed(gt_full) or ""

        prompt = build_prompt(tokenizer, problem)
        pred_raw = generate_answer(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        pred_answer = extract_answer(pred_raw)
        ok = is_equiv(pred_answer, gt_answer)
        correct += int(ok)

        details.append({
            "problem": problem,
            "pred": pred_answer,
            "pred_raw": pred_raw,
            "gt": gt_answer,
            "gt_raw": gt_full,
            "level": ex.get("level"),
            "subject": ex.get("subject"),
            "correct": ok,
        })

    n = len(details)
    acc = correct / max(n, 1)
    ci_lo, ci_hi = wilson_ci(acc, n)
    by_level = aggregate_by_level(details)
    by_subject = aggregate_by_subject(details)

    result = {
        "accuracy": round(acc, 4),
        "total": n,
        "correct": correct,
        "wilson_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "by_level": by_level,
        "by_subject": by_subject,
        "sampling_mode": args.sampling_mode,
        "seed": args.seed,
        "sample_indices": sample_indices,
        "details": details,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    badcase_path = resolve_badcase_output(args.output, args.badcase_output)
    bc = write_badcases(details, badcase_path)

    print("\n" + "=" * 60)
    print(f"MATH-500 Accuracy: {acc*100:.2f}% ({correct}/{n})  95% CI: [{ci_lo*100:.1f}, {ci_hi*100:.1f}]")
    print("\n按 Level 分层:")
    for lv, v in by_level.items():
        print(f"   Level {lv}: {v['accuracy']*100:6.2f}%  ({v['correct']}/{v['total']})")
    print("\n按 Subject 分层（前 5）:")
    for s, v in list(by_subject.items())[:5]:
        print(f"   {s:30s}: {v['accuracy']*100:6.2f}%  ({v['correct']}/{v['total']})")
    print(f"\nBadcases: {bc} → {badcase_path}")


if __name__ == "__main__":
    main()
