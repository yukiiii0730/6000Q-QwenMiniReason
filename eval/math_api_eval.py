#!/usr/bin/env python3
"""MATH-500 API 评测器（与 gsm8k_api_eval.py 同风格，复用 math_eval 的答案匹配逻辑）。

用法：
    python eval/math_api_eval.py \
        --api_base_url $API_URL --api_key $API_KEY \
        --model qwen2.5-7b-instruct \
        --max_samples 50 --output results/math_qwen25_7b.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# 复用本地评测的答案抽取与等价判断
sys.path.insert(0, str(Path(__file__).resolve().parent))
from math_eval import (  # noqa: E402
    extract_answer, extract_boxed, is_equiv, wilson_ci,
    aggregate_by_level, aggregate_by_subject, select_eval_subset, load_math500,
)


def chat_completion(base_url: str, api_key: str, model: str, prompt: str,
                    max_tokens: int, timeout: int, max_retries: int,
                    enable_thinking: bool = False, thinking_budget: int | None = None) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if enable_thinking:
        payload["enable_thinking"] = True
        if thinking_budget:
            payload["thinking_budget"] = int(thinking_budget)
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last = ""
    for i in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            msg = result["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            if reasoning and content:
                return f"<think>{reasoning}</think>\n{content}"
            return reasoning or content
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            last = f"HTTP {e.code}: {body[:200]}"
            if e.code == 403 and "FreeTierOnly" in body:
                raise RuntimeError(
                    f"免费额度已耗尽（AllocationQuota.FreeTierOnly）。\n"
                    f"解决方法：登录 DashScope 控制台 → 模型广场 → 找到模型 {model!r} "
                    f"→ 关闭「仅用免费额度」开关，改用按量付费；\n"
                    f"或在运行脚本时加 --skip-api 跳过 API 评测。\n"
                    f"原始错误：{body}"
                )
            if e.code in (429, 500, 502, 503, 504) and i < max_retries - 1:
                time.sleep(1.5 * (2 ** i))
                continue
            break
        except Exception as e:  # noqa: BLE001
            last = str(e)
            if i < max_retries - 1:
                time.sleep(1.5 * (2 ** i))
                continue
            break
    return f"__ERROR__: {last}"


def build_user_prompt(problem: str) -> str:
    return ("请一步步推理后给出最终答案，并把最终答案放在 \\boxed{} 中。\n\n"
            "题目：" + problem)


def eval_one(idx: int, ex: dict, base_url: str, api_key: str, model: str,
             max_tokens: int, timeout: int, max_retries: int,
             enable_thinking: bool, thinking_budget: int | None) -> dict:
    problem = ex.get("problem") or ex.get("question") or ""
    gt_full = ex.get("solution") or ex.get("answer") or ""
    gt_answer = ex.get("answer") or extract_boxed(gt_full) or ""

    prompt = build_user_prompt(problem)
    pred_raw = chat_completion(
        base_url, api_key, model, prompt,
        max_tokens=max_tokens, timeout=timeout, max_retries=max_retries,
        enable_thinking=enable_thinking, thinking_budget=thinking_budget,
    )
    if pred_raw.startswith("__ERROR__"):
        return {"idx": idx, "problem": problem, "pred": "", "pred_raw": pred_raw,
                "gt": gt_answer, "gt_raw": gt_full,
                "level": ex.get("level"), "subject": ex.get("subject"),
                "correct": False, "error": pred_raw}

    pred_answer = extract_answer(pred_raw)
    ok = is_equiv(pred_answer, gt_answer)
    return {"idx": idx, "problem": problem, "pred": pred_answer, "pred_raw": pred_raw,
            "gt": gt_answer, "gt_raw": gt_full,
            "level": ex.get("level"), "subject": ex.get("subject"),
            "correct": ok}


def main():
    ap = argparse.ArgumentParser(description="MATH-500 API 评测")
    ap.add_argument("--api_base_url", default=os.environ.get("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    ap.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max_samples", type=int, default=500)
    ap.add_argument("--sampling_mode", choices=["first", "random", "stratified"], default="stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="eval/math_api_result.json")
    ap.add_argument("--max_new_tokens", type=int, default=1024, help="API max_tokens")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--thinking_budget", type=int, default=4096)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺少 API key（DASHSCOPE_API_KEY 或 --api_key）")

    ds = load_math500(args.split)
    ds, sample_indices = select_eval_subset(ds, max_samples=args.max_samples,
                                            sampling_mode=args.sampling_mode, seed=args.seed)
    items = [(i, dict(ex)) for i, ex in enumerate(ds)]
    print(f"📥 MATH-500 API 评测: model={args.model}, n={len(items)}, workers={args.workers}")

    details: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(eval_one, idx, ex, args.api_base_url, args.api_key, args.model,
                        args.max_new_tokens, args.timeout, args.max_retries,
                        args.enable_thinking, args.thinking_budget): idx
            for idx, ex in items
        }
        with tqdm(total=len(futs), desc="MATH-500") as pbar:
            for fut in as_completed(futs):
                res = fut.result()
                details[res["idx"]] = res
                pbar.update(1)

    details = [d for d in details if d is not None]
    n = len(details)
    correct = sum(1 for d in details if d.get("correct"))
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
        "model": args.model,
        "sampling_mode": args.sampling_mode,
        "seed": args.seed,
        "sample_indices": sample_indices,
        "details": details,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"MATH-500 [{args.model}]  Accuracy: {acc*100:.2f}% ({correct}/{n})  "
          f"95% CI: [{ci_lo*100:.1f}, {ci_hi*100:.1f}]")
    print("\n按 Level:")
    for lv, v in by_level.items():
        print(f"   L{lv}: {v['accuracy']*100:6.2f}%  ({v['correct']}/{v['total']})")


if __name__ == "__main__":
    main()
