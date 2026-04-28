#!/usr/bin/env python3
"""错误类型分类器（创新点 B 第 1 步：诊断）

用 qwen-flash（便宜）把 student 的 GSM8K badcase 自动归类到 5 个类别，
为后续 Error-Type-Targeted DPO 提供"诊断 → 修复"的数据基础。

【5 类错误（GSM8K 上的典型 failure mode）】
  1. arithmetic        — 算术运算错（加减乘除算错）
  2. reasoning_skip    — 推理跳步（缺少中间步骤）
  3. setup_error       — 题目理解 / 列式错误（设错变量、读错条件）
  4. unit_or_format    — 单位换算 / 数值格式错（小数位、单位、四舍五入）
  5. extraction_error  — 推理过程对，但最终答案抽取错（输出末尾数字提取失败）

【输入】
  --badcase_jsonl   logs/runs/.../gsm8k_sft_badcases.jsonl  (eval 自动产物)
  字段需含 question / pred_raw / gt_raw

【输出】
  按错误类型分桶的 JSONL，便于下游 build_targeted_dpo.py 使用。
  results/errors/by_type/<type>.jsonl
  results/errors/summary.json   汇总分布饼图数据

用法：
    python scripts/classify_errors.py \
        --badcase_jsonl logs/runs/<id>/results/gsm8k_sft_badcases.jsonl \
        --output_dir    results/errors/sft

依赖：DASHSCOPE_API_KEY（或 OPENAI_API_KEY）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-flash"

VALID_TYPES = ["arithmetic", "reasoning_skip", "setup_error", "unit_or_format", "extraction_error"]

CLASSIFY_SYSTEM = (
    "你是一个数学题作答错误的诊断专家。给你一道 GSM8K 题、学生的完整作答、正确答案，"
    "请把错误归到唯一一个类别中：\n"
    "1. arithmetic       —— 算术运算错（加减乘除算错）\n"
    "2. reasoning_skip   —— 推理跳步、漏关键中间步骤、思路不完整\n"
    "3. setup_error      —— 题目理解错或列式错（变量设错、漏读条件、列错方程）\n"
    "4. unit_or_format   —— 单位换算 / 小数位 / 四舍五入错\n"
    "5. extraction_error —— 中间过程基本对，但最终答案抽取/写出错（如末尾数字写错、写出过程中的中间值）\n"
    "只返回 JSON：{\"type\": \"<one of the 5 keys>\", \"reason\": \"<10-30 字理由>\"}"
)


def chat_completion(base_url: str, api_key: str, model: str, system: str, user: str,
                    max_tokens: int = 256, timeout: int = 60, max_retries: int = 4) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
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
            return content
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            last = f"HTTP {e.code}: {body[:200]}"
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


def parse_type(raw: str) -> tuple[str, str]:
    """从 LLM 输出中解析 {type, reason}，失败时返回 unknown。"""
    if not raw or raw.startswith("__ERROR__"):
        return "unknown", raw[:200]
    txt = raw.strip()
    if "```" in txt:
        # 剥离 markdown 代码块
        parts = [seg for seg in txt.split("```") if seg.strip()]
        for seg in parts:
            seg = seg.strip()
            if seg.lower().startswith("json"):
                seg = seg[4:].strip()
            try:
                obj = json.loads(seg)
                t = str(obj.get("type", "")).strip().lower()
                if t in VALID_TYPES:
                    return t, str(obj.get("reason", ""))[:200]
            except Exception:
                continue
    try:
        obj = json.loads(txt)
        t = str(obj.get("type", "")).strip().lower()
        if t in VALID_TYPES:
            return t, str(obj.get("reason", ""))[:200]
    except Exception:
        pass
    # fallback：关键词匹配
    low = txt.lower()
    for t in VALID_TYPES:
        if t in low:
            return t, txt[:200]
    return "unknown", txt[:200]


def build_user(record: dict) -> str:
    q = record.get("question") or record.get("input") or ""
    pred_raw = record.get("pred_raw") or record.get("pred") or ""
    gt = record.get("gt_raw") or record.get("gt") or ""
    return (
        f"题目：\n{q}\n\n"
        f"学生作答：\n{pred_raw}\n\n"
        f"正确答案：\n{gt}\n\n"
        f"请按 system 指令分类。"
    )


def classify_one(record: dict, base_url: str, api_key: str, model: str,
                 max_tokens: int, timeout: int, max_retries: int) -> dict:
    raw = chat_completion(base_url, api_key, model, CLASSIFY_SYSTEM, build_user(record),
                          max_tokens=max_tokens, timeout=timeout, max_retries=max_retries)
    t, reason = parse_type(raw)
    return {**record, "error_type": t, "error_reason": reason, "_raw": raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--badcase_jsonl", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--api_base_url", default=DEFAULT_API_BASE)
    ap.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_samples", type=int, default=0, help="0 表示全部")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max_retries", type=int, default=4)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺少 API key，请设 DASHSCOPE_API_KEY 或 OPENAI_API_KEY")

    in_path = Path(args.badcase_jsonl)
    if not in_path.exists():
        raise SystemExit(f"找不到 badcase 文件: {in_path}")

    records = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    if args.max_samples > 0:
        records = records[:args.max_samples]
    print(f"📥 待分类 {len(records)} 个 badcase | model={args.model} | workers={args.workers}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_type_dir = out_dir / "by_type"
    by_type_dir.mkdir(parents=True, exist_ok=True)

    classified = []
    counter = Counter()
    sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(classify_one, r, args.api_base_url, args.api_key, args.model,
                        args.max_tokens, args.timeout, args.max_retries): r
            for r in records
        }
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            classified.append(res)
            counter[res["error_type"]] += 1
            if i % 20 == 0 or i == len(records):
                print(f"   已分类 {i}/{len(records)} | 当前分布: {dict(counter)}")
                sys.stdout.flush()

    # 全量产物
    full_path = out_dir / "classified.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for r in classified:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 按类型分桶
    buckets = {t: [] for t in VALID_TYPES + ["unknown"]}
    for r in classified:
        buckets.setdefault(r["error_type"], []).append(r)
    for t, rows in buckets.items():
        path = by_type_dir / f"{t}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "input": str(in_path),
        "n_total": len(classified),
        "distribution": {t: counter.get(t, 0) for t in VALID_TYPES + ["unknown"]},
        "distribution_pct": {
            t: round(counter.get(t, 0) / max(1, len(classified)) * 100, 2)
            for t in VALID_TYPES + ["unknown"]
        },
        "model": args.model,
        "by_type_files": {t: str(by_type_dir / f"{t}.jsonl") for t in VALID_TYPES + ["unknown"]},
    }
    sum_path = out_dir / "summary.json"
    with sum_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n📊 错误类型分布：")
    for t in VALID_TYPES + ["unknown"]:
        n = counter.get(t, 0)
        pct = n / max(1, len(classified)) * 100
        print(f"   {t:20s} {n:4d}  ({pct:5.1f}%)")
    print(f"✅ 全量: {full_path}")
    print(f"✅ 分桶: {by_type_dir}/")
    print(f"✅ 汇总: {sum_path}")


if __name__ == "__main__":
    main()
