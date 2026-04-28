#!/usr/bin/env python3
"""Error-Type-Targeted DPO 数据构造器（创新点 B 主体）

【创新叙事】
区别于"通用 Teacher-Guided DPO"——后者只用一个统一的 system prompt 让 Teacher 生成 chosen，
本脚本针对每类错误使用**定制 system prompt**，让 Teacher 在生成 chosen 时**显式强调**修复
该类错误所需的技巧。这让"诊断 → 修复"成为闭环。

【输入】
  --by_type_dir  results/errors/by_type   （classify_errors.py 产物）
  --types        默认全部 5 类，可指定子集（消融用）
  --per_type_n   每类生成多少条 DPO 对（默认 200，5 类 = 1000）

【chosen 来源】
  qwen3-235b-a22b-thinking-2507（高质量 thinking 模型）+ **类型专属 system prompt**

【rejected 来源】
  原 badcase 的 pred_raw（student 的真实错误）

【输出】
  data/processed/dpo_targeted_<tag>.json   合并的 DPO 数据
  data/processed/dpo_targeted_by_type/<type>.json   每类单独的 DPO 数据（消融对照用）

依赖：DASHSCOPE_API_KEY
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3-235b-a22b-thinking-2507"
DEFAULT_FALLBACK = "qwen-max"

# === 类型专属 system prompt（创新核心：针对每类 failure mode 给出修复重点）===
TYPE_PROMPTS = {
    "arithmetic": (
        "你正在为一个会做加减乘除算错的小模型生成示范作答。"
        "请用清晰的中文，对每一步算术都明确写出**算式 → 中间结果**，"
        "不允许跳步省略中间值。最后一行写：答案：<数字>。"
    ),
    "reasoning_skip": (
        "你正在为一个常常推理跳步的小模型生成完整 CoT 示范。"
        "每一步都用单独的句子写，并解释**为什么**这一步是必要的。"
        "最后一行写：答案：<数字>。"
    ),
    "setup_error": (
        "你正在为一个常常误读题目 / 列错方程的小模型生成示范作答。"
        "请先用一段话**复述题目要点（已知 / 求什么 / 关键约束）**，再开始解题。"
        "最后一行写：答案：<数字>。"
    ),
    "unit_or_format": (
        "你正在为一个常在单位换算 / 小数位 / 四舍五入出错的小模型生成示范。"
        "请显式写出**单位**和**保留几位**，必要时用括号标注（例如 \"600 分钟 = 10 小时\"）。"
        "最后一行写：答案：<数字 + 单位（若题目要求）>。"
    ),
    "extraction_error": (
        "你正在为一个常常**最终答案抽取错**的小模型生成示范。"
        "在推理结束后必须用**单独一行**写：\"答案：<数字>\"，**不要在该行后再写任何内容**。"
        "最后一行：答案：<数字>。"
    ),
    "unknown": (
        "请用清晰简洁的中文一步步解答，最后一行写：答案：<数字>。"
    ),
}

VALID_TYPES = ["arithmetic", "reasoning_skip", "setup_error", "unit_or_format", "extraction_error", "unknown"]


def chat_completion(base_url: str, api_key: str, model: str, system: str, user: str,
                    max_tokens: int = 4096, timeout: int = 240,
                    max_retries: int = 4, enable_thinking: bool = True,
                    thinking_budget: int | None = 4096) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
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
                return f"<think>{reasoning}</think>\n\n{content}"
            return content or reasoning
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            last = f"HTTP {e.code}: {body[:200]}"
            if e.code in (429, 500, 502, 503, 504) and i < max_retries - 1:
                time.sleep(2 * (2 ** i))
                continue
            break
        except Exception as e:  # noqa: BLE001
            last = str(e)
            if i < max_retries - 1:
                time.sleep(2 * (2 ** i))
                continue
            break
    return f"__ERROR__: {last}"


def build_user(record: dict) -> str:
    q = record.get("question") or record.get("input") or ""
    return f"题目：\n{q}"


def gen_one(record: dict, base_url: str, api_key: str, model: str,
            max_tokens: int, timeout: int, max_retries: int,
            enable_thinking: bool, thinking_budget: int) -> dict:
    err_type = record.get("error_type", "unknown")
    system = TYPE_PROMPTS.get(err_type, TYPE_PROMPTS["unknown"])
    chosen = chat_completion(
        base_url, api_key, model, system, build_user(record),
        max_tokens=max_tokens, timeout=timeout, max_retries=max_retries,
        enable_thinking=enable_thinking, thinking_budget=thinking_budget,
    )
    return {
        "prompt": record.get("question") or record.get("input") or "",
        "chosen": chosen,
        "rejected": record.get("pred_raw") or record.get("pred") or "",
        "error_type": err_type,
        "system_used": system,
    }


def load_by_type(by_type_dir: Path, types: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for t in types:
        path = by_type_dir / f"{t}.jsonl"
        if not path.exists():
            print(f"⚠️  未找到 {path}，跳过 {t}")
            out[t] = []
            continue
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        out[t] = rows
        print(f"   {t}: {len(rows)} 条 badcase")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by_type_dir", required=True, help="classify_errors.py 的 by_type/ 目录")
    ap.add_argument("--output_dir", default="data/processed", help="DPO 数据输出目录")
    ap.add_argument("--tag", default="round1", help="本次产物文件名 tag")
    ap.add_argument("--types", nargs="*", default=None, help="只处理这些类型（消融用）")
    ap.add_argument("--per_type_n", type=int, default=200, help="每类生成多少条 DPO 对")
    ap.add_argument("--api_base_url", default=DEFAULT_API_BASE)
    ap.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Teacher 模型；省钱用 qwen-max")
    ap.add_argument("--enable_thinking", action="store_true", default=True)
    ap.add_argument("--thinking_budget", type=int, default=4096)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("缺少 API key，请设 DASHSCOPE_API_KEY")

    types = args.types or VALID_TYPES[:-1]  # 默认不包含 unknown
    by_type_dir = Path(args.by_type_dir)
    print(f"📥 加载 by_type: {by_type_dir}")
    buckets = load_by_type(by_type_dir, types)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_type_dir = out_dir / f"dpo_targeted_by_type_{args.tag}"
    per_type_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for t in types:
        rows = buckets.get(t, [])[:args.per_type_n]
        if not rows:
            continue
        print(f"\n🔄 类型 {t}: 生成 {len(rows)} 条 chosen（model={args.model}）")
        per_type_rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {
                pool.submit(gen_one, r, args.api_base_url, args.api_key, args.model,
                            args.max_tokens, args.timeout, args.max_retries,
                            args.enable_thinking, args.thinking_budget): r
                for r in rows
            }
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                if res["chosen"].startswith("__ERROR__"):
                    print(f"   ⚠️  样本失败：{res['chosen'][:120]}")
                    continue
                per_type_rows.append(res)
                if i % 10 == 0 or i == len(rows):
                    print(f"      [{t}] {i}/{len(rows)}")
                    sys.stdout.flush()

        # 保存类型单独文件（消融对照用）
        per_path = per_type_dir / f"{t}.json"
        with per_path.open("w", encoding="utf-8") as f:
            json.dump(per_type_rows, f, ensure_ascii=False, indent=2)
        print(f"   ✓ {t} → {per_path} ({len(per_type_rows)} 条)")
        all_records.extend(per_type_rows)

    merged_path = out_dir / f"dpo_targeted_{args.tag}.json"
    with merged_path.open("w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 合并 DPO 数据: {merged_path} ({len(all_records)} 条)")
    print(f"✅ 类型分桶: {per_type_dir}/")


if __name__ == "__main__":
    main()
