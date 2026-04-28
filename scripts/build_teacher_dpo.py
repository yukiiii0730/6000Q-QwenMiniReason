#!/usr/bin/env python3
"""Teacher-Guided DPO 数据构造器（本地纯 API 可跑）

【设计】
chosen   = Qwen3-235B-Thinking 对题目的详细 CoT 答案
rejected = 来源二选一：
  (A) 从 evaluations 中提取的 student badcase（GPU 环境产出 *_badcases.jsonl，本地消费）
  (B) 直接传入 jsonl，每行 {question, pred_raw, ...}

【为什么这种构造比开源 DPO 数据更优】
- chosen 与训练分布完全一致（都是数学/推理题）
- rejected 是 1.5B Student 的真实错误（最难的负样本，DPO 信号最强）

用法 1：从 GSM8K badcase JSONL 构造（推荐，最贴合任务分布）
    python scripts/build_teacher_dpo.py \
        --rejected_jsonl logs/runs/<run_id>/results/gsm8k_sft_badcases.jsonl \
        --output         data/processed/dpo_teacher_round_1.json \
        --max_samples    1500

用法 2：从公开题库（NuminaMath / GSM8K-train）构造，无 student 输出时 rejected 留空（不推荐）
    python scripts/build_teacher_dpo.py \
        --source_dataset open-r1/OpenR1-Math-220k \
        --source_split   train \
        --output         data/processed/dpo_teacher_seed.json \
        --max_samples    500
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

import yaml
from tqdm import tqdm

from hf_ssl_env import apply_hf_ssl_fix

apply_hf_ssl_fix()

COT_SYSTEM_PROMPT = (
    "你是一位严谨的数学/推理导师。请用清晰的 step-by-step 方式解答下列题目，"
    "每一步都说明原因，最终在末尾用 \\boxed{...} 给出答案。"
)


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    user: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    enable_thinking: bool = True,
    thinking_budget: int | None = 8192,
    system: str | None = COT_SYSTEM_PROMPT,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    payload: dict = {
        "model": model,
        "messages": msgs,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if enable_thinking:
        payload["enable_thinking"] = True
        if thinking_budget:
            payload["thinking_budget"] = int(thinking_budget)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps(payload).encode("utf-8")

    last_error = ""
    for i in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            msg = result["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            # 拼接 thinking + answer，作为高质量 chosen
            if reasoning and content:
                return f"{reasoning.strip()}\n\n{content.strip()}"
            return content or reasoning
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            last_error = f"HTTP {e.code}: {body[:300]}"
            if e.code in (429, 500, 502, 503, 504) and i < max_retries - 1:
                time.sleep(1.5 * (2 ** i))
                continue
            break
        except Exception as e:  # noqa: BLE001
            last_error = str(e)
            if i < max_retries - 1:
                time.sleep(1.5 * (2 ** i))
                continue
            break
    raise RuntimeError(f"API 失败: {last_error}")


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_rejected_from_badcase_jsonl(path: str, max_n: int) -> list[dict]:
    """从评测脚本生成的 *_badcases.jsonl 中读取 (question, student_pred)。

    详见 eval/gsm8k_eval.py / bbh_eval.py 输出 schema：
      GSM8K: {question, pred_raw, pred, gt_raw, gt, correct=False}
      BBH  : {input,    pred,                 gt,    correct=False}
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            question = row.get("question") or row.get("input") or ""
            pred = row.get("pred_raw") or row.get("pred") or ""
            if not question or not pred:
                continue
            rows.append({"prompt": str(question).strip(), "rejected": str(pred).strip(), "_source": path})
            if max_n and len(rows) >= max_n:
                break
    return rows


def read_rejected_from_pairs_jsonl(path: str, max_n: int) -> list[dict]:
    """从已有 (prompt, rejected) JSONL 直接读。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            prompt = row.get("prompt") or row.get("question") or ""
            rejected = row.get("rejected") or row.get("pred_raw") or row.get("pred") or ""
            if not prompt or not rejected:
                continue
            rows.append({"prompt": str(prompt).strip(), "rejected": str(rejected).strip(), "_source": path})
            if max_n and len(rows) >= max_n:
                break
    return rows


def read_seed_questions_from_dataset(name: str, config: str, split: str, max_n: int) -> list[dict]:
    from datasets import load_dataset
    if config:
        ds = load_dataset(name, config, split=split)
    else:
        ds = load_dataset(name, split=split)
    if max_n and max_n < len(ds):
        ds = ds.select(range(max_n))
    rows = []
    for ex in ds:
        q = ex.get("problem") or ex.get("question") or ex.get("input") or ex.get("instruction") or ""
        if not q:
            continue
        rows.append({"prompt": str(q).strip(), "rejected": "", "_source": name})
    return rows


def cache_key(prompt: str) -> str:
    import hashlib
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Teacher-Guided DPO 数据构造器")
    parser.add_argument("--output", required=True, help="输出 DPO JSON")
    parser.add_argument("--rejected_jsonl", default="", help="badcase 或 pair jsonl 文件路径（rejected 来源）")
    parser.add_argument("--rejected_kind", choices=["badcase", "pair"], default="badcase",
                        help="badcase=eval 输出格式; pair=已是 prompt/rejected 两列")
    parser.add_argument("--source_dataset", default="", help="无 student 输出时，从公开题库取 prompt（rejected 留空）")
    parser.add_argument("--source_config", default="default")
    parser.add_argument("--source_split", default="train")
    parser.add_argument("--max_samples", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", default="", help="缓存 jsonl（默认 <output>.cache.jsonl）")
    parser.add_argument("--config", default="config/benchmark_models.yaml")
    parser.add_argument("--api_base_url", default="")
    parser.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--model", default="", help="覆盖 cot_generator.model")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--thinking_budget", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max_retries", type=int, default=4)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    api_cfg = cfg.get("api_evaluation") or {}
    teacher_cfg = (api_cfg.get("teacher_models") or {}).get("cot_generator", {})
    base_url = args.api_base_url or api_cfg.get("api_base_url", "")
    model = args.model or teacher_cfg.get("model", "qwen3-235b-a22b-thinking-2507")
    enable_thinking = bool(teacher_cfg.get("enable_thinking", True))
    max_tokens = args.max_tokens or int(teacher_cfg.get("max_tokens", 4096))
    thinking_budget = args.thinking_budget or int(teacher_cfg.get("thinking_budget", 8192))
    timeout = args.timeout or int(api_cfg.get("timeout", 180))
    max_retries = args.max_retries or int(api_cfg.get("max_retries", 4))

    if not args.api_key:
        sys.exit("❌ 缺少 DASHSCOPE_API_KEY")
    if not base_url:
        sys.exit("❌ 缺少 api_base_url")

    if args.rejected_jsonl:
        if args.rejected_kind == "badcase":
            rows = read_rejected_from_badcase_jsonl(args.rejected_jsonl, args.max_samples)
        else:
            rows = read_rejected_from_pairs_jsonl(args.rejected_jsonl, args.max_samples)
    elif args.source_dataset:
        rows = read_seed_questions_from_dataset(
            args.source_dataset, args.source_config, args.source_split, args.max_samples,
        )
    else:
        sys.exit("❌ 必须提供 --rejected_jsonl 或 --source_dataset 之一")

    print(f"📥 待生成 chosen 的样本数: {len(rows)}  | Teacher = {model}  | thinking={enable_thinking}")

    cache_path = Path(args.cache or args.output + ".cache.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, str] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cache[item["key"]] = item["chosen"]
                except Exception:
                    continue
        print(f"♻️  缓存命中: {len(cache)} 条 ({cache_path})")

    cache_fp = cache_path.open("a", encoding="utf-8")

    def _job(row):
        key = cache_key(row["prompt"])
        if key in cache:
            return row, cache[key], None
        try:
            chosen = chat_completion(
                base_url=base_url,
                api_key=args.api_key,
                model=model,
                user=row["prompt"],
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=max_retries,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget,
            )
            return row, chosen, None
        except Exception as e:  # noqa: BLE001
            return row, "", str(e)

    pairs = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, row) for row in rows]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Teacher CoT"):
            row, chosen, err = fut.result()
            if err:
                failed += 1
                continue
            if not chosen.strip():
                continue
            key = cache_key(row["prompt"])
            if key not in cache:
                cache[key] = chosen
                cache_fp.write(json.dumps({"key": key, "chosen": chosen}, ensure_ascii=False) + "\n")
                cache_fp.flush()

            pair = {
                "prompt": row["prompt"],
                "chosen": chosen.strip(),
                "rejected": row.get("rejected", "").strip(),
                "_source": row.get("_source", ""),
            }
            # 如果没有 rejected（seed 模式），就不进入最终输出（DPO 必须有 rejected）
            if pair["rejected"]:
                pairs.append(pair)
            else:
                # 仅落入 seed 子集的题，单独保留供下游脚本拼接
                pairs.append(pair)

    cache_fp.close()
    print(f"✅ 生成完成: {len(pairs)} 条 (失败 {failed})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存 → {args.output}")

    # 统计 summary
    full_pairs = sum(1 for p in pairs if p["rejected"])
    print(f"📊 完整 (prompt+chosen+rejected) 对: {full_pairs} / {len(pairs)}")


if __name__ == "__main__":
    main()
