#!/usr/bin/env python3
"""LLM 数据质量过滤器 (本地纯 API 版，无需 GPU)

用一个轻量 Teacher（默认 qwen-flash）给 SFT / DPO 数据样本打分，过滤低质样本。
评分维度（0-5 分），最终汇总为 total_score：
  - step_completeness  推理步骤完整性
  - answer_correctness 答案正确性 / 与题意一致性
  - clarity            表达清晰度
  - reasoning_quality  推理质量（无逻辑跳跃 / 无幻觉）

支持：
  - 增量缓存：同一 (prompt, output) 已评分过则跳过，重启不丢进度
  - 并发：默认 8 路 worker 并发请求 API
  - 失败重试 + 退避；评分解析失败则记 0 分
  - SFT / DPO 两种模式（DPO 模式只对 chosen 评分）

用法：
    # SFT 数据过滤
    python scripts/llm_quality_filter.py \
        --input  data/processed/sft_train.json \
        --output data/processed/sft_train_filtered.json \
        --mode   sft \
        --keep_top_ratio 0.6        # 保留得分 top 60%

    # DPO 数据过滤
    python scripts/llm_quality_filter.py \
        --input  data/processed/dpo_train.json \
        --output data/processed/dpo_train_filtered.json \
        --mode   dpo \
        --min_score 14              # 满分 20，>=14 保留
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm


SYSTEM_PROMPT = """你是一个严格的数据标注员。请按 0-5 整数对样本的四个维度评分：
- step_completeness: 推理步骤是否完整（0=只有答案；3=有部分步骤；5=每一步都解释清楚）
- answer_correctness: 答案是否正确并且与题意一致（0=明显错误；5=完全正确）
- clarity: 表达是否清晰、无重复、无语法错误（0=混乱；5=非常清晰）
- reasoning_quality: 推理是否合乎逻辑、无幻觉、无跳步（0=逻辑混乱；5=严密）

只输出 JSON，不要任何解释，例如：
{"step_completeness": 4, "answer_correctness": 5, "clarity": 4, "reasoning_quality": 4}
"""


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> str:
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
            return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            last_error = f"HTTP {e.code}: {body}"
            if e.code == 403 and "FreeTierOnly" in body:
                raise RuntimeError(
                    f"免费额度已耗尽（AllocationQuota.FreeTierOnly）。\n"
                    f"解决方法：登录 DashScope 控制台 → 模型广场 → 找到模型 {model!r} "
                    f"→ 关闭「仅用免费额度」开关；\n"
                    f"或用 --skip-filter 跳过质量过滤。"
                )
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


def parse_score(raw: str) -> dict:
    """从模型输出里解析 JSON 评分；失败则返回零分。"""
    if not raw:
        return {"step_completeness": 0, "answer_correctness": 0, "clarity": 0, "reasoning_quality": 0}
    # 尝试直接 JSON 解析
    try:
        return json.loads(raw)
    except Exception:
        pass
    # 兜底：用正则抓四个字段
    keys = ["step_completeness", "answer_correctness", "clarity", "reasoning_quality"]
    result = {}
    for k in keys:
        m = re.search(rf'"{k}"\s*:\s*(\d+)', raw)
        result[k] = int(m.group(1)) if m else 0
    return result


def total_score(scores: dict) -> int:
    return int(
        scores.get("step_completeness", 0)
        + scores.get("answer_correctness", 0)
        + scores.get("clarity", 0)
        + scores.get("reasoning_quality", 0)
    )


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_user_prompt_sft(row: dict) -> str:
    instr = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    out = (row.get("output") or "").strip()
    return f"【任务】\n{instr}\n\n【题目/输入】\n{inp}\n\n【答案/输出】\n{out}"


def build_user_prompt_dpo(row: dict) -> str:
    prompt = (row.get("prompt") or "").strip()
    chosen = (row.get("chosen") or "").strip()
    return f"【题目】\n{prompt}\n\n【候选答案 (chosen)】\n{chosen}"


def score_one(row: dict, mode: str, base_url: str, api_key: str, model: str,
              max_tokens: int, timeout: int, max_retries: int) -> dict:
    user = build_user_prompt_sft(row) if mode == "sft" else build_user_prompt_dpo(row)
    raw = chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system=SYSTEM_PROMPT,
        user=user[:8000],
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )
    scores = parse_score(raw)
    return {
        "scores": scores,
        "total_score": total_score(scores),
        "raw_response": raw[:500],
    }


def cache_key(row: dict, mode: str) -> str:
    if mode == "sft":
        sig = (row.get("instruction", "") + "||" + row.get("input", "") + "||" + row.get("output", ""))
    else:
        sig = (row.get("prompt", "") + "||" + row.get("chosen", ""))
    import hashlib
    return hashlib.md5(sig.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="LLM 数据质量过滤器（API 调用，本地可跑）")
    parser.add_argument("--input", required=True, help="输入 JSON（list[row]）")
    parser.add_argument("--output", required=True, help="输出过滤后的 JSON")
    parser.add_argument("--mode", choices=["sft", "dpo"], default="sft")
    parser.add_argument("--cache", default="", help="评分缓存 JSONL（默认 <input>.scores.jsonl）")
    parser.add_argument("--config", default="config/benchmark_models.yaml", help="读取 api_evaluation.teacher_models.quality_filter")
    parser.add_argument("--api_base_url", default="")
    parser.add_argument("--api_key", default=os.environ.get("DASHSCOPE_API_KEY", ""))
    parser.add_argument("--model", default="", help="覆盖配置里的 quality_filter 模型")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0, help="只评分前 N 条（0=全部）")
    parser.add_argument("--keep_top_ratio", type=float, default=0.0, help="保留 top 比例（0~1）")
    parser.add_argument("--min_score", type=int, default=0, help="最低总分阈值（0~20），与 keep_top_ratio 互斥")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    api_cfg = ((cfg.get("api_evaluation") or {}))
    quality_cfg = (api_cfg.get("teacher_models") or {}).get("quality_filter", {})

    base_url = args.api_base_url or api_cfg.get("api_base_url", "")
    model = args.model or quality_cfg.get("model", "qwen-flash")
    max_tokens = args.max_tokens or int(quality_cfg.get("max_tokens", 128))
    timeout = args.timeout or int(api_cfg.get("timeout", 60))
    max_retries = args.max_retries or int(api_cfg.get("max_retries", 4))

    if not args.api_key:
        sys.exit("❌ 缺少 DASHSCOPE_API_KEY（请放入 .env 或通过 --api_key 传入）")
    if not base_url:
        sys.exit("❌ 缺少 api_base_url")

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.max_samples and args.max_samples < len(rows):
        rows = rows[: args.max_samples]
    print(f"📥 加载 {len(rows)} 条样本，模式={args.mode}，模型={model}")

    cache_path = Path(args.cache or args.input + ".scores.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    scored: dict[str, dict] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    scored[item["key"]] = item
                except Exception:
                    continue
        print(f"♻️  从缓存加载 {len(scored)} 条历史评分: {cache_path}")

    pending = [(i, row) for i, row in enumerate(rows) if cache_key(row, args.mode) not in scored]
    print(f"📊 待评分: {len(pending)} / {len(rows)}")

    cache_fp = cache_path.open("a", encoding="utf-8")

    _fatal_error: list[str] = []  # 共享：检测到致命错误时快速停止

    def _job(idx_row):
        if _fatal_error:
            return idx_row[0], idx_row[1], {"scores": {}, "total_score": 0, "raw_response": ""}, "已中止（致命错误）"
        idx, row = idx_row
        try:
            result = score_one(
                row, args.mode, base_url, args.api_key, model,
                max_tokens, timeout, max_retries,
            )
            return idx, row, result, None
        except RuntimeError as e:
            err_str = str(e)
            if "FreeTierOnly" in err_str or "免费额度已耗尽" in err_str:
                _fatal_error.append(err_str)
                print(f"\n❌ 致命错误，停止评分：\n{err_str}")
            return idx, row, {"scores": {}, "total_score": 0, "raw_response": ""}, err_str
        except Exception as e:  # noqa: BLE001
            return idx, row, {"scores": {}, "total_score": 0, "raw_response": ""}, str(e)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, x) for x in pending]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM 评分"):
            idx, row, result, err = fut.result()
            key = cache_key(row, args.mode)
            entry = {"key": key, "idx": idx, "result": result}
            if err:
                entry["error"] = err
                failed += 1
            scored[key] = entry
            cache_fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
            cache_fp.flush()

    cache_fp.close()
    print(f"✅ 评分完成；失败 {failed} 条（已计 0 分）")

    enriched = []
    for row in rows:
        key = cache_key(row, args.mode)
        score_info = scored.get(key, {}).get("result", {})
        enriched_row = dict(row)
        enriched_row["_quality_score"] = score_info.get("total_score", 0)
        enriched_row["_quality_breakdown"] = score_info.get("scores", {})
        enriched.append(enriched_row)

    if args.keep_top_ratio > 0:
        keep = max(1, int(len(enriched) * args.keep_top_ratio))
        enriched.sort(key=lambda x: x["_quality_score"], reverse=True)
        kept = enriched[:keep]
        print(f"🔝 保留 top {args.keep_top_ratio*100:.0f}% = {len(kept)} 条 (阈值={kept[-1]['_quality_score']})")
    else:
        threshold = args.min_score
        kept = [x for x in enriched if x["_quality_score"] >= threshold]
        print(f"🎯 保留总分 ≥ {threshold} 的 {len(kept)} / {len(enriched)} 条")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存过滤后数据 → {args.output}")

    # 同时输出统计 summary
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary = {
        "input_count": len(rows),
        "kept_count": len(kept),
        "score_avg": sum(x["_quality_score"] for x in enriched) / max(len(enriched), 1),
        "score_distribution": {
            f"{lo}-{lo+4}": sum(1 for x in enriched if lo <= x["_quality_score"] < lo + 4)
            for lo in range(0, 21, 4)
        },
        "model": model,
        "mode": args.mode,
        "filter": {
            "keep_top_ratio": args.keep_top_ratio,
            "min_score": args.min_score,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📑 统计摘要 → {summary_path}")


if __name__ == "__main__":
    main()
