import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm


def normalize(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def extract_answer(pred: str, gt: str) -> bool:
    gt_norm = normalize(gt)
    pred_norm = normalize(pred)
    if pred_norm.startswith(gt_norm):
        return True
    tail = pred_norm[-200:]
    pattern = r"(?<![\w])" + re.escape(gt_norm) + r"(?![\w])"
    if re.search(pattern, tail):
        return True
    return re.search(pattern, pred_norm) is not None


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    enable_thinking: bool = False,
    thinking_budget: int | None = None,
) -> str:
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error = ""
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
            last_error = f"HTTP {e.code}: {body}"
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
            last_error = str(e)
            if i < max_retries - 1:
                time.sleep(1.5 * (2 ** i))
                continue
            break

    raise RuntimeError(f"API 请求失败: {last_error}")


def select_eval_subset(ds, max_samples: int, sampling_mode: str, seed: int):
    n = len(ds)
    if max_samples <= 0 or max_samples >= n:
        all_indices = list(range(n))
        return ds, all_indices

    rng = random.Random(seed)

    if sampling_mode == "first":
        indices = list(range(max_samples))
        return ds.select(indices), indices

    if sampling_mode == "random":
        indices = sorted(rng.sample(range(n), max_samples))
        return ds.select(indices), indices

    inputs = ds["input"]
    lengths = [len(str(x)) for x in inputs]
    sorted_indices = sorted(range(n), key=lambda i: lengths[i])
    bins = 5
    groups = [sorted_indices[i * n // bins:(i + 1) * n // bins] for i in range(bins)]

    base = max_samples // bins
    rem = max_samples % bins
    picked = []
    for i, g in enumerate(groups):
        k = base + (1 if i < rem else 0)
        if k > 0 and len(g) > 0:
            picked.extend(rng.sample(g, min(k, len(g))))

    picked_set = set(picked)
    if len(picked) < max_samples:
        remaining = [i for i in range(n) if i not in picked_set]
        need = max_samples - len(picked)
        picked.extend(rng.sample(remaining, need))

    indices = sorted(picked[:max_samples])
    return ds.select(indices), indices


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


def main():
    parser = argparse.ArgumentParser(description="BBH API 评测（OpenAI-compatible）")
    parser.add_argument("--api_base_url", required=True)
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", required=True)
    parser.add_argument("--subset", default="boolean_expressions")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--sampling_mode", choices=["first", "random", "stratified"], default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--enable_thinking", action="store_true", help="为 Qwen3 thinking 模型启用思考模式")
    parser.add_argument("--thinking_budget", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--badcase_output", default="", help="badcase 输出路径（默认跟随 output 自动生成 *_badcases.jsonl）")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("缺少 API Key，请通过 --api_key 或 OPENAI_API_KEY 提供")

    ds = load_dataset("lukaemon/bbh", args.subset, split="test")
    ds, sample_indices = select_eval_subset(
        ds,
        max_samples=args.max_samples,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
    )

    correct = 0
    details: List[dict] = []
    for ex in tqdm(ds, desc=f"BBH-{args.model}"):
        prompt = f"{ex['input']}\n请只输出最终答案。"
        pred = chat_completion(
            base_url=args.api_base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_new_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget or None,
        )
        gt = ex["target"]
        ok = extract_answer(pred, gt)
        correct += int(ok)
        details.append({"input": ex["input"], "pred": pred, "gt": gt, "correct": ok})

    acc = correct / max(len(details), 1)
    result = {
        "model": args.model,
        "subset": args.subset,
        "accuracy": acc,
        "total": len(details),
        "correct": correct,
        "sampling_mode": args.sampling_mode,
        "seed": args.seed,
        "sample_indices": sample_indices,
        "details": details,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    badcase_output = resolve_badcase_output(args.output, args.badcase_output)
    badcase_count = write_badcases(details, badcase_output)

    print(f"BBH({args.subset}) Accuracy [{args.model}]: {acc:.4f} ({correct}/{len(details)})")
    print(f"BBH({args.subset}) Badcases [{args.model}]: {badcase_count} -> {badcase_output}")


if __name__ == "__main__":
    main()
