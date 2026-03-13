import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import List

from datasets import load_dataset
from tqdm import tqdm


def extract_number(text: str) -> str:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
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
            return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="ignore")
            except Exception:
                pass
            last_error = f"HTTP {e.code}: {body}"
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

    questions = ds["question"]
    lengths = [len(str(q)) for q in questions]
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


def main():
    parser = argparse.ArgumentParser(description="GSM8K API 评测（OpenAI-compatible）")
    parser.add_argument("--api_base_url", required=True)
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--sampling_mode", choices=["first", "random", "stratified"], default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("缺少 API Key，请通过 --api_key 或 OPENAI_API_KEY 提供")

    ds = load_dataset("gsm8k", "main", split=args.split)
    ds, sample_indices = select_eval_subset(
        ds,
        max_samples=args.max_samples,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
    )

    correct = 0
    details: List[dict] = []
    for ex in tqdm(ds, desc=f"GSM8K-{args.model}"):
        prompt = f"请解答以下数学题，并在最后给出数字答案。\n题目：{ex['question']}\n答案："
        pred = chat_completion(
            base_url=args.api_base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_new_tokens,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        pred_num = extract_number(pred)
        gt_num = extract_number(ex["answer"])
        ok = pred_num == gt_num and pred_num != ""
        correct += int(ok)
        details.append({"question": ex["question"], "pred": pred_num, "gt": gt_num, "correct": ok})

    acc = correct / max(len(details), 1)
    result = {
        "model": args.model,
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
    print(f"GSM8K Accuracy [{args.model}]: {acc:.4f} ({correct}/{len(details)})")


if __name__ == "__main__":
    main()
