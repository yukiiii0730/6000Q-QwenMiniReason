#!/usr/bin/env python3
"""
从现有本地缓存数据中筛选高质量子集：
- SFT: Numina 15k + Magpie 15k（总 30k）
- DPO: 15k（若原始不足，则取全部可用）
"""
import argparse
import json
import random
import re
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def score_sft(row: dict) -> float:
    instruction = str(row.get("instruction", "")).strip()
    input_text = str(row.get("input", "")).strip()
    output_text = str(row.get("output", "")).strip()

    score = 0.0
    out_len = len(output_text)
    in_len = len(input_text)

    # 长度分
    if 120 <= out_len <= 2800:
        score += 2.0
    elif 80 <= out_len <= 3500:
        score += 1.0

    if in_len > 0 and 20 <= in_len <= 2000:
        score += 0.8

    # 推理痕迹
    low = output_text.lower()
    reasoning_hits = len(re.findall(r"(因此|所以|先|再|最后|step by step|therefore|hence|final answer)", low))
    score += min(reasoning_hits, 4) * 0.35

    # 数学/结构化痕迹
    if re.search(r"[\d=+\-*/^]", output_text):
        score += 0.5
    if "\\boxed{" in output_text or "答案" in output_text:
        score += 0.3

    # 明显低质量惩罚
    bad = ["as an ai", "i can't assist", "抱歉", "无法帮助", "lorem ipsum"]
    if any(b in low for b in bad):
        score -= 2.0

    if len(instruction) < 4:
        score -= 0.2

    return score


def score_dpo(row: dict) -> float:
    prompt = str(row.get("prompt", "")).strip()
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()

    pl, cl, rl = len(prompt), len(chosen), len(rejected)
    score = 0.0

    if 30 <= pl <= 2600:
        score += 1.0
    if 120 <= cl <= 3000:
        score += 2.0
    elif 80 <= cl <= 4000:
        score += 1.0
    if 40 <= rl <= 4000:
        score += 0.8

    if chosen != rejected:
        score += 1.2
    score += min(abs(cl - rl) / 250.0, 1.2)

    low = chosen.lower()
    if re.search(r"[\d=+\-*/^]", chosen):
        score += 0.4
    if re.search(r"(therefore|hence|step by step|因此|所以|最后|final answer)", low):
        score += 0.3

    if cl < 40 or rl < 20:
        score -= 2.0

    return score


def topk(rows, k: int, score_fn, seed: int):
    rng = random.Random(seed)
    scored = [(score_fn(x), rng.random(), x) for x in rows]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [x for _, _, x in scored[: min(k, len(scored))]]


def split_sft_sources(rows):
    """
    当前 data/processed/sft_train.json 的来源区分：
    - Numina: input 非空（problem/solution）
    - Magpie: input 为空（instruction/response）
    """
    numina = []
    magpie = []
    for x in rows:
        input_text = str(x.get("input", "")).strip()
        if input_text:
            numina.append(x)
        else:
            magpie.append(x)
    return numina, magpie


def main():
    parser = argparse.ArgumentParser(description="从当前缓存数据构建高质量子集")
    parser.add_argument("--sft_in", required=True)
    parser.add_argument("--dpo_in", required=True)
    parser.add_argument("--sft_out", required=True)
    parser.add_argument("--dpo_out", required=True)
    parser.add_argument("--sft_per_source", type=int, default=15000)
    parser.add_argument("--dpo_n", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sft_rows = load_json(args.sft_in)
    dpo_rows = load_json(args.dpo_in)

    numina_rows, magpie_rows = split_sft_sources(sft_rows)
    pick_numina = topk(numina_rows, args.sft_per_source, score_sft, seed=args.seed)
    pick_magpie = topk(magpie_rows, args.sft_per_source, score_sft, seed=args.seed + 1)

    sft_hq = pick_numina + pick_magpie
    random.Random(args.seed).shuffle(sft_hq)

    dpo_target = min(args.dpo_n, len(dpo_rows))
    dpo_hq = topk(dpo_rows, dpo_target, score_dpo, seed=args.seed)

    save_json(args.sft_out, sft_hq)
    save_json(args.dpo_out, dpo_hq)

    print(f"[HQ] SFT 原始: {len(sft_rows)}")
    print(f"[HQ]  Numina 候选: {len(numina_rows)} -> 选中: {len(pick_numina)}")
    print(f"[HQ]  Magpie 候选: {len(magpie_rows)} -> 选中: {len(pick_magpie)}")
    print(f"[HQ] SFT 输出: {len(sft_hq)} -> {args.sft_out}")

    if len(dpo_rows) < args.dpo_n:
        print(f"[HQ] DPO 原始仅 {len(dpo_rows)} 条，少于目标 {args.dpo_n}，已全部使用")
    print(f"[HQ] DPO 输出: {len(dpo_hq)} -> {args.dpo_out}")


if __name__ == "__main__":
    main()
