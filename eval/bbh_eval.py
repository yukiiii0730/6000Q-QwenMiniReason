import argparse
import json
import random
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm
import torch

from model_loader import load_model_and_tokenizer


def normalize(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def extract_answer(pred: str, gt: str) -> bool:
    """从模型输出（含 CoT）健壮地匹配 BBH 答案。
    优先级：末段"答案：X" / "answer is X" > \\boxed{X} > 末段词边界匹配 > 全文词边界匹配。
    """
    import re
    gt_norm = normalize(gt)
    pred_norm = normalize(pred)
    if not pred_norm:
        return False

    pred_low = pred.replace("\n", " ").strip().lower()

    m = re.findall(r"\\boxed\{\s*([^}]+?)\s*\}", pred_low)
    if m and normalize(m[-1]) == gt_norm:
        return True

    tail_low = pred_low[-300:]
    for pat in [
        r"答案\s*[:：是为等于]+\s*([^\n,。.，]+)",
        r"final answer\s*[:：is]+\s*([^\n,。.，]+)",
        r"the answer is\s*([^\n,。.，]+)",
    ]:
        m = re.findall(pat, tail_low)
        if m and normalize(m[-1].strip(" .。,，:：")) == gt_norm:
            return True

    if pred_norm.startswith(gt_norm):
        return True
    pattern = r"(?<![\w])" + re.escape(gt_norm) + r"(?![\w])"
    if re.search(pattern, pred_norm[-300:]):
        return True
    return bool(re.search(pattern, pred_norm))


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

    # stratified: 按 input 长度分桶，分层抽样（确定性）
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


def build_prompt(tokenizer, question: str, system: str | None = None) -> str:
    """套 chat_template（Qwen2.5-Instruct 必须）。"""
    user_msg = (
        "请简洁地一步步推理，并在最后一行写："
        "答案：<最终结果>。\n\n题目：" + question
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
        return prefix + user_msg + "\n答案："


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
    parser = argparse.ArgumentParser(description="BBH 评测")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--subset", default="boolean_expressions")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--sampling_mode", choices=["first", "random", "stratified"], default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval/bbh_result.json")
    parser.add_argument("--badcase_output", default="", help="badcase 输出路径（默认跟随 output 自动生成 *_badcases.jsonl）")
    parser.add_argument("--load_in_4bit", action="store_true", help="使用 bitsandbytes 4bit 量化加载模型")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, load_in_4bit=args.load_in_4bit)

    ds = load_dataset("lukaemon/bbh", args.subset, split="test")
    ds, sample_indices = select_eval_subset(
        ds,
        max_samples=args.max_samples,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
    )

    details: List[dict] = []
    correct = 0
    for ex in tqdm(ds, desc=f"BBH-{args.subset}"):
        prompt = build_prompt(tokenizer, ex["input"])
        pred = generate(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        gt = ex["target"]
        ok = extract_answer(pred, gt)
        correct += int(ok)
        details.append({"input": ex["input"], "pred": pred, "gt": gt, "correct": ok})

    acc = correct / max(len(details), 1)
    result = {
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

    print(f"BBH({args.subset}) Accuracy: {acc:.4f} ({correct}/{len(details)})")
    print(f"BBH({args.subset}) Badcases: {badcase_count} -> {badcase_output}")


if __name__ == "__main__":
    main()
