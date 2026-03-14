import argparse
import json
import random
import re
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def extract_number(text: str) -> str:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


def load_model_and_tokenizer(model_path: str, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


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

    # stratified: 按题目长度分桶，分层抽样（确定性）
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

    if len(picked) < max_samples:
        remaining = [i for i in range(n) if i not in set(picked)]
        need = max_samples - len(picked)
        picked.extend(rng.sample(remaining, need))

    indices = sorted(picked[:max_samples])
    return ds.select(indices), indices


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


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
    parser = argparse.ArgumentParser(description="GSM8K 自动评测")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--sampling_mode", choices=["first", "random", "stratified"], default="stratified")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="eval/gsm8k_result.json")
    parser.add_argument("--badcase_output", default="", help="badcase 输出路径（默认跟随 output 自动生成 *_badcases.jsonl）")
    parser.add_argument("--load_in_4bit", action="store_true", help="使用 bitsandbytes 4bit 量化加载模型")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, load_in_4bit=args.load_in_4bit)

    ds = load_dataset("gsm8k", "main", split=args.split)
    ds, sample_indices = select_eval_subset(
        ds,
        max_samples=args.max_samples,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
    )

    correct = 0
    details: List[dict] = []
    for ex in tqdm(ds, desc="Evaluating"):
        prompt = f"请解答以下数学题，并在最后给出数字答案。\n题目：{ex['question']}\n答案："
        pred_raw = generate_answer(model, tokenizer, prompt)
        pred_num = extract_number(pred_raw)
        gt_num = extract_number(ex["answer"])
        ok = pred_num == gt_num and pred_num != ""
        correct += int(ok)
        details.append(
            {
                "question": ex["question"],
                "pred": pred_num,
                "pred_raw": pred_raw,
                "gt": gt_num,
                "gt_raw": ex["answer"],
                "correct": ok,
            }
        )

    acc = correct / max(len(details), 1)
    result = {
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

    print(f"GSM8K Accuracy: {acc:.4f} ({correct}/{len(details)})")
    print(f"GSM8K Badcases: {badcase_count} -> {badcase_output}")


if __name__ == "__main__":
    main()
