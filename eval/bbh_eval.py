import argparse
import json
import random
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


def normalize(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def extract_answer(pred: str, gt: str) -> bool:
    """从模型输出中提取答案，支持思维链输出。
    优先匹配末尾最后一次出现的 gt（大小写不敏感），回退到全文任意位置匹配。
    """
    import re
    gt_norm = normalize(gt)
    pred_norm = normalize(pred)

    # 1. 直接前缀匹配（原始逻辑，兼容直接输出答案的模型）
    if pred_norm.startswith(gt_norm):
        return True

    # 2. 在输出末尾 200 字符内查找答案（CoT 模型把答案放在最后）
    tail = pred_norm[-200:]
    # 用词边界匹配，避免 "False" 误匹配 "Falsely"
    pattern = r'(?<![\w])' + re.escape(gt_norm) + r'(?![\w])'
    if re.search(pattern, tail):
        return True

    # 3. 全文搜索（最宽松，作为最后手段）
    if re.search(pattern, pred_norm):
        return True

    return False


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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


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
        prompt = f"{ex['input']}\n请只输出最终答案。"
        pred = generate(model, tokenizer, prompt)
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
