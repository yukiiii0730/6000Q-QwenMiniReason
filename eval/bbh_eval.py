import argparse
import json
from typing import List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def normalize(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


def main():
    parser = argparse.ArgumentParser(description="BBH 评测")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--subset", default="boolean_expressions")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="eval/bbh_result.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    ds = load_dataset("lukaemon/bbh", args.subset, split="test")
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    details: List[dict] = []
    correct = 0
    for ex in tqdm(ds, desc=f"BBH-{args.subset}"):
        prompt = f"{ex['input']}\n请只输出最终答案。"
        pred = generate(model, tokenizer, prompt)
        gt = ex["target"]
        ok = normalize(pred).startswith(normalize(gt))
        correct += int(ok)
        details.append({"input": ex["input"], "pred": pred, "gt": gt, "correct": ok})

    acc = correct / max(len(details), 1)
    result = {
        "subset": args.subset,
        "accuracy": acc,
        "total": len(details),
        "correct": correct,
        "details": details,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"BBH({args.subset}) Accuracy: {acc:.4f} ({correct}/{len(details)})")


if __name__ == "__main__":
    main()
