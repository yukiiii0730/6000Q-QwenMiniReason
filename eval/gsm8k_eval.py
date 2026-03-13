import argparse
import json
import re
from typing import List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def extract_number(text: str) -> str:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else ""


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="GSM8K 自动评测")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="eval/gsm8k_result.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    ds = load_dataset("gsm8k", "main", split=args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    correct = 0
    details: List[dict] = []
    for ex in tqdm(ds, desc="Evaluating"):
        prompt = f"请解答以下数学题，并在最后给出数字答案。\n题目：{ex['question']}\n答案："
        pred = generate_answer(model, tokenizer, prompt)
        pred_num = extract_number(pred)
        gt_num = extract_number(ex["answer"])
        ok = pred_num == gt_num and pred_num != ""
        correct += int(ok)
        details.append({"question": ex["question"], "pred": pred_num, "gt": gt_num, "correct": ok})

    acc = correct / max(len(details), 1)
    result = {"accuracy": acc, "total": len(details), "correct": correct, "details": details}

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"GSM8K Accuracy: {acc:.4f} ({correct}/{len(details)})")


if __name__ == "__main__":
    main()
