import argparse
import json
from typing import List

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
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
        ok = extract_answer(pred, gt)
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
