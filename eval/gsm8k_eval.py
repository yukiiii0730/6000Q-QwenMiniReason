import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm
import torch

from model_loader import load_model_and_tokenizer

CHECKPOINT_EVERY = 20  # 每 N 条保存一次断点 + 打印进度


def extract_number(text: str) -> str:
    """从模型输出鲁棒地抽取最终数字答案。
    优先级：\\boxed{...} > #### N > "答案"/"answer is" 标志 > 末段最后一个数字。"""
    if not text:
        return ""
    s = text.replace(",", "")

    m = re.findall(r"\\boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}", s)
    if m:
        return m[-1]
    m = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return m[-1]
    for pat in [
        r"答案\s*[:：是为等于]+\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"final answer\s*(?:is|:)?\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"the answer is\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"答案[:：]?\s*\\\(\s*(-?\d+(?:\.\d+)?)",
    ]:
        m = re.findall(pat, s, flags=re.IGNORECASE)
        if m:
            return m[-1]
    tail = s[-300:]
    nums = re.findall(r"-?\d+(?:\.\d+)?", tail)
    if nums:
        return nums[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else ""


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


def build_prompt(tokenizer, question: str, system: str | None = None) -> str:
    """优先用 chat_template（Qwen2.5-Instruct 强依赖），失败回退裸文本。"""
    user_msg = (
        "请一步步推理后给出最终答案，并在最后一行写："
        "答案：<数字>。\n\n题目：" + question
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


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
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
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


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


def _save_checkpoint(output_path: str, details: list, sample_indices: list, args):
    """保存断点（中间结果），格式与最终输出一致。"""
    acc = sum(1 for d in details if d.get("correct")) / max(len(details), 1)
    result = {
        "accuracy": acc,
        "total": len(details),
        "correct": sum(1 for d in details if d.get("correct")),
        "sampling_mode": args.sampling_mode,
        "seed": args.seed,
        "sample_indices": sample_indices,
        "details": details,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


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
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成最大 token 数（CoT 需要 >= 768）")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, load_in_4bit=args.load_in_4bit)

    ds = load_dataset("gsm8k", "main", split=args.split)
    ds, sample_indices = select_eval_subset(
        ds,
        max_samples=args.max_samples,
        sampling_mode=args.sampling_mode,
        seed=args.seed,
    )

    # --- 断点续跑：检查已有 checkpoint ---
    ckpt_path = Path(args.output)
    details: List[dict] = []
    correct = 0
    done_questions: set = set()
    if ckpt_path.exists() and not getattr(args, "force", False):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                old = json.load(f)
            details = old.get("details", [])
            correct = sum(1 for d in details if d.get("correct"))
            done_questions = {d["question"] for d in details}
            if details:
                print(f"[续跑] 已有 {len(details)} 条结果，跳过已完成")
        except Exception:
            pass

    total = len(ds)
    t0 = time.time()
    for ex in tqdm(ds, desc="Evaluating", initial=len(details), total=total):
        if ex["question"] in done_questions:
            continue
        prompt = build_prompt(tokenizer, ex["question"])
        pred_raw = generate_answer(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
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
        # 进度打印 + 断点保存
        if len(details) % CHECKPOINT_EVERY == 0:
            elapsed = time.time() - t0
            acc_so_far = correct / max(len(details), 1)
            speed = len(details) / max(elapsed, 1e-6)
            eta = (total - len(details)) / max(speed, 1e-6)
            print(f"  [{len(details)}/{total}] acc={acc_so_far:.1%}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")
            _save_checkpoint(args.output, details, sample_indices, args)

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
