"""Step 5: Iterative DPO。

流程：
  1. 用当前 SFT 模型在训练集上推理，收集错误输出
  2. 用正确答案（ground truth）作为 chosen，模型错误输出作为 rejected
  3. 构造 DPO pairs，训练一轮 DPO
  4. 可多次迭代：DPO 后的模型再推理 → 再收集错误 → 再 DPO

用法：
    # 第 1 轮 iterative DPO
    python scripts/iterative_dpo.py \
        --sft_model /path/to/sft_output \
        --eval_data data/processed/sft_train.json \
        --output_dir outputs/iterative_dpo_r1 \
        --config config/dpo_config.yaml \
        --max_eval_samples 5000 \
        --round 1

    # 第 2 轮（用第 1 轮的 DPO 模型继续）
    python scripts/iterative_dpo.py \
        --sft_model outputs/iterative_dpo_r1 \
        --eval_data data/processed/sft_train.json \
        --output_dir outputs/iterative_dpo_r2 \
        --config config/dpo_config.yaml \
        --round 2
"""

import argparse
import json
import os
import re
import sys
import types
import yaml
import torch
from pathlib import Path
from datasets import Dataset

# ── TRL 依赖修复 ──────────────────────────────────────────────
for _mod_name in ("llm_blender", "llm_blender.blender", "llm_blender.blender.blender",
                   "llm_blender.blender.blender_utils", "mergekit", "mergekit.config"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

from unsloth import FastLanguageModel
from trl import DPOTrainer, DPOConfig
from transformers import EarlyStoppingCallback


def extract_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def find_latest_checkpoint(output_dir: str):
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = []
    for p in out.glob("checkpoint-*"):
        if p.is_dir():
            try:
                ckpts.append((int(p.name.split("-")[-1]), p))
            except ValueError:
                continue
    return str(max(ckpts, key=lambda x: x[0])[1]) if ckpts else None


def collect_errors(model, tokenizer, data: list, max_samples: int, seed: int) -> list:
    """用模型推理训练集，收集错误输出作为 DPO rejected"""
    import random
    rng = random.Random(seed)

    # 只取有 input 和 output 的数据
    candidates = [d for d in data if d.get("input", "").strip() and d.get("output", "").strip()]
    rng.shuffle(candidates)
    candidates = candidates[:max_samples]

    print(f"🔍 用模型推理 {len(candidates)} 道题，收集错误输出...")

    dpo_pairs = []
    correct_count = 0

    for i, row in enumerate(candidates):
        question = row["input"].strip()
        gt_answer = row["output"].strip()
        gt_num = extract_number(gt_answer)

        prompt = f"请解答以下数学题，逐步推理后给出最终答案。\n题目：{question}\n解答："
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 去掉 prompt 部分
        if pred_text.startswith(prompt):
            pred_text = pred_text[len(prompt):]
        pred_num = extract_number(pred_text)

        if pred_num == gt_num and pred_num:
            correct_count += 1
        else:
            # 模型答错了 → 构造 DPO pair
            dpo_pairs.append({
                "prompt": question,
                "chosen": gt_answer,            # ground truth 作为 chosen
                "rejected": pred_text.strip(),   # 模型错误输出作为 rejected
            })

        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(candidates)}，正确 {correct_count}，错误 {len(dpo_pairs)}")

    acc = correct_count / max(len(candidates), 1)
    print(f"\n📊 模型准确率: {acc:.2%} ({correct_count}/{len(candidates)})")
    print(f"📊 收集到 {len(dpo_pairs)} 个错误样本作为 DPO pairs")
    return dpo_pairs


def main():
    parser = argparse.ArgumentParser(description="Step 5: Iterative DPO")
    parser.add_argument("--sft_model", required=True, help="SFT/DPO 模型路径")
    parser.add_argument("--eval_data", required=True, help="SFT 训练数据（用于推理收集错误）")
    parser.add_argument("--output_dir", required=True, help="DPO 输出目录")
    parser.add_argument("--config", default="config/dpo_config.yaml")
    parser.add_argument("--max_eval_samples", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--round", type=int, default=1, help="迭代轮数标记")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"🔄 Iterative DPO - Round {args.round}")
    print(f"   模型: {args.sft_model}")
    print(f"   数据: {args.eval_data}")

    # ── Step 1: 加载模型 ──────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.sft_model,
        max_seq_length=cfg.get("max_seq_length", 1536),
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    FastLanguageModel.for_inference(model)

    # ── Step 2: 推理收集错误 ──────────────────────────────────────
    with open(args.eval_data, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    dpo_pairs = collect_errors(model, tokenizer, eval_data,
                                max_samples=args.max_eval_samples, seed=args.seed)

    if len(dpo_pairs) < 10:
        print("⚠️  错误样本太少（<10），跳过 DPO 训练")
        return

    # 保存收集到的 DPO 数据
    pairs_path = Path(args.output_dir) / f"iterative_dpo_pairs_r{args.round}.json"
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pairs_path, "w", encoding="utf-8") as f:
        json.dump(dpo_pairs, f, ensure_ascii=False, indent=2)
    print(f"💾 DPO pairs 已保存: {pairs_path}")

    # ── Step 3: DPO 训练 ──────────────────────────────────────────
    # 重新加载模型用于训练
    FastLanguageModel.for_training(model)

    has_lora = bool(getattr(model, "peft_config", None))
    if not has_lora:
        lora_cfg = cfg.get("lora", {})
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.0),
            target_modules=lora_cfg.get("target_modules", ["q_proj","k_proj","v_proj","o_proj"]),
            use_dora=lora_cfg.get("use_dora", False),
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    # 构造 Dataset
    ds = Dataset.from_list(dpo_pairs)
    split = ds.train_test_split(test_size=0.05, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"📊 DPO 训练集: {len(train_ds)} | 验证集: {len(eval_ds)}")

    train_cfg = cfg.get("train", {})
    train_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 2),
        warmup_steps=50,
        max_steps=args.max_steps,
        learning_rate=float(train_cfg.get("learning_rate", 3e-6)),
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        bf16=True, fp16=False,
        beta=float(cfg.get("beta", 0.3)),
        max_length=cfg.get("max_seq_length", 1536),
        max_prompt_length=768,
        report_to="none",
        seed=args.seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    resume_ckpt = find_latest_checkpoint(args.output_dir)
    if resume_ckpt:
        print(f"♻️  断点续训: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Iterative DPO Round {args.round} 完成 → {args.output_dir}")
    print(f"   下一轮可运行:")
    print(f"   python scripts/iterative_dpo.py --sft_model {args.output_dir} --round {args.round+1} ...")


if __name__ == "__main__":
    main()
