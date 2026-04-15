"""全量微调（Full Fine-Tuning）训练脚本。

与 LoRA/DoRA 实验对比的第三组：不使用任何 adapter，直接微调全部参数。
1.5B 模型在 A100-80GB 上可以直接全量微调，无需量化。

用法：
    python scripts/full_finetune.py --config config/sft_config.yaml
"""

import argparse
import json
import os
import sys
import types
import yaml
import torch
from pathlib import Path
from datasets import load_dataset

# ── 修复 TRL 导入依赖缺失问题 ─────────────────────────────────
for _mod_name in ("llm_blender", "llm_blender.blender", "llm_blender.blender.blender",
                   "llm_blender.blender.blender_utils", "mergekit", "mergekit.config"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig


DEFAULT_INSTRUCTION = "请先进行清晰的逐步推理，再给出最终答案。"


# ── 数据格式化 ─────────────────────────────────────────────────

def normalize_example(example: dict) -> dict:
    if "problem" in example and "solution" in example:
        return {"instruction": DEFAULT_INSTRUCTION,
                "input": str(example["problem"]).strip(),
                "output": str(example["solution"]).strip()}
    if "instruction" in example and "response" in example:
        return {"instruction": str(example["instruction"]).strip(),
                "input": "",
                "output": str(example["response"]).strip()}
    return {"instruction": str(example.get("instruction", DEFAULT_INSTRUCTION)).strip(),
            "input": str(example.get("input", "")).strip(),
            "output": str(example.get("output", "")).strip()}


def formatting_func(example, tokenizer):
    normed = normalize_example(example)
    query = normed["instruction"]
    if normed["input"]:
        query = f"{normed['instruction']}\n\n{normed['input']}"
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": normed["output"]}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


# ── 断点续训 ───────────────────────────────────────────────────

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


# ── 主流程 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full fine-tuning (no LoRA)")
    parser.add_argument("--config", default="config/sft_config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # HF Token
    hf_token = os.environ.get("HF_TOKEN") or cfg.get("hf_token", "").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # ── 加载模型（不量化、不加 LoRA）──────────────────────────────
    # 1.5B bf16 ≈ 3GB 参数，A100-80GB 绰绰有余
    print(f"📥 加载模型: {cfg['model_name']} (全量 bf16，不量化)")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 开启 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()
    print(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M (全部可训练)")

    # ── 加载数据 ──────────────────────────────────────────────────
    ds_path = cfg.get("dataset_path")
    if ds_path:
        raw_ds = load_dataset("json", data_files=ds_path, split="train")
    else:
        raise ValueError("config 中缺少 dataset_path")

    eos = tokenizer.eos_token
    ds = raw_ds.map(
        lambda ex: {"text": formatting_func(ex, tokenizer) + eos},
        remove_columns=raw_ds.column_names
    )

    split = ds.train_test_split(test_size=0.05, seed=cfg["seed"])
    train_ds, eval_ds = split["train"], split["test"]
    print(f"📊 训练集: {len(train_ds)} 条 | 验证集: {len(eval_ds)} 条")

    # ── 训练配置 ──────────────────────────────────────────────────
    # 全量微调 vs LoRA 的配置差异：
    #   - 学习率更低（全量微调不需要 LoRA 那么大的 lr）
    #   - batch_size 更小（全量微调显存占用更大）
    #   - 使用 adamw_torch（不用 paged_adamw_8bit，全量微调不需要量化优化器）
    train_cfg = cfg["train"]
    train_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        warmup_steps=train_cfg.get("warmup_steps", 150),
        max_steps=train_cfg.get("max_steps", 1800),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim="adamw_torch",          # 全量微调用标准优化器
        bf16=True,
        fp16=False,
        report_to="none",
        seed=cfg["seed"],
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=True,
        dataset_text_field="text",
        packing=bool(train_cfg.get("packing", True)),
    )

    tokenizer.model_max_length = cfg["max_seq_length"]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=train_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # ── 训练 ──────────────────────────────────────────────────────
    resume_ckpt = find_latest_checkpoint(cfg["output_dir"])
    if resume_ckpt:
        print(f"♻️  检测到断点: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    # ── 保存 ──────────────────────────────────────────────────────
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # 导出训练日志
    history = list(getattr(trainer.state, "log_history", []) or [])
    if history:
        log_path = Path(cfg["output_dir"]) / "trainer_log_history.jsonl"
        with log_path.open("w", encoding="utf-8") as f:
            for row in history:
                if isinstance(row, dict):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"📝 训练日志: {log_path}")

    print(f"✅ 全量微调完成: {cfg['output_dir']}")


if __name__ == "__main__":
    main()
