"""Step 3: SFT 分阶段训练。

TA 建议：不要直接混合 NuminaMath + Magpie，改为分阶段训练：
  阶段 1：仅用 NuminaMath（基础数学逻辑对齐）
  阶段 2：在阶段 1 的基础上引入 Magpie-Reasoning（通用推理增强）

小模型容易在混合训练中产生"灾难性遗忘"或逻辑混淆，分阶段可以缓解。

用法：
    python scripts/staged_sft.py --config config/sft_config.yaml

    # 只跑阶段 1
    python scripts/staged_sft.py --config config/sft_config.yaml --stage 1

    # 从阶段 2 开始（已有阶段 1 产物）
    python scripts/staged_sft.py --config config/sft_config.yaml --stage 2
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

# ── TRL 依赖修复 ──────────────────────────────────────────────
for _mod_name in ("llm_blender", "llm_blender.blender", "llm_blender.blender.blender",
                   "llm_blender.blender.blender_utils", "mergekit", "mergekit.config"):
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import EarlyStoppingCallback


DEFAULT_INSTRUCTION = "请先进行清晰的逐步推理，再给出最终答案。"


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


def run_stage(cfg: dict, stage_name: str, data_path: str, model_name: str,
              output_dir: str, max_steps: int):
    """运行一个训练阶段"""
    print(f"\n{'='*60}")
    print(f"  阶段: {stage_name}")
    print(f"  数据: {data_path}")
    print(f"  模型: {model_name}")
    print(f"  输出: {output_dir}")
    print(f"  步数: {max_steps}")
    print(f"{'='*60}\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg.get("load_in_4bit", True),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 如果已有 LoRA adapter（阶段 2 从阶段 1 继续），不重复创建
    has_lora = bool(getattr(model, "peft_config", None))
    if not has_lora:
        lora_cfg = cfg.get("lora", {})
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.0),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            use_dora=lora_cfg.get("use_dora", False),
            use_gradient_checkpointing="unsloth",
            random_state=cfg.get("seed", 42),
        )

    # 加载数据
    raw_ds = load_dataset("json", data_files=data_path, split="train")
    eos = tokenizer.eos_token
    ds = raw_ds.map(
        lambda ex: {"text": formatting_func(ex, tokenizer) + eos},
        remove_columns=raw_ds.column_names
    )
    split = ds.train_test_split(test_size=0.05, seed=cfg.get("seed", 42))
    train_ds, eval_ds = split["train"], split["test"]
    print(f"📊 训练集: {len(train_ds)} | 验证集: {len(eval_ds)}")

    train_cfg = cfg.get("train", {})
    train_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 16),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        warmup_steps=train_cfg.get("warmup_steps", 100),
        max_steps=max_steps,
        learning_rate=float(train_cfg.get("learning_rate", 8e-5)),
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type="cosine",
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        bf16=True, fp16=False,
        report_to="none",
        seed=cfg.get("seed", 42),
        dataset_text_field="text",
        packing=True,
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

    resume_ckpt = find_latest_checkpoint(output_dir)
    if resume_ckpt:
        print(f"♻️  断点续训: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ {stage_name} 完成 → {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Step 3: SFT 分阶段训练")
    parser.add_argument("--config", default="config/sft_config.yaml")
    parser.add_argument("--stage", type=int, default=0, help="只运行指定阶段（1 or 2），0=全部")
    parser.add_argument("--numina_data", default="", help="阶段1 NuminaMath 数据路径")
    parser.add_argument("--magpie_data", default="", help="阶段2 Magpie 数据路径")
    parser.add_argument("--stage1_steps", type=int, default=1000)
    parser.add_argument("--stage2_steps", type=int, default=800)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_output = cfg.get("output_dir", "outputs/sft_staged")
    stage1_output = base_output + "_stage1"
    stage2_output = base_output + "_stage2"

    # 数据路径默认值
    numina_data = args.numina_data or "data/processed/sft_numina.json"
    magpie_data = args.magpie_data or "data/processed/sft_magpie.json"

    # 检查数据是否存在
    if not Path(numina_data).exists() and args.stage in (0, 1):
        print(f"❌ 阶段1 数据不存在: {numina_data}")
        print(f"   请先用 Cell 7 或 prepare_data.py 准备分开的数据集")
        print(f"   或用 --numina_data 指定路径")
        return

    # 阶段 1：NuminaMath 基础数学逻辑
    if args.stage in (0, 1):
        run_stage(
            cfg=cfg,
            stage_name="阶段1: NuminaMath 基础逻辑对齐",
            data_path=numina_data,
            model_name=cfg["model_name"],
            output_dir=stage1_output,
            max_steps=args.stage1_steps,
        )

    # 阶段 2：在阶段 1 基础上加 Magpie
    if args.stage in (0, 2):
        if not Path(magpie_data).exists():
            print(f"❌ 阶段2 数据不存在: {magpie_data}")
            return
        if not Path(stage1_output).exists() and args.stage == 2:
            print(f"❌ 阶段1 产物不存在: {stage1_output}")
            print(f"   请先运行阶段1")
            return

        run_stage(
            cfg=cfg,
            stage_name="阶段2: Magpie 通用推理增强",
            data_path=magpie_data,
            model_name=stage1_output,   # 从阶段 1 的输出继续
            output_dir=stage2_output,
            max_steps=args.stage2_steps,
        )

    print(f"\n🎉 分阶段训练完成！")
    print(f"   阶段1 模型: {stage1_output}")
    print(f"   阶段2 模型: {stage2_output}")


if __name__ == "__main__":
    main()
