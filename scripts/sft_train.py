import argparse
import csv
import json
import yaml
import torch
import os
from pathlib import Path
from unsloth import FastLanguageModel          # ← 必须在 trl 之前导入
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig


PROMPT_TEMPLATE = """### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"""

DEFAULT_INSTRUCTION = "请先进行清晰的逐步推理，再给出最终答案。"


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_example(example: dict) -> dict:
    """将不同数据集的字段统一映射为 instruction / input / output。

    NuminaMath-CoT : problem + solution
    Magpie-Reasoning: instruction + response
    通用 Alpaca    : instruction + input + output
    """
    if "problem" in example and "solution" in example:
        return {
            "instruction": DEFAULT_INSTRUCTION,
            "input": str(example["problem"]).strip(),
            "output": str(example["solution"]).strip(),
        }
    if "instruction" in example and "response" in example:
        return {
            "instruction": str(example["instruction"]).strip(),
            "input": "",
            "output": str(example["response"]).strip(),
        }
    # 通用 Alpaca 格式兜底
    return {
        "instruction": str(example.get("instruction", DEFAULT_INSTRUCTION)).strip(),
        "input": str(example.get("input", "")).strip(),
        "output": str(example.get("output", "")).strip(),
    }


def formatting_func(example):
    normed = normalize_example(example)
    text = PROMPT_TEMPLATE.format(**normed)
    return {"text": text}


def load_sft_datasets(cfg: dict):
    """从 HuggingFace Hub 加载并合并所有 SFT 数据集。"""
    # 新格式：datasets 列表（在线加载）
    if "datasets" in cfg:
        parts = []
        for ds_cfg in cfg["datasets"]:
            ds = load_dataset(ds_cfg["name"], split=ds_cfg.get("split", "train"))
            max_n = ds_cfg.get("max_samples")
            if max_n and max_n < len(ds):
                ds = ds.select(range(max_n))
            parts.append(ds)
        merged = concatenate_datasets(parts).shuffle(seed=42)
        return merged
    # 兼容旧格式：本地 JSON 文件
    return load_dataset("json", data_files=cfg["dataset_path"], split="train")


def optimize_torch_runtime():
    """启用安全的运行时优化，提高 GPU 利用率与吞吐。"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def find_latest_checkpoint(output_dir: str):
    out = Path(output_dir)
    if not out.exists():
        return None
    ckpts = []
    for p in out.glob("checkpoint-*"):
        if p.is_dir():
            try:
                step = int(p.name.split("-")[-1])
                ckpts.append((step, p))
            except ValueError:
                continue
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return str(ckpts[-1][1])


def export_trainer_metrics(trainer, output_dir: str):
    """导出训练过程指标，便于后续对比与可视化。"""
    history = list(getattr(trainer.state, "log_history", []) or [])
    if not history:
        print("⚠️  未检测到 trainer.log_history，跳过指标导出")
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "trainer_log_history.jsonl"
    csv_path = out_dir / "trainer_log_history.csv"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in history:
            if isinstance(row, dict):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    keys = sorted({k for row in history if isinstance(row, dict) for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            if isinstance(row, dict):
                writer.writerow({k: row.get(k) for k in keys})

    print(f"📝 已导出训练指标: {jsonl_path}")
    print(f"📝 已导出训练指标: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="SFT training with Unsloth")
    parser.add_argument("--config", default="config/sft_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    optimize_torch_runtime()

    # 优先使用环境变量 HF_TOKEN（来自 .env）；其次读取 config
    import os
    hf_token = os.environ.get("HF_TOKEN") or cfg.get("hf_token", "").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("✅ HuggingFace Token 已加载")
    else:
        print("⚠️  未配置 HF_TOKEN，如需访问私有模型/数据集请在 .env 中设置 HF_TOKEN=hf_xxx")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )

    # 显式设置 padding token，避免部分 Qwen 量化权重缺失 pad_token 的警告
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )

    raw_ds = load_sft_datasets(cfg)
    eos = tokenizer.eos_token
    ds = raw_ds.map(
        lambda ex: {"text": formatting_func(ex)["text"] + eos},
        remove_columns=raw_ds.column_names
    )

    # T4 / Turing GPU 不支持 bf16，强制降级为 fp16
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if cfg["train"].get("bf16") and not torch.cuda.is_bf16_supported():
            print(f"⚠️  {gpu_name} 不支持 bf16，自动切换为 fp16")
            cfg["train"]["bf16"] = False
            cfg["train"]["fp16"] = True

    train_args = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        warmup_steps=cfg["train"]["warmup_steps"],
        max_steps=cfg["train"]["max_steps"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        weight_decay=float(cfg["train"]["weight_decay"]),
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        optim=cfg["train"]["optim"],
        fp16=bool(cfg["train"]["fp16"]),
        bf16=bool(cfg["train"]["bf16"]),
        report_to="none",
        seed=cfg["seed"],
        dataloader_num_workers=int(cfg["train"].get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(cfg["train"].get("dataloader_pin_memory", True)),
        # SFTConfig 专属参数
        dataset_text_field="text",
        packing=bool(cfg["train"].get("packing", True)),
    )

    # 通过 tokenizer 控制最大序列长度（兼容各版本 TRL）
    tokenizer.model_max_length = cfg["max_seq_length"]

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        args=train_args,
    )

    resume_ckpt = find_latest_checkpoint(cfg["output_dir"])
    if resume_ckpt:
        print(f"♻️  检测到断点，继续训练: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    export_trainer_metrics(trainer, cfg["output_dir"])
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
