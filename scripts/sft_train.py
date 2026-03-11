import argparse
import yaml
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel


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
            ds = load_dataset(ds_cfg["name"], split=ds_cfg.get("split", "train"), trust_remote_code=True)
            max_n = ds_cfg.get("max_samples")
            if max_n and max_n < len(ds):
                ds = ds.select(range(max_n))
            parts.append(ds)
        merged = concatenate_datasets(parts).shuffle(seed=42)
        return merged
    # 兼容旧格式：本地 JSON 文件
    return load_dataset("json", data_files=cfg["dataset_path"], split="train")


def main():
    parser = argparse.ArgumentParser(description="SFT training with Unsloth")
    parser.add_argument("--config", default="config/sft_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )

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
    ds = raw_ds.map(formatting_func, remove_columns=raw_ds.column_names)

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
        # SFTConfig 专属参数
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        max_seq_length=cfg["max_seq_length"],
        args=train_args,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
