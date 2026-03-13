import argparse
import yaml
from unsloth import FastLanguageModel          # ← 必须在 trl/transformers 之前导入
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_dpo_example(example: dict) -> dict:
    """Orca-Math-Pairs 字段映射为 DPO 标准格式：prompt / chosen / rejected。

    数据集字段参考：
      question            → prompt
      correct_solution    → chosen
      incorrect_solution  → rejected
    """
    prompt = (
        example.get("question")
        or example.get("prompt")
        or example.get("input", "")
    )
    chosen = (
        example.get("correct_solution")
        or example.get("chosen")
        or example.get("answer", "")
    )
    rejected = (
        example.get("incorrect_solution")
        or example.get("rejected")
        or example.get("wrong_answer", "")
    )
    return {
        "prompt": str(prompt).strip(),
        "chosen": str(chosen).strip(),
        "rejected": str(rejected).strip(),
    }


def load_dpo_dataset(cfg: dict):
    """从 HuggingFace Hub 加载 DPO 数据集，兼容旧版本本地 JSON 格式。"""
    if "dataset" in cfg:
        ds_cfg = cfg["dataset"]
        ds = load_dataset(ds_cfg["name"], split=ds_cfg.get("split", "train"))
        max_n = ds_cfg.get("max_samples")
        if max_n and max_n < len(ds):
            ds = ds.select(range(max_n))
        ds = ds.map(normalize_dpo_example, remove_columns=ds.column_names)
        # 过滤採样中 rejected 为空的条目
        ds = ds.filter(lambda x: len(x["rejected"]) > 0)
        return ds
    # 将旧格式本地 JSON
    return load_dataset("json", data_files=cfg["dataset_path"], split="train")


def main():
    parser = argparse.ArgumentParser(description="DPO training with Unsloth")
    parser.add_argument("--config", default="config/dpo_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 从 config 读取 HuggingFace Token
    import os
    hf_token = cfg.get("hf_token", "").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("✅ HuggingFace Token 已加载")
    else:
        print("⚠️  未配置 hf_token，如需访问私有模型/数据集请先在 config/dpo_config.yaml 中填写")

    base_name = cfg.get("base_adapter_path") or cfg["model_name"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_name,
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

    ds = load_dpo_dataset(cfg)

    train_args = DPOConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        warmup_steps=cfg["train"]["warmup_steps"],
        max_steps=cfg["train"]["max_steps"],
        learning_rate=float(cfg["train"]["learning_rate"]),
        logging_steps=cfg["train"]["logging_steps"],
        save_steps=cfg["train"]["save_steps"],
        eval_steps=cfg["train"]["eval_steps"],
        weight_decay=float(cfg["train"]["weight_decay"]),
        lr_scheduler_type=cfg["train"]["lr_scheduler_type"],
        optim=cfg["train"]["optim"],
        fp16=bool(cfg["train"]["fp16"]),
        bf16=bool(cfg["train"]["bf16"]),
        beta=float(cfg["beta"]),
        max_length=cfg["max_seq_length"],
        max_prompt_length=min(1024, cfg["max_seq_length"] // 2),
        report_to="none",
        seed=cfg["seed"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
