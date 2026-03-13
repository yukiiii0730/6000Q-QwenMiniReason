import argparse
import yaml
import torch
import os
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


def get_active_lora_config(model, fallback_cfg: dict | None = None) -> dict:
    """返回当前实际生效的 LoRA 配置，优先读取模型中已加载的适配器。"""
    peft_cfg = getattr(model, "peft_config", None)

    if isinstance(peft_cfg, dict) and peft_cfg:
        active_cfg = next(iter(peft_cfg.values()))
        return {
            "source": "loaded_adapter",
            "r": getattr(active_cfg, "r", None),
            "alpha": getattr(active_cfg, "lora_alpha", None),
            "dropout": getattr(active_cfg, "lora_dropout", None),
            "target_modules": list(getattr(active_cfg, "target_modules", []) or []),
        }

    if fallback_cfg is None:
        return {
            "source": "unknown",
            "r": None,
            "alpha": None,
            "dropout": None,
            "target_modules": [],
        }

    return {
        "source": "config",
        "r": fallback_cfg["r"],
        "alpha": fallback_cfg["alpha"],
        "dropout": fallback_cfg["dropout"],
        "target_modules": list(fallback_cfg["target_modules"]),
    }


def log_runtime_settings(base_name: str, tokenizer, model, lora_cfg: dict):
    print(f"ℹ️ 当前加载路径: {base_name}")
    print(
        "ℹ️ 当前实际 padding 设置: "
        f"pad_token={tokenizer.pad_token!r}, "
        f"pad_token_id={tokenizer.pad_token_id}, "
        f"padding_side={tokenizer.padding_side}"
    )
    print(
        "ℹ️ 当前实际 LoRA 设置: "
        f"source={lora_cfg['source']}, "
        f"r={lora_cfg['r']}, "
        f"alpha={lora_cfg['alpha']}, "
        f"dropout={lora_cfg['dropout']}, "
        f"target_modules={lora_cfg['target_modules']}"
    )
    print(f"ℹ️ 当前模型 pad_token_id: {getattr(model.config, 'pad_token_id', None)}")


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


def main():
    parser = argparse.ArgumentParser(description="DPO training with Unsloth")
    parser.add_argument("--config", default="config/dpo_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    optimize_torch_runtime()

    # 优先使用环境变量 HF_TOKEN（来自 .env）；其次读取 config
    hf_token = os.environ.get("HF_TOKEN") or cfg.get("hf_token", "").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("✅ HuggingFace Token 已加载")
    else:
        print("⚠️  未配置 HF_TOKEN，如需访问私有模型/数据集请在 .env 中设置 HF_TOKEN=hf_xxx")

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

    # 若 base_name 指向已训练好的 LoRA 目录（如 outputs/sft），模型会自带适配器。
    # 此时不能再次 get_peft_model，否则会触发 "already has LoRA adapters" 报错。
    has_existing_lora = bool(getattr(model, "peft_config", None))
    if has_existing_lora:
        print("ℹ️ 检测到已存在 LoRA 适配器，跳过新建 LoRA。")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=cfg["seed"],
        )

    active_lora_cfg = get_active_lora_config(model, cfg.get("lora"))
    log_runtime_settings(base_name, tokenizer, model, active_lora_cfg)

    ds = load_dpo_dataset(cfg)

    # T4 / Turing GPU 不支持 bf16，强制降级为 fp16
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if cfg["train"].get("bf16") and not torch.cuda.is_bf16_supported():
            print(f"⚠️  {gpu_name} 不支持 bf16，自动切换为 fp16")
            cfg["train"]["bf16"] = False
            cfg["train"]["fp16"] = True

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
        dataloader_num_workers=int(cfg["train"].get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(cfg["train"].get("dataloader_pin_memory", True)),
        group_by_length=bool(cfg["train"].get("group_by_length", True)),
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
