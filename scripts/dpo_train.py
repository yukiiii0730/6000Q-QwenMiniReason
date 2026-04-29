"""DPO 偏好优化训练 (Unsloth + TRL)

【v4 主要更新】
1. 支持 DoRA：cfg.lora.use_dora=true 时启用，与 SFT 训练保持一致。
2. 支持 loss_type 切换：sigmoid (DPO) / ipo / hinge / robust 等
   通过 cfg.loss_type 配置；TRL 的 DPOConfig 原生支持。
3. 支持 Error-Type-Weighted DPO（创新核心）：
   Targeted DPO 数据可带 error_type 字段；训练时按 cfg.error_type_weights
   把每条样本复制 N 份（按权重 ceil 处理），实现"修复难度高的错误类型
   给更大权重"。
4. 兼容多 schema 数据：argilla / Intel / Teacher-DPO / Targeted-DPO
"""
import argparse
import csv
import gc
import json
import os
from pathlib import Path

import torch
import yaml
from unsloth import FastLanguageModel  # noqa: F401  必须最先导入
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_dpo_example(example: dict) -> dict:
    """统一映射为 DPO 标准字段：prompt / chosen / rejected (+ error_type)。

    支持 schema：
    - argilla/distilabel-math-preference-dpo : instruction + chosen_response + rejected_response
    - argilla/ultrafeedback-binarized        : prompt + chosen + rejected (list/str)
    - 自建 Teacher-DPO                        : prompt + chosen + rejected (str)
    - Targeted-DPO（创新核心）                : prompt + chosen + rejected + error_type
    - Intel/orca_dpo_pairs                   : question + chosen + rejected
    """
    prompt = (
        example.get("prompt")
        or example.get("instruction")
        or example.get("question")
        or example.get("input")
        or ""
    )
    chosen = (
        example.get("chosen")
        or example.get("chosen_response")
        or example.get("correct_solution")
        or ""
    )
    rejected = (
        example.get("rejected")
        or example.get("rejected_response")
        or example.get("incorrect_solution")
        or ""
    )
    error_type = str(example.get("error_type", "") or "").strip()

    # ultrafeedback 风格的 list[{role,content}]
    def _flatten(x):
        if isinstance(x, list):
            return "\n".join(c.get("content", "") for c in x if isinstance(c, dict))
        return x

    chosen = _flatten(chosen)
    rejected = _flatten(rejected)

    return {
        "prompt": str(prompt).strip(),
        "chosen": str(chosen).strip(),
        "rejected": str(rejected).strip(),
        "error_type": error_type,
    }


def apply_error_type_weights(ds, weights: dict) -> "Dataset":  # noqa: F821
    """按 error_type 权重把样本复制（向上取整）实现 weighted DPO。
    例如 weights={"arithmetic": 1.0, "reasoning_skip": 2.0} → reasoning_skip 样本数翻倍。"""
    import math as _math
    if not weights:
        return ds
    counts = {}
    for ex in ds:
        et = ex.get("error_type", "")
        counts[et] = counts.get(et, 0) + 1
    print(f"📊 Error-type 权重应用前分布: {counts}")
    # 把字典权重 default = 1.0
    repeated_records = []
    for ex in ds:
        et = ex.get("error_type", "") or "_default"
        w = float(weights.get(et, weights.get("_default", 1.0)))
        n_copies = max(1, int(_math.ceil(w)))
        for _ in range(n_copies):
            repeated_records.append(dict(ex))
    from datasets import Dataset
    new_ds = Dataset.from_list(repeated_records).shuffle(seed=42)
    print(f"📊 加权后样本数: {len(ds)} → {len(new_ds)}")
    return new_ds


def load_dpo_dataset(cfg: dict):
    if "dataset" in cfg and isinstance(cfg["dataset"], dict):
        ds_cfg = cfg["dataset"]
        kwargs = {"split": ds_cfg.get("split", "train")}
        if ds_cfg.get("config"):
            ds = load_dataset(ds_cfg["name"], ds_cfg["config"], **kwargs)
        else:
            ds = load_dataset(ds_cfg["name"], **kwargs)
        max_n = ds_cfg.get("max_samples")
        if max_n and max_n < len(ds):
            ds = ds.select(range(max_n))
        ds = ds.map(normalize_dpo_example, remove_columns=ds.column_names)
    else:
        # 本地 JSON
        ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")
        ds = ds.map(normalize_dpo_example, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"] and x["chosen"] != x["rejected"])

    # Error-Type 权重加权（创新核心）
    weights = cfg.get("error_type_weights")
    if weights:
        ds = apply_error_type_weights(ds, weights)
    return ds


def get_active_lora_config(model, fallback_cfg: dict | None = None) -> dict:
    peft_cfg = getattr(model, "peft_config", None)
    if isinstance(peft_cfg, dict) and peft_cfg:
        active = next(iter(peft_cfg.values()))
        return {
            "source": "loaded_adapter",
            "r": getattr(active, "r", None),
            "alpha": getattr(active, "lora_alpha", None),
            "dropout": getattr(active, "lora_dropout", None),
            "target_modules": list(getattr(active, "target_modules", []) or []),
            "use_dora": getattr(active, "use_dora", False),
        }
    if fallback_cfg is None:
        return {"source": "unknown"}
    return {
        "source": "config",
        "r": fallback_cfg.get("r"),
        "alpha": fallback_cfg.get("alpha"),
        "dropout": fallback_cfg.get("dropout"),
        "target_modules": list(fallback_cfg.get("target_modules", [])),
        "use_dora": bool(fallback_cfg.get("use_dora", False)),
    }


def attach_peft_adapter(model, lora_cfg: dict, seed: int):
    """与 sft_train.py 中的逻辑一致：优先 Unsloth，DoRA 不兼容时 fallback peft。"""
    use_dora = bool(lora_cfg.get("use_dora", False))
    try:
        kwargs = dict(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
        if use_dora:
            kwargs["use_dora"] = True
        return FastLanguageModel.get_peft_model(model, **kwargs), "unsloth"
    except TypeError as e:
        if not use_dora:
            raise
        print(f"⚠️  Unsloth 不支持 use_dora ({e})，回退到 peft 原生 DoRA")
    except Exception as e:  # noqa: BLE001
        if not use_dora:
            raise
        print(f"⚠️  Unsloth 路径异常 ({e})，回退到 peft 原生 DoRA")

    from peft import LoraConfig, get_peft_model
    peft_cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=use_dora,
    )
    model = get_peft_model(model, peft_cfg)
    return model, "peft"


def optimize_torch_runtime():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def cleanup_training_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


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
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return str(ckpts[-1][1])


def export_trainer_metrics(trainer, output_dir: str):
    history = list(getattr(trainer.state, "log_history", []) or [])
    if not history:
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


def main():
    parser = argparse.ArgumentParser(description="DPO training with Unsloth (DoRA + Teacher-DPO)")
    parser.add_argument("--config", default="config/dpo_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    optimize_torch_runtime()

    hf_token = os.environ.get("HF_TOKEN") or (cfg.get("hf_token", "") or "").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("✅ HuggingFace Token 已加载")

    base_name = cfg.get("base_adapter_path") or cfg["model_name"]
    print(f"📥 加载基座: {base_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_name,
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    has_existing_lora = bool(getattr(model, "peft_config", None))
    if has_existing_lora:
        print("ℹ️ 检测到已存在 LoRA 适配器（来自 SFT），跳过新建")
    else:
        model, peft_path = attach_peft_adapter(model, cfg["lora"], cfg["seed"])
        print(f"🔧 PEFT 路径: {peft_path}, use_dora={cfg['lora'].get('use_dora', False)}")

    active_lora_cfg = get_active_lora_config(model, cfg.get("lora"))
    print(f"ℹ️ 活动 LoRA: {active_lora_cfg}")

    ds = load_dpo_dataset(cfg)
    print(f"📊 DPO 训练集: {len(ds)} 条")

    if torch.cuda.is_available() and cfg["train"].get("bf16") and not torch.cuda.is_bf16_supported():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"⚠️  {gpu_name} 不支持 bf16，自动切换为 fp16")
        cfg["train"]["bf16"] = False
        cfg["train"]["fp16"] = True

    seq_len = int(cfg["max_seq_length"])
    max_prompt_length = max(768, seq_len // 2)  # 修复：避免长 CoT chosen 截断

    loss_type = str(cfg.get("loss_type", "sigmoid")).lower()
    print(f"🎯 DPO loss_type = {loss_type}, beta = {cfg['beta']}")

    dpo_kwargs = dict(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=int(cfg["train"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["train"]["gradient_accumulation_steps"]),
        warmup_steps=int(cfg["train"]["warmup_steps"]),
        max_steps=int(cfg["train"]["max_steps"]),
        learning_rate=float(cfg["train"]["learning_rate"]),
        logging_steps=int(cfg["train"]["logging_steps"]),
        save_steps=int(cfg["train"]["save_steps"]),
        eval_steps=int(cfg["train"]["eval_steps"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        lr_scheduler_type=str(cfg["train"]["lr_scheduler_type"]),
        optim=str(cfg["train"]["optim"]),
        fp16=bool(cfg["train"]["fp16"]),
        bf16=bool(cfg["train"]["bf16"]),
        beta=float(cfg["beta"]),
        max_length=seq_len,
        max_prompt_length=max_prompt_length,
        report_to="none",
        seed=int(cfg["seed"]),
        dataloader_num_workers=int(cfg["train"].get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(cfg["train"].get("dataloader_pin_memory", True)),
    )
    # TRL 不同版本对 loss_type 字段的支持略有差异，做兼容处理
    try:
        train_args = DPOConfig(**dpo_kwargs, loss_type=loss_type)
    except TypeError:
        print(f"⚠️  DPOConfig 不支持 loss_type 参数（TRL 版本太旧），改用默认 sigmoid")
        train_args = DPOConfig(**dpo_kwargs)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=train_args,
        train_dataset=ds,
        processing_class=tokenizer,
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
    del ds
    del trainer
    cleanup_training_memory()


if __name__ == "__main__":
    main()
