"""SFT 监督微调 (Unsloth + TRL)

【v2 主要更新】
1. 支持 DoRA：cfg.lora.use_dora=true 时启用 weight-decomposed LoRA。
   优先尝试 Unsloth 原生支持；不兼容时 fallback 到 peft.LoraConfig(use_dora=True)。
2. 支持 stages 两段式课程：cfg.stages 列表里依次定义多个训练阶段，
   前一阶段产出的 adapter 自动作为下一阶段起点；每阶段可单独覆盖 lr/steps/warmup/dataset。
3. 数据集 field_map 灵活映射：兼容 OpenR1-Math、Magpie、Numina 多种字段。
"""

import argparse
import csv
import copy
import json
import os
import yaml
from pathlib import Path

import torch
from unsloth import FastLanguageModel  # 必须在 trl 之前导入
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTTrainer, SFTConfig


PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
DEFAULT_INSTRUCTION = "请先进行清晰的逐步推理，再给出最终答案。"


# =============================================================================
# Helpers
# =============================================================================
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def export_trainer_metrics(trainer, output_dir: str, tag: str = ""):
    history = list(getattr(trainer.state, "log_history", []) or [])
    if not history:
        print("⚠️  未检测到 trainer.log_history，跳过指标导出")
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    jsonl_path = out_dir / f"trainer_log_history{suffix}.jsonl"
    csv_path = out_dir / f"trainer_log_history{suffix}.csv"
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
    # 同步一份不带 tag 的最终版本，便于下游脚本统一定位
    if tag:
        out_dir.joinpath("trainer_log_history.jsonl").write_text(
            jsonl_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        out_dir.joinpath("trainer_log_history.csv").write_text(
            csv_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    print(f"📝 已导出训练指标: {jsonl_path}")


# =============================================================================
# Data loading & formatting
# =============================================================================
def normalize_example_with_map(example: dict, field_map: dict | None) -> dict:
    """将不同数据集的字段映射为 instruction/input/output。

    field_map 形如 {"prompt": "problem", "response": "solution"}。
    """
    if field_map:
        prompt_key = field_map.get("prompt", "instruction")
        response_key = field_map.get("response", "output")
        prompt = str(example.get(prompt_key, "")).strip()
        response = str(example.get(response_key, "")).strip()
        return {
            "instruction": DEFAULT_INSTRUCTION,
            "input": prompt,
            "output": response,
        }

    # 兼容老路径：自动推断
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
    return {
        "instruction": str(example.get("instruction", DEFAULT_INSTRUCTION)).strip(),
        "input": str(example.get("input", "")).strip(),
        "output": str(example.get("output", "")).strip(),
    }


def _pick_openr1_solution(example: dict) -> str:
    """OpenR1-Math-220k：从 generations 列表挑一条 verified 正确的。"""
    gens = example.get("generations") or []
    correctness = example.get("correctness_math_verify") or []
    for g, ok in zip(gens, correctness):
        if ok and isinstance(g, str) and g.strip():
            return g.strip()
    return str(example.get("solution") or (gens[0] if gens else "")).strip()


def _augment_openr1(ds):
    """如果是 OpenR1-Math 风格，把 generations 拍平成 solution 字段，便于统一映射。"""
    if "generations" in ds.column_names and "correctness_math_verify" in ds.column_names:
        return ds.map(lambda x: {"solution": _pick_openr1_solution(x)})
    return ds


def load_one_dataset(spec: dict):
    """支持三种来源：
    1) name=...      → 在线 HF datasets
    2) path=...      → 本地 JSON / JSONL 文件，可附 filter_source 过滤 source 字段
    3) name+path     → name 优先
    """
    if spec.get("name"):
        name = spec["name"]
        cfg_name = spec.get("config")
        split = spec.get("split", "train")
        print(f"📥 加载数据集 {name} (config={cfg_name}, split={split})")
        if cfg_name:
            ds = load_dataset(name, cfg_name, split=split)
        else:
            ds = load_dataset(name, split=split)
        ds = _augment_openr1(ds)
    elif spec.get("path"):
        path = spec["path"]
        print(f"📥 加载本地数据 {path}")
        ds = load_local_json(path)
        # 多源融合数据按 source 过滤（v4 五段课程用）
        filt = spec.get("filter_source")
        prefix_filt = spec.get("filter_source_prefix")
        if filt or prefix_filt:
            keep_exact = set(filt) if isinstance(filt, (list, tuple)) else ({filt} if filt else set())
            keep_prefix = tuple(prefix_filt) if isinstance(prefix_filt, (list, tuple)) else ((prefix_filt,) if prefix_filt else ())
            before = len(ds)
            def _match(x, keep_exact=keep_exact, keep_prefix=keep_prefix):
                src = x.get("source", "")
                if src in keep_exact:
                    return True
                if keep_prefix and src.startswith(keep_prefix):
                    return True
                return False
            ds = ds.filter(_match)
            tag = list(keep_exact) + [f"{p}*" for p in keep_prefix]
            print(f"   ✓ filter_source={tag}: {before} → {len(ds)} 条")
    else:
        raise ValueError(f"数据集 spec 缺少 name 或 path: {spec}")
    max_n = spec.get("max_samples")
    if max_n and max_n < len(ds):
        ds = ds.select(range(max_n))
    return ds


def load_local_json(path: str) -> Dataset:
    return load_dataset("json", data_files=path, split="train")


def make_text_dataset(ds, field_map, eos: str):
    def _to_text(ex):
        normed = normalize_example_with_map(ex, field_map)
        text = PROMPT_TEMPLATE.format(**normed) + eos
        return {"text": text}
    return ds.map(_to_text, remove_columns=ds.column_names)


# =============================================================================
# Model & PEFT
# =============================================================================
def attach_peft_adapter(model, lora_cfg: dict, seed: int):
    """优先用 Unsloth 原生路径；若启用 DoRA 但 Unsloth 不支持则回退 peft 原生 LoraConfig。"""
    use_dora = bool(lora_cfg.get("use_dora", False))

    # 路径 A: Unsloth 原生（速度最快）
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
        print(f"⚠️  Unsloth 不支持 use_dora ({e})，回退到 peft 原生 DoRA 路径")
    except Exception as e:  # noqa: BLE001
        if not use_dora:
            raise
        print(f"⚠️  Unsloth 路径异常 ({e})，回退到 peft 原生 DoRA 路径")

    # 路径 B: peft 原生（稳定 fallback）
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


def load_base_model(model_name: str, max_seq_length: int, load_in_4bit: bool):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# =============================================================================
# Stages
# =============================================================================
def expand_stages_from_cfg(cfg: dict) -> list[dict]:
    """构造统一的 stages 列表。
    - 若 cfg["stages"] 存在 → 直接使用，并合并默认 train 字段
    - 否则：把 cfg["datasets"] / cfg["dataset_path"] 包装为单段
    """
    base_train = dict(cfg.get("train") or {})

    if cfg.get("stages"):
        result = []
        for st in cfg["stages"]:
            stage_train = dict(base_train)
            stage_train.update(dict(st.get("train") or {}))
            result.append({
                "name": st.get("name", f"stage_{len(result)+1}"),
                "dataset": st.get("dataset"),
                "datasets": st.get("datasets"),
                "dataset_path": st.get("dataset_path"),
                "field_map": (st.get("dataset") or {}).get("field_map") if st.get("dataset") else None,
                "train": stage_train,
            })
        return result

    # 单段兜底
    if cfg.get("datasets"):
        return [{
            "name": "single_stage",
            "datasets": cfg["datasets"],
            "field_map": None,
            "train": base_train,
        }]
    return [{
        "name": "single_stage_local",
        "dataset_path": cfg.get("dataset_path"),
        "field_map": None,
        "train": base_train,
    }]


def build_stage_dataset(stage: dict, eos: str):
    """根据 stage 描述构造合并后的 text dataset。"""
    if stage.get("dataset"):
        ds = load_one_dataset(stage["dataset"])
        return make_text_dataset(ds, stage.get("field_map"), eos)
    if stage.get("datasets"):
        parts = []
        for ds_cfg in stage["datasets"]:
            ds = load_one_dataset(ds_cfg)
            parts.append(make_text_dataset(ds, ds_cfg.get("field_map"), eos))
        return concatenate_datasets(parts).shuffle(seed=42)
    if stage.get("dataset_path"):
        ds = load_local_json(stage["dataset_path"])
        return make_text_dataset(ds, stage.get("field_map"), eos)
    raise ValueError(f"Stage {stage.get('name')} 缺少数据来源")


def build_train_args(cfg: dict, stage_train: dict, output_dir: str) -> SFTConfig:
    train_cfg = stage_train
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        warmup_steps=int(train_cfg["warmup_steps"]),
        max_steps=int(train_cfg["max_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        logging_steps=int(train_cfg["logging_steps"]),
        save_steps=int(train_cfg["save_steps"]),
        weight_decay=float(train_cfg["weight_decay"]),
        lr_scheduler_type=str(train_cfg["lr_scheduler_type"]),
        optim=str(train_cfg["optim"]),
        fp16=bool(train_cfg["fp16"]),
        bf16=bool(train_cfg["bf16"]),
        report_to="none",
        seed=int(cfg["seed"]),
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=bool(train_cfg.get("dataloader_pin_memory", True)),
        dataset_text_field="text",
        packing=bool(train_cfg.get("packing", True)),
    )


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="SFT training with Unsloth (DoRA + Curriculum)")
    parser.add_argument("--config", default="config/sft_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    optimize_torch_runtime()

    hf_token = os.environ.get("HF_TOKEN") or (cfg.get("hf_token", "") or "").strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("✅ HuggingFace Token 已加载")

    output_dir_root = cfg["output_dir"]
    Path(output_dir_root).mkdir(parents=True, exist_ok=True)

    # bf16 自动降级 fp16
    if torch.cuda.is_available() and cfg.get("train", {}).get("bf16") and not torch.cuda.is_bf16_supported():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"⚠️  {gpu_name} 不支持 bf16，自动切换为 fp16")
        cfg["train"]["bf16"] = False
        cfg["train"]["fp16"] = True

    stages = expand_stages_from_cfg(cfg)
    print(f"📚 训练课程共 {len(stages)} 个阶段:")
    for i, st in enumerate(stages, 1):
        ds_repr = (
            (st.get("dataset") or {}).get("name")
            or [d.get("name") for d in (st.get("datasets") or [])]
            or st.get("dataset_path")
        )
        print(f"   Stage {i}: {st['name']} | data={ds_repr} | "
              f"max_steps={st['train'].get('max_steps')} | lr={st['train'].get('learning_rate')}")

    # 加载 base 模型 + LoRA
    model, tokenizer = load_base_model(
        cfg["model_name"],
        max_seq_length=cfg["max_seq_length"],
        load_in_4bit=cfg["load_in_4bit"],
    )
    tokenizer.model_max_length = cfg["max_seq_length"]

    model, peft_path = attach_peft_adapter(model, cfg["lora"], cfg["seed"])
    use_dora_flag = bool(cfg["lora"].get("use_dora", False))
    print(f"🔧 PEFT 路径: {peft_path}  | use_dora={use_dora_flag}")

    eos = tokenizer.eos_token

    for i, stage in enumerate(stages, 1):
        stage_dir = (
            output_dir_root if len(stages) == 1
            else os.path.join(output_dir_root, f"stage{i}_{stage['name']}")
        )
        Path(stage_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n========== Stage {i}/{len(stages)} : {stage['name']} ==========")
        print(f"📂 输出目录: {stage_dir}")

        ds = build_stage_dataset(stage, eos)
        train_args = build_train_args(cfg, stage["train"], stage_dir)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=ds,
            args=train_args,
        )

        resume_ckpt = find_latest_checkpoint(stage_dir)
        if resume_ckpt:
            print(f"♻️  检测到断点，继续训练: {resume_ckpt}")
            trainer.train(resume_from_checkpoint=resume_ckpt)
        else:
            trainer.train()

        export_trainer_metrics(trainer, stage_dir, tag=stage["name"])
        trainer.save_model(stage_dir)
        tokenizer.save_pretrained(stage_dir)

    # 最终 adapter 同步到 output_dir 根（让下游 merge_lora / dpo 能直接拿到）
    if len(stages) > 1:
        final_stage_dir = os.path.join(output_dir_root, f"stage{len(stages)}_{stages[-1]['name']}")
        # 最终阶段权重直接复制一份到根目录，保持原有 dpo_config.base_adapter_path 不变
        import shutil
        for fname in os.listdir(final_stage_dir):
            src = os.path.join(final_stage_dir, fname)
            dst = os.path.join(output_dir_root, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print(f"\n📌 最终 adapter 已同步至: {output_dir_root}")


if __name__ == "__main__":
    main()
