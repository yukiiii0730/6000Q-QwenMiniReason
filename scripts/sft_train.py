import argparse
import yaml
import torch
import os
from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir): return None
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not checkpoints: return None
    last_cp = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    if not os.path.exists(os.path.join(last_cp, "trainer_state.json")):
        print(f"⚠️ Warning: {last_cp} is corrupted. Starting fresh.")
        return None
    return last_cp

DEFAULT_INSTRUCTION = "请先进行清晰的逐步推理，再给出最终答案。"
PROMPT_TEMPLATE = """### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"""

def normalize_example(example: dict) -> dict:
    if "problem" in example and "solution" in example: return {"instruction": DEFAULT_INSTRUCTION, "input": str(example["problem"]).strip(), "output": str(example["solution"]).strip()}
    if "instruction" in example and "response" in example: return {"instruction": str(example["instruction"]).strip(), "input": "", "output": str(example["response"]).strip()}
    return {"instruction": str(example.get("instruction", DEFAULT_INSTRUCTION)).strip(), "input": str(example.get("input", "")).strip(), "output": str(example.get("output", "")).strip()}

def load_sft_datasets(cfg: dict):
    if "dataset_path" in cfg: return load_dataset("json", data_files=cfg["dataset_path"], split="train")
    parts = []
    for ds_cfg in cfg["datasets"]:
        ds = load_dataset(ds_cfg["name"], split=ds_cfg.get("split", "train"))
        max_n = ds_cfg.get("max_samples")
        if max_n and max_n < len(ds): ds = ds.select(range(max_n))
        parts.append(ds)
    return concatenate_datasets(parts).shuffle(seed=42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sft_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("hf_token"): os.environ["HF_TOKEN"] = cfg["hf_token"]
    
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=cfg["model_name"], max_seq_length=cfg["max_seq_length"], load_in_4bit=cfg["load_in_4bit"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = FastLanguageModel.get_peft_model(model, r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=0, target_modules=cfg["lora"]["target_modules"], use_gradient_checkpointing="unsloth", random_state=cfg["seed"])
    
    ds = load_sft_datasets(cfg).map(lambda ex: {"text": PROMPT_TEMPLATE.format(**normalize_example(ex)) + tokenizer.eos_token}, remove_columns=load_sft_datasets(cfg).column_names)
    
    train_args = SFTConfig(output_dir=cfg["output_dir"], per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"], gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"], warmup_steps=cfg["train"]["warmup_steps"], max_steps=cfg["train"]["max_steps"], learning_rate=float(cfg["train"]["learning_rate"]), logging_steps=cfg["train"]["logging_steps"], save_steps=cfg["train"].get("save_steps", 100), save_total_limit=cfg["train"].get("save_total_limit", 3), weight_decay=float(cfg["train"]["weight_decay"]), fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(), report_to="none", dataset_text_field="text")
    
    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, args=train_args)
    last_cp = get_last_checkpoint(cfg["output_dir"])
    trainer.train(resume_from_checkpoint=last_cp)
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    main()
