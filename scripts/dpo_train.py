import argparse
import yaml
import torch
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_last_checkpoint(output_dir):
    if not os.path.exists(output_dir): return None
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))]
    if not checkpoints: return None
    last_cp = max(checkpoints, key=lambda x: int(x.split('-')[-1]))
    if not os.path.exists(os.path.join(last_cp, "trainer_state.json")):
        return None
    return last_cp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/dpo_config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("hf_token"): os.environ["HF_TOKEN"] = cfg["hf_token"]

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=cfg.get("base_adapter_path") or cfg["model_name"], max_seq_length=cfg["max_seq_length"], load_in_4bit=cfg["load_in_4bit"])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(model, r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=0, target_modules=cfg["lora"]["target_modules"], use_gradient_checkpointing="unsloth", random_state=cfg["seed"])

    ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    
    train_args = DPOConfig(output_dir=cfg["output_dir"], per_device_train_batch_size=cfg["train"]["per_device_train_batch_size"], gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"], warmup_steps=cfg["train"]["warmup_steps"], max_steps=cfg["train"]["max_steps"], learning_rate=float(cfg["train"]["learning_rate"]), logging_steps=cfg["train"]["logging_steps"], save_steps=cfg["train"].get("save_steps", 100), save_total_limit=3, fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(), beta=float(cfg["beta"]), max_length=cfg["max_seq_length"], max_prompt_length=cfg["max_seq_length"] // 2, report_to="none")

    trainer = DPOTrainer(model=model, ref_model=None, args=train_args, train_dataset=ds, processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=get_last_checkpoint(cfg["output_dir"]))
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    main()
