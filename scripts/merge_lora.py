import argparse
import json
import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def read_base_model(adapter_path: str) -> str:
    """从 adapter_config.json 读取 base_model_name_or_path。"""
    cfg_file = os.path.join(adapter_path, "adapter_config.json")
    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            return json.load(f).get("base_model_name_or_path", adapter_path)
    except Exception:
        return adapter_path


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 到基础模型")
    parser.add_argument("--adapter_path", required=True,  help="LoRA adapter 目录（outputs/sft 等）")
    parser.add_argument("--output_path",  required=True,  help="合并后 fp16 模型输出目录")
    parser.add_argument("--config",       default="config/dpo_config.yaml", help="读取 max_seq_length / hf_token 等参数")
    parser.add_argument("--save_method",  default="merged_16bit",
                        choices=["merged_16bit", "merged_4bit", "lora"],
                        help="保存格式：merged_16bit（默认）/ merged_4bit / lora（仅保存 adapter）")
    args = parser.parse_args()

    # 读取配置
    hf_token = os.environ.get("HF_TOKEN", "")
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        hf_token = cfg.get("hf_token", hf_token).strip()
    except Exception:
        pass

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # 从 adapter_config.json 读取 base model（如 Qwen/Qwen2.5-1.5B-Instruct）
    base_model = read_base_model(args.adapter_path)
    print(f"📥 加载 base model: {base_model}  (fp16)")

    if args.save_method == "lora":
        # 仅复制 adapter 文件（无需加载完整模型）
        import shutil
        shutil.copytree(args.adapter_path, args.output_path, dirs_exist_ok=True)
        print(f"✅ LoRA adapter 已复制至: {args.output_path}")
        return

    # 用 transformers 直接加载 fp16 base model（命中 HF 缓存，不重复下载）
    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        token=hf_token or None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        token=hf_token or None,
    )

    print(f"🔗 载入 LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("🔀 合并 LoRA 权重...")
    model = model.merge_and_unload()
    # 加载时已指定 torch_dtype=float16，无需再次转换

    print(f"💾 保存合并模型 → {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)

    # transformers 5.x 的 save_pretrained 在合并 PEFT 模型后会触发
    # revert_weight_conversion 的 NotImplementedError，直接用 safetensors 绕开。
    # 用 save_model 而非 save_file，自动处理 tied weights（lm_head/embed_tokens 共享内存）。
    from safetensors.torch import save_model
    save_model(model, os.path.join(args.output_path, "model.safetensors"))

    # 配置文件 / tokenizer 仍走正常路径
    model.config.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    # 同步 generation_config（如有）
    if hasattr(model, "generation_config"):
        model.generation_config.save_pretrained(args.output_path)

    print(f"✅ 合并完成: {args.output_path}")


if __name__ == "__main__":
    main()
