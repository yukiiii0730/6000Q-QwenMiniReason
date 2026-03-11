import argparse
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 到基础模型")
    parser.add_argument("--adapter_path", required=True, help="LoRA/PEFT 模型目录")
    parser.add_argument("--output_path", required=True, help="合并后模型输出目录")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print(f"合并完成: {args.output_path}")


if __name__ == "__main__":
    main()
