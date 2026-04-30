"""评测脚本共用的本机模型加载（CUDA / MPS / CPU）。

bitsandbytes 4bit 仅适用于 NVIDIA+CUDA；Apple Silicon 上应走 float16 + MPS。

当 merged 模型包含 bitsandbytes 量化权重（训练时 A100 保存）时，
自动回退到 base model + LoRA/DoRA adapter 加载。"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _mps_ok() -> bool:
    b = getattr(torch.backends, "mps", None)
    return b is not None and b.is_available()


def _has_quantized_weights(model_path: str) -> bool:
    """检查 safetensors 是否包含 uint8 量化权重。"""
    from safetensors import safe_open

    sf = Path(model_path) / "model.safetensors"
    if not sf.exists():
        return False
    try:
        f = safe_open(str(sf), framework="pt")
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype == torch.uint8:
                return True
    except Exception:
        pass
    return False


def _find_adapter_dir(model_path: str) -> list[str]:
    """根据 merged 模型路径找到对应的 adapter 目录。

    约定：
      outputs/sft_merged → outputs/sft
      outputs/merged     → outputs/sft + outputs/dpo
    """
    p = Path(model_path).resolve()
    adapters = []

    if p.name == "sft_merged":
        sft_dir = p.parent / "sft"
        if (sft_dir / "adapter_config.json").exists():
            adapters.append(str(sft_dir))
    elif p.name == "merged":
        # DPO 模型：先 SFT adapter，再 DPO adapter
        sft_dir = p.parent / "sft"
        dpo_dir = p.parent / "dpo"
        if (sft_dir / "adapter_config.json").exists():
            adapters.append(str(sft_dir))
        if (dpo_dir / "adapter_config.json").exists():
            adapters.append(str(dpo_dir))

    return adapters


def load_model_and_tokenizer(model_path: str, load_in_4bit: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if load_in_4bit and not torch.cuda.is_available():
        warnings.warn(
            "4bit 需 NVIDIA+CUDA；当前环境无 CUDA，已自动改为半精度全量加载（MPS/CPU）。"
            " Apple 上的 bitsandbytes 通常无 GPU 量化，请勿使用 --load_in_4bit。",
            UserWarning,
            stacklevel=2,
        )
        load_in_4bit = False

    # 清除 config.json 中内嵌的 quantization_config
    cfg_path = Path(model_path) / "config.json"
    cfg_dict = json.loads(cfg_path.read_text(encoding="utf-8"))
    if "quantization_config" in cfg_dict:
        del cfg_dict["quantization_config"]
        cfg_path.write_text(json.dumps(cfg_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    # 检测量化权重：如果 merged 模型包含 uint8 权重且不在 CUDA 上，
    # 回退到 base model + adapter 加载
    use_adapter_fallback = False
    adapter_dirs = []
    if not load_in_4bit and not torch.cuda.is_available():
        if _has_quantized_weights(model_path):
            adapter_dirs = _find_adapter_dir(model_path)
            if adapter_dirs:
                use_adapter_fallback = True
                warnings.warn(
                    f"检测到量化权重，回退到 base model + adapter: {adapter_dirs}",
                    UserWarning,
                    stacklevel=2,
                )

    if use_adapter_fallback:
        return _load_with_adapters(adapter_dirs, tokenizer)

    model_kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        if torch.cuda.is_available() or _mps_ok():
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = torch.float32

    model_kwargs["ignore_mismatched_sizes"] = True
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _load_with_adapters(adapter_dirs: list[str], tokenizer):
    """加载 base model + 一个或多个 LoRA/DoRA adapter。"""
    from peft import PeftModel

    dtype = torch.float16 if (_mps_ok() or torch.cuda.is_available()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    for adapter_dir in adapter_dirs:
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"  已加载 adapter: {adapter_dir}")

    # 合并 adapter 到 base（推理时不需要 adapter 开销）
    model = model.merge_and_unload()
    print(f"  adapter 已合并到 base model")
    return model, tokenizer
