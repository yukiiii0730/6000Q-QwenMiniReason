# Qwen-Reasoning-Enhance

基于 **Unsloth + TRL** 对 Qwen2.5-1.5B-Instruct 进行两阶段微调（SFT → DPO），提升模型在数学推理（GSM8K）及综合推理（BBH）任务上的能力。

---

## 实验结果（实测）

> 评测数据：GSM8K / BBH 各 50 条（固定 seed 的分层抽样）

| 模型阶段 | GSM8K Acc | BBH Acc |
|---|---|---|
| Qwen2.5-1.5B-Instruct（Baseline） | **32.0%** | **82.0%** |
| + SFT only | — | — |
| + **SFT + DPO**（完整两阶段） | **46.0%** | **84.0%** |
| Δ vs Baseline | **+14.0pp** | **+2.0pp** |

> 完整四行对比表由 `run_train.sh` 自动打印，结果存入 `logs/compare_metrics.json`。

---

## 项目结构

```
Qwen-Reasoning-Enhance/
├── run_train.sh              # 🚀 一键全流程脚本（推荐入口）
├── config/
│   ├── sft_config.yaml       # SFT 超参数（LoRA r=16, alpha=32 等）
│   ├── dpo_config.yaml       # DPO 超参数
│   └── benchmark_models.yaml # 7B/14B 对照评测配置（本地+API）
├── scripts/
│   ├── prepare_data.py       # 数据集下载与预处理
│   ├── sft_train.py          # 第一阶段：SFT（监督微调）
│   ├── dpo_train.py          # 第二阶段：DPO（直接偏好优化）
│   └── merge_lora.py         # LoRA 权重合并（transformers + peft）
├── eval/
│   ├── gsm8k_eval.py         # GSM8K 自动评测
│   ├── bbh_eval.py           # BBH 自动评测（含 CoT 答案提取）
│   ├── gsm8k_api_eval.py     # GSM8K API 评测（OpenAI-compatible）
│   ├── bbh_api_eval.py       # BBH API 评测（OpenAI-compatible）
│   ├── benchmark_open_models.py # 本地开源模型批量评测
│   ├── compare_table.py      # 统一对比表输出
│   └── visualize.py          # 雷达图 & Loss 曲线可视化
├── data/
│   ├── raw/                  # 原始数据占位目录
│   ├── processed/            # 预处理后的训练数据（quick 模式本地 JSON）
│   └── process.py            # 格式转换脚本（→ Alpaca / DPO 格式）
├── notebooks/
│   ├── colab_train.ipynb     # Colab 一键训练示例
│   └── inference_test.ipynb  # 零样本 vs 微调模型推理对比
└── requirements.txt
```

---

## 环境安装

> 推荐 Python 3.10+，GPU 显存 ≥ 16 GB（A100 / L20 / 4090 最佳；T4 可运行但较慢）。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

在 `config/sft_config.yaml` 与 `config/dpo_config.yaml` 中填写 HuggingFace Token：

```yaml
hf_token: hf_YOUR_TOKEN_HERE   # 从 https://huggingface.co/settings/tokens 获取
```

并在 `config/benchmark_models.yaml` 中填写 7B/14B API 评测配置：

```yaml
api_evaluation:
    api_base_url: https://your-openai-compatible-endpoint/v1
    api_key: YOUR_API_KEY
    reference_models:
        model_7b:
            model: your-7b-model-name
        model_14b:
            model: your-14b-model-name
```

---

## 快速开始

### 一键全流程（推荐）

```bash
# 快速测试（本地 JSON，500 条，SFT=50 steps，DPO=30 steps，约 10 分钟）
bash run_train.sh --quick

# 完整训练（HuggingFace 在线数据集，各 50k 条，生产级超参）
bash run_train.sh
```

完整流程自动执行以下步骤：

| 步骤 | 描述 | 产物 |
|---|---|---|
| **Step 0a** | Baseline 评测（训练前原始模型） | `logs/gsm8k_baseline.json`, `logs/bbh_baseline.json` |
| **Step 0b** | 7B/14B 参考模型 API 评测 | `logs/gsm8k_qwen25_7b.json`, `logs/bbh_qwen25_7b.json`, `logs/gsm8k_qwen25_14b.json`, `logs/bbh_qwen25_14b.json` |
| **Step 1** | 数据集下载与预处理 | `data/processed/` |
| **Step 2** | SFT 监督微调 | `outputs/sft/` |
| **Step 3** | DPO 偏好优化 | `outputs/dpo/` |
| **Step 4a** | 合并 SFT adapter → 评测 SFT 单独效果 | `outputs/sft_merged/`, `logs/gsm8k_sft.json`, `logs/bbh_sft.json` |
| **Step 4b** | 合并最终 SFT+DPO 权重 | `outputs/merged/` |
| **Step 5** | 最终评测 + 四行对比表 | `logs/gsm8k_result.json`, `logs/bbh_result.json`, `logs/compare_metrics.json` |

所有步骤均支持**断点续跑**（自动检测已有产物，跳过已完成阶段）。

### 常用参数

```bash
bash run_train.sh --skip-data        # 跳过数据下载（已有缓存时）
bash run_train.sh --skip-sft         # 跳过 SFT（使用已有 outputs/sft/）
bash run_train.sh --skip-baseline    # 跳过 Baseline 评测
bash run_train.sh --only-sft         # 仅训练 SFT，不做 DPO
bash run_train.sh --force            # 强制重跑所有阶段
bash run_train.sh --sft_n 20000 --dpo_n 20000  # 自定义采样量
```

---

## 数据集

### SFT 数据集（完整训练模式）

训练脚本在完整模式下**直接从 HuggingFace Hub 在线拉取**，无需手动下载：

| 数据集 | HuggingFace ID | 字段 | 说明 |
|---|---|---|---|
| **NuminaMath-CoT** | [`AI-MO/NuminaMath-CoT`](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | `problem` / `solution` | 数学竞赛题 + 详细 CoT 推理，约 86 万条 |
| **Magpie-Reasoning-150K** | [`Magpie-Align/Magpie-Reasoning-150K`](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-150K) | `instruction` / `response` | 蒸馏自 Llama-3.1-70B，逻辑密度高，15 万条 |

默认各取 5 万条（通过 `--sft_n` 自定义）。

### DPO 数据集（完整训练模式）

| 数据集 | HuggingFace ID | 字段 | 说明 |
|---|---|---|---|
| **Orca-Math-Pairs** | [`microsoft/orca-math-word-problems-200k`](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | `question` / `correct_solution` / `incorrect_solution` | 专为 DPO 设计，含正确和错误推理路径，20 万条 |

自动映射为 `prompt / chosen / rejected` 格式，默认取 5 万条（`--dpo_n` 自定义）。

### 快速测试模式数据集

`--quick` 模式使用本地预生成的 JSON 文件（500 条），由 `scripts/prepare_data.py` 自动创建：

- SFT: `data/processed/sft_train.json`（NuminaMath-CoT + Magpie 各 250 条）
- DPO: `data/processed/dpo_train.json`（orca_dpo_pairs 500 条）

---

## 手动运行各步骤

若需精细控制，可手动运行各脚本：

### 数据准备

```bash
python scripts/prepare_data.py --sft_n 50000 --dpo_n 50000
```

### SFT 训练

```bash
python scripts/sft_train.py --config config/sft_config.yaml
```

关键超参数（`config/sft_config.yaml`）：

| 参数 | 值 |
|---|---|
| LoRA r / alpha | 16 / 32 |
| Learning rate | 2e-4 |
| Batch size | 2 × 8 grad_accum |
| Max steps（完整）| ~1000（quick: 50）|
| Optimizer | paged_adamw_8bit |
| Quantization | 4-bit NF4 |

### DPO 训练

```bash
python scripts/dpo_train.py --config config/dpo_config.yaml
```

| 参数 | 值 |
|---|---|
| β (beta) | 0.1 |
| Learning rate | 1e-5 |
| Max steps（完整）| ~800（quick: 30）|
| 基座 | outputs/sft（SFT 产物）|

### 合并 LoRA 权重

`merge_lora.py` 使用 `transformers + peft` 直接合并（不依赖 Unsloth fp16 权重下载），自动读取 `adapter_config.json` 中的 base model 路径：

```bash
# 合并 SFT adapter
python scripts/merge_lora.py \
    --adapter_path outputs/sft \
    --output_path  outputs/sft_merged

# 合并 SFT+DPO（完整两阶段）
python scripts/merge_lora.py \
    --adapter_path outputs/dpo \
    --output_path  outputs/merged
```

---

## 评估

### GSM8K

```bash
python eval/gsm8k_eval.py \
    --model_path  outputs/merged \
    --max_samples 50 \
    --output      logs/gsm8k_result.json
```

### BBH

```bash
python eval/bbh_eval.py \
    --model_path  outputs/merged \
    --subset      boolean_expressions \
    --max_samples 50 \
    --output      logs/bbh_result.json
```

BBH 评测内置三级答案匹配（前缀匹配 → 末尾 200 字符 → 全文搜索），兼容 CoT 长输出。

为降低评测耗时，默认使用 `max_samples=50` + 固定 `seed=42` 的分层抽样（按题目长度分桶），比“前 50 条”更稳健、可复现。

### 与 7B / 14B 开源模型做横向对照

统一配置在 [config/benchmark_models.yaml](config/benchmark_models.yaml)：

- `api_evaluation`：供 `run_train.sh` 的 Step 0b 使用（API 评测）
- `evaluation + models`：供本地批量评测脚本使用

本地批量评测运行方式：

```bash
python eval/benchmark_open_models.py --config config/benchmark_models.yaml
```

结果会写入 `logs/open_model_benchmarks/`。

### 可视化

```bash
python eval/visualize.py \
    --metrics_json logs/compare_metrics.json \
    --out_dir      eval/figures
```

---

## 快速推理测试

打开 `notebooks/inference_test.ipynb`，可直接对比同一道题在基础模型与微调模型上的输出差异。

---

## Google Colab 使用指南

**不需要提前下载数据集**，Colab 可直接通过 `datasets` 库从 HuggingFace Hub 在线拉取。

### 推荐流程

```python
# 1. 安装依赖（每次新 Session 需要）
!pip install unsloth trl peft bitsandbytes transformers datasets accelerate pyyaml safetensors -q

# 2. 克隆本项目
!git clone https://github.com/your-repo/Qwen-Reasoning-Enhance.git
%cd Qwen-Reasoning-Enhance

# 3. 设置 HuggingFace Token（使用 Colab Secrets，避免硬编码）
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# 4. 运行快速测试
!bash run_train.sh --quick
```

> **Colab 免费版**（T4，15GB）可正常运行 Qwen2.5-1.5B-Instruct 的 4-bit 量化训练（约 6-8 GB 显存）。
> **Colab Pro/Pro+**（A100）速度提升约 5-8×。

挂载 Google Drive 持久化保存检查点：

```python
from google.colab import drive
drive.mount('/content/drive')
# 在 config yaml 中将 output_dir 改为 Drive 路径：
# output_dir: "/content/drive/MyDrive/Qwen-Reasoning/outputs/sft"
```

---

## 注意事项

- `unsloth` 仅支持 Linux / WSL2 + CUDA，macOS 仅用于代码开发，训练需在 GPU 机器上运行。
- 4-bit 量化训练时 `bf16=true` 与 `fp16=true` 不可同时开启；T4 不支持 bf16，`run_train.sh` 会自动检测并降级为 fp16。
- DPO 训练时 `ref_model=None` 表示使用 SFT 模型自身作为参考（Unsloth 内置 PEFT 参考实现）。
- **请勿将 HuggingFace Token 明文提交**，在 `config/*.yaml` 中填写 `YOUR_HF_TOKEN_HERE` 占位符，使用环境变量或 Colab Secrets 管理真实 Token。
- `merge_lora.py` 使用 `safetensors.torch.save_model` 保存（而非 `save_file`），可正确处理 `lm_head` / `embed_tokens` 权重共享问题。

---

## 参考资料

- [Unsloth 官方文档](https://github.com/unslothai/unsloth)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [GSM8K Dataset](https://huggingface.co/datasets/gsm8k)
- [BIG-Bench Hard (BBH)](https://huggingface.co/datasets/lukaemon/bbh)
- [Qwen2.5-1.5B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [Magpie-Reasoning-150K](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-150K)
- [Orca-Math-Word-Problems-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k)
