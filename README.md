# Qwen-Reasoning-Enhance

基于 **Unsloth + TRL** 对 Qwen2.5-1.5B-Instruct 进行两阶段微调（SFT → DPO），提升模型在数学推理（GSM8K）及综合推理（BBH）任务上的能力。

---

## 项目结构

```
Qwen-Reasoning-Enhance/
├── data/
│   ├── raw/                  # 原始数据（GSM8K、Orca-Math-Pairs 等）
│   ├── processed/            # 预处理后的训练数据
│   └── process.py            # 格式转换脚本（→ Alpaca / DPO 格式）
├── scripts/
│   ├── sft_train.py          # 第一阶段：SFT（监督微调）
│   ├── dpo_train.py          # 第二阶段：DPO（直接偏好优化）
│   └── merge_lora.py         # LoRA 权重合并脚本
├── eval/
│   ├── gsm8k_eval.py         # GSM8K 自动评测
│   ├── bbh_eval.py           # BBH 自动评测
│   └── visualize.py          # 雷达图 & Loss 曲线可视化
├── notebooks/
│   └── inference_test.ipynb  # 零样本 vs 微调模型推理对比
├── config/
│   ├── sft_config.yaml       # SFT 超参数（LoRA r=16, alpha=32 等）
│   └── dpo_config.yaml       # DPO 超参数
└── requirements.txt
```

---

## 环境安装

> 推荐 Python 3.10+，GPU 显存 ≥ 16GB（A100/4090 最佳）。

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含：`unsloth`, `trl`, `peft`, `bitsandbytes`, `transformers`, `datasets`, `accelerate` 等。

---

## 数据准备

### SFT 数据集

训练脚本会**直接从 HuggingFace Hub 在线拉取**，无需手动下载。两个数据集在 `config/sft_config.yaml` 的 `datasets` 字段中配置：

| 数据集 | HuggingFace ID | 字段 | 说明 |
|---|---|---|---|
| **NuminaMath-CoT** | [`AI-MO/NuminaMath-CoT`](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | `problem` / `solution` | 数学竞赛题 + 详细 CoT 推理，约 86万条，**首选 SFT 起始集** |
| **Magpie-Reasoning-150K** | [`Magpie-Align/Magpie-Reasoning-150K`](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-150K) | `instruction` / `response` | 蒸馏自 Llama-3.1-70B，逻辑密度高，15万条 |

每个数据集可通过 `max_samples` 控制采样量（默认各取 5万条），按需在 `config/sft_config.yaml` 中调整。

### DPO 数据集

| 数据集 | HuggingFace ID | 字段 | 说明 |
|---|---|---|---|
| **Orca-Math-Pairs** | [`microsoft/orca-math-word-problems-200k`](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | `question` / `correct_solution` / `incorrect_solution` | 专为 DPO 设计，含正确和错误推理路径，20 万条 |

训练脚本会直接从 HuggingFace Hub 在线拉取，自动映射为 DPO 所需的 `prompt / chosen / rejected` 格式，**无需手动下载。**

采样量可在 `config/dpo_config.yaml` 的 `max_samples` 中调整（默认 5 万条）。

---

## 训练流程

### 第一阶段：SFT（监督微调）

```bash
python scripts/sft_train.py --config config/sft_config.yaml
```

关键超参数（见 `config/sft_config.yaml`）：

| 参数 | 值 |
|---|---|
| LoRA r | 16 |
| LoRA alpha | 32 |
| Learning rate | 2e-4 |
| Batch size | 2 × 8 (grad accum) |
| Max steps | 1000 |
| Optimizer | paged_adamw_8bit |
| Quantization | 4-bit NF4 |

训练产物保存至 `outputs/sft/`。

### 第二阶段：DPO（直接偏好优化）

```bash
python scripts/dpo_train.py --config config/dpo_config.yaml
```

关键超参数（见 `config/dpo_config.yaml`）：

| 参数 | 值 |
|---|---|
| β (beta) | 0.1 |
| Learning rate | 1e-5 |
| Max steps | 800 |
| 基座 | outputs/sft（SFT 产物） |

训练产物保存至 `outputs/dpo/`。

### 合并 LoRA 权重

将 LoRA adapter 合并回基础模型，便于部署：

```bash
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
    --max_samples 1319 \
    --output      eval/gsm8k_result.json
```

### BBH

```bash
python eval/bbh_eval.py \
    --model_path outputs/merged \
    --subset     boolean_expressions \
    --max_samples 250 \
    --output     eval/bbh_result.json
```

### 可视化

生成**雷达图**（多基准横向对比）：

```bash
python eval/visualize.py \
    --metrics_json eval/metrics_summary.json \
    --out_dir      eval/figures
```

`metrics_summary.json` 示例：
```json
{
  "GSM8K": 0.72,
  "BBH-Boolean": 0.65,
  "BBH-Causal": 0.58,
  "BBH-Date": 0.61
}
```

生成**Loss 曲线**：

```bash
python eval/visualize.py \
    --loss_csv eval/train_log.csv \
    --out_dir  eval/figures
```

`train_log.csv` 需包含 `step` 和 `loss` 两列。

---

## 快速推理测试

打开 `notebooks/inference_test.ipynb`，可直接对比同一道题在基础模型与微调模型上的输出差异。

---

## 实验记录（预期目标）

| 模型阶段 | GSM8K Acc | BBH Avg |
|---|---|---|
| Qwen2.5-1.5B-Instruct（基线） | ~45% | ~40% |
| + SFT | ~62% | ~50% |
| + DPO | ~66% | ~53% |

---

## Google Colab 使用指南

**不需要提前把数据集下载到本地。** Colab 可直接通过 `datasets` 库从 HuggingFace Hub 在线拉取，无需手动下载。

### 推荐流程

**1. 安装依赖（每次新 Session 需要重跑）**

```python
# Colab 第一个 Cell
!pip install unsloth trl peft bitsandbytes transformers datasets accelerate pyyaml -q
```

**2. 设置 HuggingFace Token（访问私有模型或数据集时需要）**

```python
from huggingface_hub import login
login(token="hf_YOUR_TOKEN")  # 建议用 Colab Secrets 管理，不要硬编码
```

或在 Colab 左侧「🔑 Secrets」面板中添加 `HF_TOKEN`，然后：

```python
import os
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

**3. 挂载 Google Drive（持久化保存模型检查点）**

Colab Session 结束后 `/content` 下的文件会丢失，建议将输出目录指向 Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
# 然后在 config yaml 中把 output_dir 改为：
# output_dir: "/content/drive/MyDrive/Qwen-Reasoning/outputs/sft"
```

**4. 克隆本项目并运行**

```python
!git clone https://github.com/your-repo/Qwen-Reasoning-Enhance.git
%cd Qwen-Reasoning-Enhance
!python scripts/sft_train.py --config config/sft_config.yaml
```

> **Colab 免费版** 仅提供 T4（15GB），Qwen2.5-1.5B-Instruct 的 4-bit 量化训练约占用 6-8GB，可以正常运行。
> **Colab Pro/Pro+** 可使用 A100，速度提升约 5-8×。

---

## 注意事项

- `unsloth` 目前仅支持 Linux / WSL2 + CUDA，macOS 仅用于代码开发，训练需在 GPU 机器上运行。
- 4-bit 量化训练时 `bf16=true` 与 `fp16=true` 不可同时开启；T4 不支持 bf16，在 Colab T4 上需将配置改为 `fp16: true` / `bf16: false`。
- DPO 训练时 `ref_model=None` 表示使用 SFT 模型本身作为参考模型（Unsloth 内置 PEFT 参考实现）。
- **请勿将 HuggingFace Token 明文写入代码或 README**，使用 Colab Secrets 或环境变量管理。

---

## 参考资料

- [Unsloth 官方文档](https://github.com/unslothai/unsloth)
- [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [GSM8K Dataset](https://huggingface.co/datasets/gsm8k)
- [BIG-Bench Hard (BBH)](https://huggingface.co/datasets/lukaemon/bbh)
- [Qwen2.5-1.5B-Instruct on HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
