# Qwen-Reasoning-Enhance

基于 **Unsloth + TRL** 对 Qwen2.5-1.5B-Instruct 进行多阶段微调（SFT → DPO），提升模型在数学推理（GSM8K）及综合推理（BBH）任务上的能力。

---

## 实验结果（实测）

> 评测数据：GSM8K / BBH 各 50 条（固定 seed 的分层抽样）

| 模型阶段 | GSM8K Acc | BBH Acc |
|---|---|---|
| Qwen2.5-1.5B-Instruct（Baseline） | **32.0%** | **82.0%** |
| + SFT only | — | — |
| + **SFT + DPO**（完整两阶段） | **46.0%** | **84.0%** |
| Δ vs Baseline | **+14.0pp** | **+2.0pp** |

> 完整对比表由评测 notebook 自动生成，结果存入 `eval/` 目录。

---

## 优化路线图

按 TA 建议的优先级执行：

| Step | 内容 | 平台 | 脚本/Notebook |
|---|---|---|---|
| **1** | LoRA vs DoRA vs 全量微调对比 | Colab A100 | `colab_dora_experiment.ipynb` / `colab_full_finetune.ipynb` |
| **2** | DPO 数据质量清洗 + Teacher-Guided 构造 | Colab / 本地 | `dpo_quality_filter.py` / `teacher_guided_dpo.py` |
| **3** | SFT 分阶段训练（NuminaMath → Magpie） | HPC H20 | `staged_sft.py` |
| **4** | Long-CoT 数据合成 | 不需要 GPU | `synthesize_long_cot.py` |
| **5** | Iterative DPO（多轮迭代） | HPC H20 | `iterative_dpo.py` |

---

## 项目结构

```
Qwen-Reasoning-Enhance/
├── run_train.sh                  # 🚀 一键全流程脚本
├── config/
│   ├── sft_config.yaml           # SFT 超参数（LoRA/DoRA r=16, alpha=32 等）
│   ├── dpo_config.yaml           # DPO 超参数
│   └── benchmark_models.yaml     # 7B/14B 对照评测配置
├── scripts/
│   ├── sft_train.py              # SFT 训练（LoRA/DoRA，支持 use_dora 切换）
│   ├── dpo_train.py              # DPO 偏好优化
│   ├── full_finetune.py          # 全量微调（不用 LoRA，直接训练全部参数）
│   ├── staged_sft.py             # Step 3: SFT 分阶段训练
│   ├── iterative_dpo.py          # Step 5: Iterative DPO（推理→收集错误→DPO）
│   ├── dpo_quality_filter.py     # Step 2a: DPO 数据质量清洗（规则+API）
│   ├── teacher_guided_dpo.py     # Step 2b: Teacher-Guided DPO 数据构造
│   ├── synthesize_long_cot.py    # Step 4: Long-CoT 数据合成
│   ├── merge_lora.py             # LoRA/DoRA 权重合并
│   ├── prepare_data.py           # 数据集下载与预处理
│   ├── build_hq_subsets.py       # 高质量数据子集筛选
│   ├── backfill_badcases.py      # 回填 badcase
│   └── write_stage_report.py     # 阶段报告生成
├── eval/
│   ├── gsm8k_eval.py             # GSM8K 自动评测
│   ├── bbh_eval.py               # BBH 自动评测
│   ├── gsm8k_api_eval.py         # GSM8K API 评测
│   ├── bbh_api_eval.py           # BBH API 评测
│   ├── benchmark_open_models.py  # 批量评测开源模型
│   ├── compare_table.py          # 统一对比表
│   └── visualize.py              # 雷达图 & Loss 曲线
├── data/
│   ├── processed/                # 预处理后的训练数据
│   └── process.py                # 格式转换脚本
├── notebooks/
│   ├── colab_train.ipynb         # Colab 基线训练（LoRA SFT→DPO）
│   ├── colab_dora_experiment.ipynb  # Step 1: DoRA 对比实验
│   ├── colab_full_finetune.ipynb    # Step 1: 全量微调对比实验
│   └── inference_test.ipynb      # 推理对比
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

### Step 1：训练方法对比（Colab A100）

```bash
# LoRA 基线（colab_train.ipynb）
python scripts/sft_train.py --config config/sft_config.yaml
python scripts/dpo_train.py --config config/dpo_config.yaml

# DoRA 对比（colab_dora_experiment.ipynb）
# 在 config 中设置 lora.use_dora: true，其他不变

# 全量微调对比（colab_full_finetune.ipynb）
python scripts/full_finetune.py --config config/sft_config.yaml
```

### Step 2：数据质量清洗（Colab / 本地，不需要 GPU）

```bash
# 2a: 规则过滤（快速）
python scripts/dpo_quality_filter.py \
    --input data/processed/dpo_train.json \
    --output data/processed/dpo_filtered.json \
    --rule_only

# 2a: API 过滤（更精准）
python scripts/dpo_quality_filter.py \
    --input data/processed/dpo_train.json \
    --output data/processed/dpo_filtered.json \
    --api_base_url https://api.openai.com/v1 \
    --api_key sk-xxx --model gpt-4o-mini

# 2b: Teacher-Guided DPO 构造
python scripts/teacher_guided_dpo.py \
    --questions data/processed/sft_train.json \
    --student_model outputs/sft \
    --teacher_api_url https://api.openai.com/v1 \
    --teacher_api_key sk-xxx --teacher_model gpt-4o-mini \
    --output data/processed/dpo_teacher_guided.json
```

### Step 3：SFT 分阶段训练（HPC H20）

```bash
# 两阶段全跑
python scripts/staged_sft.py --config config/sft_config.yaml

# 只跑阶段 1（NuminaMath 基础逻辑对齐）
python scripts/staged_sft.py --config config/sft_config.yaml --stage 1

# 从阶段 2 开始（Magpie 通用推理增强）
python scripts/staged_sft.py --config config/sft_config.yaml --stage 2
```

### Step 4：Long-CoT 数据合成（不需要 GPU，纯 API）

```bash
python scripts/synthesize_long_cot.py \
    --input data/processed/sft_train.json \
    --output data/processed/sft_long_cot.json \
    --api_base_url https://api.openai.com/v1 \
    --api_key sk-xxx --model qwen2.5-72b-instruct \
    --max_samples 3000
```

### Step 5：Iterative DPO（HPC H20）

```bash
# 第 1 轮：用 SFT 模型推理 → 收集错误 → DPO
python scripts/iterative_dpo.py \
    --sft_model outputs/sft \
    --eval_data data/processed/sft_train.json \
    --output_dir outputs/iterative_dpo_r1 \
    --round 1

# 第 2 轮：用第 1 轮模型继续迭代
python scripts/iterative_dpo.py \
    --sft_model outputs/iterative_dpo_r1 \
    --eval_data data/processed/sft_train.json \
    --output_dir outputs/iterative_dpo_r2 \
    --round 2
```

### 合并 LoRA 权重

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

项目提供了多个 Colab Notebook，按需选用：

| Notebook | 用途 | GPU |
|---|---|---|
| `colab_train.ipynb` | LoRA 基线训练（SFT→DPO） | A100 / T4 |
| `colab_dora_experiment.ipynb` | DoRA vs LoRA 对比 | A100 |
| `colab_full_finetune.ipynb` | 全量微调对比 | A100 |

**使用流程**：打开 Notebook → 选择 A100 GPU → 按顺序运行 Cell。

> 断线重连后从 Cell 1（挂载 Drive）重新开始，数据缓存和 checkpoint 都在 Drive 上，不会丢失。

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
