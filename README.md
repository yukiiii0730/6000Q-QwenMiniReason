# Qwen-Reasoning-Enhance（v4）

在 **Qwen2.5-1.5B-Instruct** 上通过**目标对齐的五段数据课程**与**诊断驱动的偏好优化**，实现接近 7B 模型的数学推理能力。

> 迭代历史见 [`AI2AI.md`](AI2AI.md)，完整架构与方案见 [`me2AI.md`](me2AI.md)。

---

## 核心方案（v4）

| 维度 | 方案 |
|---|---|
| **基座模型** | Qwen2.5-1.5B-Instruct |
| **SFT 数据** | v4 五段课程（三剑客 + in-distribution 锚点，共 ~38k） |
| **DPO 数据** | Error-Type-Targeted（创新核心）+ Teacher-Guided + distilabel 兜底 |
| **PEFT** | DoRA（r=16，alpha=32） |
| **评测** | GSM8K + MATH-500 + BBH-27 三维 |
| **Baseline** | Qwen 官方公开值 + 自跑 50 题 sanity check |

---

## 数据策略（v4 五段课程）

| 阶段 | 数据集 | 采样 | 作用 |
|---|---|---|---|
| **A** in-distribution | `openai/gsm8k` train | 7.5k | 直接对齐 GSM8K 评测分布 |
| **B1** R1 推理深度 | `open-r1/OpenR1-Math-220k`（verified） | 10k | 高质量长 CoT，DeepSeek-R1 蒸馏 |
| **B2** 应用题广度 | `microsoft/orca-math-word-problems-200k` | 15k | 步骤短、覆盖广，1.5B 友好，GPT-4 蒸馏 |
| **B3** 题型多样 | `AI-MO/NuminaMath-CoT`（去 gsm8k 子集） | 8k | 奥赛/AMC/AOPS 多源专家 |
| **C** 通用推理 | `Magpie-Align/Magpie-Reasoning-150K` | 3k | 防 BBH 退化（占比 ~7%） |
| **DPO 兜底** | `argilla/distilabel-math-preference-dpo` | 5k | 无 Teacher 数据时的 fallback |

跨集去重（SHA-1）+ 长度过滤（<1024/<2048 双档）后实际约 38k。

---

## 快速开始

### 架构分工

```
本地（CPU，run_local.sh）                GPU（Colab/服务器，run_gpu.sh）
──────────────────────────              ──────────────────────────────
① 数据下载 & 预处理                       ⑤ SFT 五段式课程训练
② 7B/14B API 评测（sanity check）        ⑥ 合并 + SFT 评测
③ 数据质量过滤（可选）                     ⑦ 错误诊断（5类分类）
④ Teacher DPO 数据生成                   ⑧ Targeted DPO 训练
                                         ⑨ 合并 + 最终评测
```

### Step 1 — 环境准备

```bash
cp .env.example .env
# 编辑 .env，填入：
#   DASHSCOPE_API_KEY=sk-xxx    （DashScope API，必须）
#   HF_TOKEN=hf_xxx             （HuggingFace Token，可选）

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-macos.txt   # 本地阶段（无需 torch/unsloth）
```

> **DashScope 注意**：请在 [DashScope 控制台](https://dashscope.console.aliyun.com/) 对用到的模型（`qwen2.5-7b-instruct`、`qwen2.5-14b-instruct`、`qwen-flash`）关闭「仅用免费额度」，否则会遇到 HTTP 403 错误。

### Step 2 — 本地阶段（无需 GPU）

```bash
bash run_local.sh                  # 完整本地流程（约 1-2 小时）
bash run_local.sh --quick          # 快速测试（数据各 500 条，约 15 分钟）
bash run_local.sh --force          # 忽略缓存，强制重新生成所有结果
bash run_local.sh --skip-filter    # 跳过 LLM 质量过滤（v4 数据源质量已高，推荐）
bash run_local.sh --skip-data      # 跳过数据下载（已有缓存时用）
bash run_local.sh --skip-api       # 跳过 API 评测（已有结果时用）
bash run_local.sh --eval-235b      # 额外跑 Qwen3-235B-Thinking 评测（费用较高）
bash run_local.sh --full-baseline  # 自跑完整 baseline（不用官方公开值）
```

完成后产出：

```
data/processed/
├── sft_train.json              # 五段课程 SFT 数据（~38k 条）
├── dpo_train.json              # DPO 兜底数据（5k 条）
└── dpo_teacher_round_1.json   # Teacher 生成的 DPO 数据（可选）

logs/
├── gsm8k_qwen25_7b_sanity.json    # 7B  sanity check 结果（50 题）
├── math_qwen25_7b_sanity.json
├── bbh_qwen25_7b_sanity.json
├── gsm8k_qwen25_14b_sanity.json   # 14B sanity check 结果（50 题）
├── math_qwen25_14b_sanity.json
└── bbh_qwen25_14b_sanity.json
```

### Step 3 — 同步到 GPU 环境

**方式 A：rsync 到服务器**

```bash
rsync -avz --progress data/   user@server:/path/to/project/data/
rsync -avz --progress logs/   user@server:/path/to/project/logs/
rsync -avz --progress config/ user@server:/path/to/project/config/
```

**方式 B：打包上传到 Google Drive（Colab 推荐）**

```bash
zip -r local_artifacts.zip data/ logs/ config/
# 上传到 Drive，Colab 挂载后 cp 到 /content/project/
```

### Step 4 — GPU 阶段（Colab A100 / 服务器）

**Colab 训练：** 打开 `notebooks/colab_train.ipynb`，按顺序执行各 Cell（Group B/C 训练）。

**Colab 消融实验：** 打开 `notebooks/colab_ablation.ipynb`，完整流水线（P0-P5）：
- P0: 环境 + Drive 同步（~5 min）
- P1: Group B 扩充评测 n=200（~1.5h）
- P2: 错误分类 + Targeted DPO 数据生成（~30 min，需 DASHSCOPE_API_KEY）
- P3: Group A 训练 + 评测（~4h）
- P4: Group D 训练 + 评测（~1.5h）
- P5: 汇总 + 同步

每个 Phase 支持断点续跑，完成后自动同步到 Google Drive。

**服务器 / Colab（仅训练、评测在本地做）：** 可用 `bash run_gpu.sh --skip-eval` 只跑 SFT / DPO / 合并，不上 GSM8K·MATH·BBH 本地评；将 `outputs/` 同步到本机后执行 `bash run_eval_local.sh`（与 `run_gpu` 默认评测子集一致）。先自检本机： `python3 scripts/check_eval_env.py`（需已同步 `outputs/sft_merged` 与 `outputs/merged`）。

**服务器：**

```bash
pip install -r requirements.txt

bash run_gpu.sh                         # 完整：SFT → DPO → 评测
bash run_gpu.sh --skip-eval             # 只训练+合并，评测在本地用 run_eval_local.sh
bash run_eval_local.sh                  # 本机：评 SFT 与最终合并模型（需已同步 outputs/）
bash run_gpu.sh --use-teacher-dpo       # 追加 Teacher 生成的 DPO 数据
bash run_gpu.sh --use-filtered-data     # 使用质量过滤后的数据（需先跑 filter）
bash run_gpu.sh --iterative-dpo 2       # 额外跑 2 轮 Iterative DPO
bash run_gpu.sh --quick                 # 快速测试（50/30 步）
bash run_gpu.sh --skip-sft              # 跳过 SFT（已有 outputs/sft_merged）
```

完成后产出：

```
outputs/
├── sft_merged/     # 合并后 SFT 模型
└── merged/         # 最终 SFT+DPO 模型

logs/
├── gsm8k_sft.json | math_sft.json | bbh_sft.json     # SFT 评测
└── gsm8k_result.json | math_result.json | bbh_result.json  # 最终评测
```

### Step 5 — 本地生成对比表

`run_eval_local.sh` 结束时若 `logs/*.json` 已齐，会调用 `compare_table.py`；仅补表时：

```bash
python3 eval/compare_table.py
python3 eval/visualize.py --metrics_json logs/compare_metrics.json --out_dir eval/figures
```

---

## 项目结构

```
.
├── run_local.sh              # 本地阶段一键脚本（无需 GPU）
├── run_gpu.sh                # GPU 阶段一键脚本
├── run_eval_local.sh         # 仅本机评测（可 --quick；--load-in-4bit 仅 NVIDIA+CUDA，Mac 用默认半精度）
├── run_train.sh              # 兼容入口（本地有 GPU 时使用）
│
├── config/
│   ├── sft_config.yaml       # v4 五段课程超参 + DoRA 配置
│   ├── dpo_config.yaml       # DPO 超参（loss_type / beta / seq_len）
│   └── benchmark_models.yaml # 参考模型 + Teacher 模型分工
│
├── scripts/
│   ├── prepare_data.py       # v4 数据下载：三剑客 + 长度过滤 + 去重
│   ├── sft_train.py          # SFT（DoRA + 五段课程）
│   ├── dpo_train.py          # DPO（loss_type 可切换 + 加权支持）
│   ├── merge_lora.py         # LoRA/DoRA 合并
│   ├── classify_errors.py    # badcase 分类（5 类错误）
│   ├── build_targeted_dpo.py # Error-Type-Targeted DPO 数据生成
│   ├── build_teacher_dpo.py  # Teacher-Guided DPO 数据生成
│   ├── llm_quality_filter.py # 数据质量评分（可选）
│   ├── iterative_dpo_loop.py # Iterative DPO 编排
│   ├── run_ablation.py       # 6 组 ablation 对照实验
│   ├── stats_significance.py # McNemar 检验 + bootstrap CI
│   ├── check_eval_env.py     # 本机跑 eval 前自检（CPU/MPS/CUDA、内存）
│   └── watchdog_run.py       # 进程监控（卡死自动重启）
│
├── eval/
│   ├── gsm8k_eval.py / gsm8k_api_eval.py   # GSM8K 本地 / API 评测
│   ├── math_eval.py  / math_api_eval.py    # MATH-500 本地 / API 评测
│   ├── bbh_eval.py   / bbh_api_eval.py     # BBH 本地 / API 评测
│   ├── bbh_full_eval.py                    # BBH 27 子任务 wrapper
│   ├── compare_table.py                    # 生成对比表（优先官方 baseline）
│   ├── visualize.py                        # 雷达图 + 错误分布 + ablation 图
│   └── published_baselines.json            # Qwen 官方公开值
│
├── data/processed/           # 本地生成的训练数据（gitignore 大文件）
├── logs/                     # 评测结果 JSON + 每次运行日志
└── notebooks/
    ├── colab_train.ipynb     # Colab GPU 训练入口（Group B/C）
    └── colab_ablation.ipynb  # Colab 完整消融实验（P0-P5：评测+错误分类+Group A/D 训练）
```

---

## 评测与 Baseline 策略

| 模型 | GSM8K | MATH-500 | BBH-27 | 来源 |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct（目标模型） | 73.2% | 55.2% | 自跑全量 | 官方 + GPU 阶段产出 |
| Qwen2.5-7B-Instruct | **91.6%** | **75.5%** | **70.3%** | Qwen 官方 + 50题 sanity check |
| Qwen2.5-14B-Instruct | **94.0%** | **80.0%** | **78.9%** | Qwen 官方 + 50题 sanity check |

> 引用：Qwen Team, "Qwen2.5 Technical Report", arXiv:2412.15115。1.5B 的 GSM8K/MATH 为官方值，BBH 无官方值需 GPU 阶段自跑。训练后结果将回填本表。

**v4 Group B（DoRA + 五段课程 + Standard DPO）已完成（2026-04-29）：**

| 模型 | GSM8K | MATH-500 | BBH-27 macro |
|---|---|---|---|
| Qwen2.5-1.5B-Instruct（Baseline）| 46% | ~55%（官方）| ~38%（估算）|
| **Ours SFT only** | **62%** | **50%** | **38.52%** |
| **Ours SFT + DPO（Group B）** | **66%** | **56%** | **41.11%** |
| Qwen2.5-7B-Instruct（参考）| 82% | 70% | — |

> 当前 n=50（CI ±14pp），扩充评测 n=200 和消融实验（Groups A/D）在 `colab_ablation.ipynb` 中执行。

---

## Ablation 实验（6 组对照）

| Group | SFT | DPO | 验证目的 |
|---|---|---|---|
| A | LoRA + 单段混合 | Standard DPO | 经典 baseline |
| B | DoRA + 五段课程 | Standard DPO | DoRA + 课程效果 |
| C | DoRA + 五段课程 | Teacher-Guided | Teacher 数据效果 |
| **D** | DoRA + 五段课程 | **Error-Type-Targeted** | **创新核心** |
| E | DoRA + 五段课程 | IPO + Targeted 数据 | 损失函数改进 |
| F | DoRA + 五段课程 | Weighted Targeted DPO | 加权版创新 |

```bash
python3 scripts/run_ablation.py --eval_n 200 --bbh_n 100   # 完整 6 组
python3 scripts/run_ablation.py --groups D                  # 只跑创新组
python3 scripts/run_ablation.py --quick                     # 快速 smoke test
```

---

## Error-Type-Targeted DPO 闭环（创新点 B）

```
SFT 模型评测 GSM8K → 自动收集 badcase
   ↓
scripts/classify_errors.py   （qwen-flash 归到 5 类）
   ↓
scripts/build_targeted_dpo.py（每类用专属 system prompt 让 Teacher 生成 chosen）
   ↓
Targeted DPO 训练（可选 weighted loss）
   ↓
再次评测 → 对比每类错误的修复率
```

```bash
# 1. 错误分类（约 ¥0.5 / 300 条）
python3 scripts/classify_errors.py \
    --badcase_jsonl logs/runs/<run_id>/results/gsm8k_sft_badcases.jsonl \
    --output_dir results/errors/sft

# 2. 生成 Targeted DPO 数据
python3 scripts/build_targeted_dpo.py \
    --by_type_dir results/errors/sft/by_type \
    --tag round1 --per_type_n 200
```

---

## 注意事项

- `unsloth` 仅支持 Linux + CUDA，**macOS 无法跑 SFT/DPO**；`run_local.sh` 不依赖 unsloth，可在 macOS 直接运行。
- **DashScope 免费额度**：`qwen2.5-7b-instruct`、`qwen2.5-14b-instruct`、`qwen-flash` 均有免费额度上限，建议提前在控制台关闭「仅用免费额度」。
- **数据质量过滤可跳过**：v4 数据源（OpenR1/NuminaMath/Orca-Math）本身质量已有保证，`--skip-filter` 对训练效果影响可忽略。
- **缓存机制**：`run_local.sh` 检测到结果文件已存在则跳过，重新生成用 `--force`，重跑特定步骤用对应 `--skip-*` 组合。
- 请勿提交 `.env` 或含 Token 的文件；`config/*.yaml` 中无明文密钥，统一通过 `.env` 或 Colab Secrets 管理。

---

## 参考资料

- [Unsloth](https://github.com/unslothai/unsloth) · [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) · [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [Orca-Math-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) · [Magpie-Reasoning-150K](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-150K)
- [GSM8K](https://huggingface.co/datasets/gsm8k) · [MATH](https://huggingface.co/datasets/hendrycks/competition_math) · [BBH](https://huggingface.co/datasets/lukaemon/bbh)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
