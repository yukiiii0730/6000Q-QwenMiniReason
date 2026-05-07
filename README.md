# Qwen-Reasoning-Enhance（v4）

> **Goal**: On **Qwen2.5-1.5B-Instruct**, approach the math reasoning capability of a 7B model through a **5-stage aligned data curriculum** and **diagnosis-driven preference optimization**.

> Iteration history: [`AI2AI.md`](AI2AI.md) · Full architecture & design: [`me2AI.md`](me2AI.md)

---

## Core Design（v4）

| Dimension | Approach |
|---|---|
| **Base Model** | Qwen2.5-1.5B-Instruct |
| **SFT Data** | v4 Five-stage curriculum（Trio + in-distribution anchor, ~38k） |
| **DPO Data** | Error-Type-Targeted（innovation core）+ Teacher-Guided + distilabel fallback |
| **PEFT** | DoRA（r=16, alpha=32）|
| **Evaluation** | GSM8K + MATH-500 + BBH-27（lm-evaluation-harness official protocol, in progress） |
| **Baseline** | Qwen official published values + self-run 50-question sanity check |

---

## Data Strategy（v4 Five-stage Curriculum）

| Stage | Dataset | Samples | Role |
|---|---|---|---|
| **A** in-distribution | `openai/gsm8k` train | 7.5k | Direct alignment to GSM8K eval distribution |
| **B1** R1 reasoning depth | `open-r1/OpenR1-Math-220k`（verified） | 10k | High-quality long CoT, DeepSeek-R1 distillation |
| **B2** word problem breadth | `microsoft/orca-math-word-problems-200k` | 15k | Short steps, wide coverage, 1.5B-friendly（GPT-4 distillation）|
| **B3** problem diversity | `AI-MO/NuminaMath-CoT`（excl. gsm8k subset） | 8k | Olympiad/AMC/AOPS multi-source expert |
| **C** general reasoning | `Magpie-Align/Magpie-Reasoning-150K` | 3k | Prevent BBH degradation（~7% of total）|
| **DPO fallback** | `argilla/distilabel-math-preference-dpo` | 5k | Fallback without teacher data |

After cross-dataset deduplication（SHA-1）+ length filtering（<1024/<2048 dual threshold）: **~38k samples**.

---

## Architecture

```
Local（CPU, run_local.sh）                GPU（Colab/Server, run_gpu.sh / colab_ablation.ipynb）
──────────────────────────              ──────────────────────────────────────────────────────
① Data download & preprocessing         ⑤ SFT 5-stage curriculum training
② 7B/14B API eval（sanity check）        ⑥ Merge + SFT evaluation
③ Data quality filtering（optional）     ⑦ Error diagnosis（5-class classification）
④ Teacher DPO data generation           ⑧ Targeted DPO training
                                         ⑨ Merge + Final evaluation（lm-eval harness）
```

---

## Quick Start

### Step 1 — Environment Setup

```bash
cp .env.example .env
# Fill in: DASHSCOPE_API_KEY=sk-xxx  /  HF_TOKEN=hf_xxx

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-macos.txt   # Local stage（no GPU needed）
```

### Step 2 — Local Stage（no GPU）

```bash
bash run_local.sh                  # Full local pipeline（~1-2h）
bash run_local.sh --quick          # Quick test（500 samples each, ~15min）
bash run_local.sh --skip-filter    # Skip LLM quality filtering（recommended for v4）
```

### Step 3 — Sync to GPU Environment

```bash
# Server
rsync -avz --progress data/ user@server:/path/to/project/data/
rsync -avz --progress logs/ user@server:/path/to/project/logs/

# Colab（pack and upload to Google Drive）
zip -r local_artifacts.zip data/ logs/ config/
```

### Step 4 — GPU Stage（Colab A100 / Server）

```bash
# Server
bash run_gpu.sh                         # Full: SFT → DPO → eval
bash run_gpu.sh --skip-eval             # Train+merge only, eval locally
bash run_eval_expanded.sh               # Local expanded eval（n=200, BBH 30/task）★ recommended
bash run_eval_expanded.sh --group-c     # Also eval Colab Group C model
bash run_local_pipeline.sh              # Local L3-L6: error classify→targeted data→stats→viz

# Colab Ablation（open notebooks/colab_ablation.ipynb）
# G1-G3: Group A（LoRA + single-stage SFT + Standard DPO）
# G4-G5: Group D（Error-Type-Targeted DPO, innovation core）
# G6-G7: Group E/F（IPO + Weighted, optional）
```

### Step 5 — Generate Comparison Table

```bash
python3 eval/compare_table.py
python3 eval/visualize.py --metrics_json logs/compare_metrics.json --out_dir eval/figures
```

---

## Current Experimental Results（v4, as of 2026-05-06）

> ⚠️ **Note**: Results below use our custom eval protocol（chat-template + zero-shot）.  
> Absolute values differ from Qwen official numbers（which use lm-evaluation-harness 8-shot）.  
> **Re-evaluation with official protocol is in progress**（`lm_eval` directory）.  
> Relative Δ across groups is meaningful; absolute numbers will be updated.

### Ablation Results（custom protocol, n=200, CI ±6.9pp）

| Group | Configuration | GSM8K | MATH-500 | BBH-27 |
|---|---|---|---|---|
| **A SFT** | LoRA + Single-stage SFT | 63.5% | 44.5% | 38.5% |
| **A DPO** | + Standard DPO | 59.5% ⚠️ | 51.25% (n=80) | — |
| **B（SFT only）** | DoRA + 5-stage Curriculum | 62.0% | 44.0% | **38.8%** |
| **B** | DoRA + 5-stage Curriculum + Standard DPO | 62.0% | **47.5%** | TBD |
| **D**（innovation）| DoRA + 5-stage + **Error-Type-Targeted DPO** | **64.5%** | 44.0% | 37.4% |
| **Teacher SFT** | LoRA + 1409 teacher CoT（DeepSeek-R1-Distill style）| **65.0%** | 43.5% | — |
| Qwen2.5-1.5B（official）| — | 73.2% | 55.2% | — |
| Qwen2.5-7B（official）| — | 91.6% | 75.5% | — |

### Key Findings

- **Group A DPO regression**: GSM8K **-4.0pp** (63.5%→59.5%), MATH +6.75pp (n=80) — single-stage SFT base is unstable for DPO
- **Standard DPO（B SFT→B DPO）**: GSM8K +0.0pp, MATH **+3.5pp** — improves harder reasoning
- **Targeted DPO（B DPO→D DPO）**: GSM8K **+2.5pp**, MATH -3.5pp — improves targeted task, slight regression on harder MATH（within CI）
- **BBH**: No catastrophic degradation across any group（37–39%）
- **Teacher SFT（E12/E13）**: Only 1409 teacher CoT samples achieve **65.0% GSM8K** — new high, surpassing 38k mixed data A/B SFT and Targeted DPO. Validates DeepSeek-R1-Distill insight: quality > quantity
- **Two runs（Colab T4 + GPU L20）reproducible**: loss curves differ by <0.02 across all 5 stages（seed=42）

---

## Ablation Experiment（6 groups planned）

| Group | SFT | DPO | Purpose |
|---|---|---|---|
| A | LoRA + Single-stage mix | Standard DPO | Classic baseline |
| B | DoRA + 5-stage curriculum | Standard DPO | DoRA + Curriculum contribution |
| C | DoRA + 5-stage curriculum | Teacher-Guided | Teacher data effect |
| **D** | DoRA + 5-stage curriculum | **Error-Type-Targeted** | **Innovation core** |
| E | DoRA + 5-stage curriculum | IPO + Targeted data | Loss function improvement |
| F | DoRA + 5-stage curriculum | Weighted Targeted DPO | Weighted innovation variant |

Status: A(SFT✅ DPO✅) B✅ C（eval pending）D✅ E/F（optional）Teacher SFT✅ E14（eval pending）

---

## Error-Type-Targeted DPO Pipeline（Innovation）

```
SFT model → GSM8K evaluation → collect badcases
   ↓
scripts/classify_errors.py   （qwen-flash: 5-class）
   ↓
scripts/build_targeted_dpo.py（type-specific system prompt → Teacher generates chosen）
   ↓
Targeted DPO training（optional weighted loss）
   ↓
Re-evaluate → compare error repair rate by type
```

**5 Error Types**: arithmetic / reasoning\_skip / setup\_error / unit\_or\_format / extraction\_error

---

## Project Structure

```
.
├── run_local.sh              # Local stage one-click（no GPU）
├── run_gpu.sh                # GPU stage one-click
├── run_eval_expanded.sh      # Expanded eval（n=200, BBH 30/task）★
├── run_eval_local.sh         # Local eval only
├── run_local_pipeline.sh     # L3-L6: error analysis → targeted data → stats → viz
├── run_train.sh              # Compatibility entry
│
├── config/
│   ├── sft_config.yaml               # v4 5-stage curriculum + DoRA
│   ├── sft_config_group_a.yaml       # Group A（LoRA + single-stage）
│   ├── dpo_config.yaml               # DPO hyperparams
│   ├── dpo_config_group_a.yaml       # Group A DPO
│   ├── dpo_config_group_d.yaml       # Group D Targeted DPO
│   └── benchmark_models.yaml
│
├── scripts/
│   ├── sft_train.py          # SFT（DoRA + 5-stage curriculum）
│   ├── dpo_train.py          # DPO（loss_type switchable + weighted）
│   ├── merge_lora.py
│   ├── classify_errors.py    # Badcase classification（5 types）
│   ├── build_targeted_dpo.py # Targeted DPO data generation
│   ├── build_teacher_dpo.py  # Teacher-Guided DPO data generation
│   ├── prepare_data.py       # v4 data download + filter + dedup
│   ├── run_ablation.py       # 6-group ablation orchestrator
│   ├── stats_significance.py # McNemar + bootstrap CI
│   └── watchdog_run.py       # Process monitor（auto-restart）
│
├── eval/
│   ├── gsm8k_eval.py / gsm8k_api_eval.py
│   ├── math_eval.py  / math_api_eval.py
│   ├── bbh_eval.py   / bbh_full_eval.py
│   ├── compare_table.py      # Comparison table（official baseline priority）
│   ├── visualize.py          # Radar chart + error dist + ablation bar chart
│   └── published_baselines.json
│
├── notebooks/
│   ├── colab_train.ipynb     # Original Colab training entry
│   └── colab_ablation.ipynb  # Ablation study（G1-G7）★
│
├── data/processed/           # Training data（gitignore large files）
├── logs/                     # Eval results JSON + run logs
└── logs 2/                   # Ablation results（2026-05-04 Colab run）
```

---

## References

- [Unsloth](https://github.com/unslothai/unsloth) · [TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
- [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) · [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
- [Orca-Math-200k](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) · [Magpie-Reasoning-150K](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-150K)
- [GSM8K](https://huggingface.co/datasets/gsm8k) · [MATH](https://huggingface.co/datasets/hendrycks/competition_math) · [BBH](https://huggingface.co/datasets/lukaemon/bbh)
- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) · [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) · [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
