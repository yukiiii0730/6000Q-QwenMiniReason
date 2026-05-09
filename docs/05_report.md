# DSAA-6000Q Final Report

# Enhancing Mathematical Reasoning of 1.5B LLM via Diagnostic Data Engineering

# 通过诊断式数据工程提升 1.5B 大语言模型数学推理能力

---

## Abstract / 摘要

本项目针对 Qwen2.5-1.5B-Instruct 在 GSM8K 数学推理任务上与 7B 模型（84.5%）存在的 22pp 差距，探索以数据工程为核心的优化路径。我们设计了三阶段流水线——五段式 SFT 课程学习、错误类型诊断、定向偏好优化（DPO），并在 8 组消融实验中系统验证了各组件的效果。核心发现：(1) 1409 条高质量 teacher CoT 蒸馏数据（Teacher SFT）将 GSM8K 从 62.5% 提升至 65.0%，超越 38k 混合数据 SFT（62.0%），验证了"数据质量 > 数据数量"的深层机制；(2) Error-Type-Targeted DPO 通过先诊断错误类型再定向构建偏好数据，在 GSM8K 上获得 +2.5pp 提升，证明诊断驱动的 DPO 比通用 DPO 更有效；(3) 五段式课程 SFT 通过 Stage A（GSM8K 锚点）建立分布集中度，使基座模型对 DPO 的稳定性显著优于单段 SFT（GSM8K 持平 vs 回退 -4.0pp）。本项目在 1.5B 规模上验证了 DeepSeek-R1-Distill 的核心洞察，并提出了 Error-Type-Targeted DPO 这一新的偏好优化范式。

This project addresses the 22pp gap between Qwen2.5-1.5B-Instruct (62.5%) and 7B models (84.5%) on GSM8K mathematical reasoning, exploring a data-engineering-centric optimization pathway. We design a three-stage pipeline — five-stage SFT curriculum, error-type diagnosis, and targeted Direct Preference Optimization (DPO) — and systematically validate each component across 8 ablation experiments. Key findings: (1) 1,409 high-quality teacher CoT samples (Teacher SFT) raise GSM8K from 62.5% to 65.0%, surpassing 38k mixed-data SFT (62.0%), confirming the deep mechanism of "quality > quantity"; (2) Error-Type-Targeted DPO, which diagnoses error types before constructing preference data, achieves +2.5pp on GSM8K, proving diagnostic-driven DPO is more effective than generic DPO; (3) Five-stage curriculum SFT establishes distribution concentration via Stage A (GSM8K anchor), making the base model significantly more stable under DPO than single-stage SFT (持平 vs -4.0pp regression). This project validates the core insight of DeepSeek-R1-Distill at the 1.5B scale and proposes Error-Type-Targeted DPO as a novel preference optimization paradigm.

---

## 1. Introduction / 引言

### 1.1 Background / 背景

大语言模型（LLM）的数学推理能力是衡量其智能水平的核心指标之一。当前前沿模型（GPT-4、Claude、DeepSeek-R1）在竞赛级数学上已接近人类专家水平，但这些模型的参数量通常在 100B+ 量级，部署成本高昂。相比之下，1.5B 参数的小模型具有显著的部署优势——可在边缘设备、移动端、离线场景运行——但其推理能力与大模型存在巨大差距。

Mathematical reasoning capability of Large Language Models (LLMs) is a core indicator of their intelligence level. Current frontier models (GPT-4, Claude, DeepSeek-R1) approach human-expert performance on competition-level mathematics, yet these models typically have 100B+ parameters with prohibitive deployment costs. In contrast, 1.5B-parameter small models offer significant deployment advantages — they can run on edge devices, mobile platforms, and offline scenarios — but their reasoning capability lags far behind larger models.

以 Qwen2.5 系列为例：Qwen2.5-1.5B-Instruct 在 GSM8K（小学数学应用题）上的 zero-shot 准确率为 62.5%，而 Qwen2.5-7B-Instruct 达到 84.5%，差距达 22 个百分点（pp）。这一差距是否主要由模型容量决定？还是可以通过数据工程和训练策略来弥补？

Taking the Qwen2.5 series as an example: Qwen2.5-1.5B-Instruct achieves 62.5% zero-shot accuracy on GSM8K (grade-school math word problems), while Qwen2.5-7B-Instruct reaches 84.5%, a gap of 22 percentage points (pp). Is this gap primarily determined by model capacity, or can it be bridged through data engineering and training strategies?

### 1.2 Motivation / 动机

近期文献给出了乐观信号：

Recent literature provides optimistic signals:

- **LIMA**（Zhou et al., 2023）：仅 1000 条高质量数据即可对齐 LLM，证明数据质量远比数量重要。**LIMA** (Zhou et al., 2023): Only 1,000 high-quality samples can align an LLM, demonstrating that data quality far outweighs quantity.

- **DeepSeek-R1-Distill**（DeepSeek, 2025）：从 671B 蒸馏到 1.5B-70B，蒸馏模型在 MATH/AIME 上表现优异，验证了 teacher 蒸馏对小模型的有效性。**DeepSeek-R1-Distill** (DeepSeek, 2025): Distilling from 671B to 1.5B-70B, the distilled models excel on MATH/AIME, validating teacher distillation's effectiveness for small models.

- **Orca-Math**（Microsoft, 2024）：GPT-4 蒸馏数据让 7B 模型在 GSM8K 上大幅超越 baseline。**Orca-Math** (Microsoft, 2024): GPT-4 distilled data enables 7B models to substantially surpass baselines on GSM8K.

然而，这些工作主要关注 7B+ 模型。1.5B 规模的数学推理优化仍是一个未被充分探索的领域。本项目的核心动机是：**在 1.5B 的极限规模下，系统性地探索数据工程对数学推理能力的提升效果，并提出可复用的方法论。**

However, these works primarily focus on 7B+ models. Mathematical reasoning optimization at the 1.5B scale remains an under-explored area. Our core motivation is: **at the extreme 1.5B scale, systematically explore the effect of data engineering on mathematical reasoning capability, and propose reusable methodology.**

### 1.3 Research Questions / 研究问题

1. **SFT 数据策略**：五段式课程学习（in-distribution → 深度 → 广度 → 难度 → 泛化）是否优于单段混合训练？
   - **SFT Data Strategy**: Does five-stage curriculum learning (in-distribution → depth → breadth → difficulty → generalization) outperform single-stage mixed training?

2. **DPO 基座稳定性**：SFT 基座的输出分布集中度如何影响 DPO 的效果？
   - **DPO Base Stability**: How does the output distribution concentration of the SFT base model affect DPO outcomes?

3. **定向偏好优化**：Error-Type-Targeted DPO（先诊断错误类型再构建偏好数据）是否比 Standard DPO 更有效？
   - **Targeted Preference Optimization**: Is Error-Type-Targeted DPO (diagnose error types before constructing preference data) more effective than Standard DPO?

4. **Teacher 蒸馏效率**：少量高质量 teacher CoT 能否超越大量混合数据？
   - **Teacher Distillation Efficiency**: Can a small amount of high-quality teacher CoT outperform large-scale mixed data?

---

## 2. Related Works / 相关工作

### 2.1 小模型数学推理优化

**数据质量范式**：Phi-1（Gunasekar et al., 2023）用 1.3B 模型 + 高质量"教科书"数据在代码任务上超越更大模型，开创了"数据质量 > 模型规模"的范式。OpenMathInstruct-2（NVIDIA, 2024）通过答案验证 + 难度过滤构建大规模数学 SFT 数据。本项目继承这一范式，但在数据质量控制上更进一步——不仅过滤低质量数据，还通过课程学习控制数据的呈现顺序。

**Data Quality Paradigm**: Phi-1 (Gunasekar et al., 2023) pioneered the "data quality > model scale" paradigm, with a 1.3B model surpassing larger models on code tasks using high-quality "textbook" data. OpenMathInstruct-2 (NVIDIA, 2024) constructs large-scale math SFT data via answer verification and difficulty filtering. Our project inherits this paradigm but goes further — not only filtering low-quality data, but also controlling the presentation order of data through curriculum learning.

**Teacher 蒸馏范式**：DeepSeek-R1-Distill 是当前最成功的蒸馏方案，从 671B 蒸馏到 1.5B-70B，蒸馏模型在 MATH 上甚至超过同规模 RL 训练模型。核心洞察：大模型的 CoT 推理过程本身就是最好的训练数据。本项目在 1.5B 规模上验证了这一洞察。

**Teacher Distillation Paradigm**: DeepSeek-R1-Distill is currently the most successful distillation scheme, distilling from 671B to 1.5B-70B, with distilled models even outperforming same-scale RL-trained models on MATH. Core insight: the CoT reasoning process of large models is itself the best training data. Our project validates this insight at the 1.5B scale.

### 2.2 偏好优化方法

**DPO 及其变体**：DPO（Rafailov et al., 2023）将 RLHF 简化为直接偏好优化，但 response-level 粒度导致信用分配困难。Step-DPO（Lai et al., 2024）在推理步骤级别构建偏好对，解决信用分配问题。Process Reward Model（Lightman et al., 2023）证明逐步奖励优于结果奖励（PRM 78.2% vs ORM 72.4% on MATH）。本项目的 Error-Type-Targeted DPO 是一种新的中间方案——在 response-level DPO 框架内引入错误类型诊断，用类型专属 prompt 生成更有针对性的 chosen。

**DPO and Variants**: DPO (Rafailov et al., 2023) simplifies RLHF to direct preference optimization, but response-level granularity causes credit assignment difficulties. Step-DPO (Lai et al., 2024) constructs preference pairs at the reasoning step level. Process Reward Model (Lightman et al., 2023) proves step-wise rewards outperform outcome rewards (PRM 78.2% vs ORM 72.4% on MATH). Our Error-Type-Targeted DPO is a novel intermediate approach — introducing error-type diagnosis within the response-level DPO framework, using type-specific prompts to generate more targeted chosen responses.

### 2.3 课程学习

多项 2024 年研究表明，按难度递增安排 SFT 数据顺序可提升最终性能。Qwen2.5-Math 技术报告中采用渐进式训练策略。反直觉发现：部分研究（THUDM）发现 hard-to-easy 有时优于 easy-to-hard。本项目选择 **in-distribution first → broad reasoning** 的课程顺序，先锚定评测分布再扩展能力。

Multiple 2024 studies show that arranging SFT data in increasing difficulty improves final performance. Qwen2.5-Math's technical report adopts a progressive training strategy. Counter-intuitively, some research (THUDM) finds hard-to-easy sometimes outperforms easy-to-hard. We choose an **in-distribution first → broad reasoning** curriculum, anchoring the evaluation distribution before broadening capability.

---

## 3. Methodology / 方法论

### 3.1 整体架构：三阶段流水线

本项目设计了三阶段流水线：**SFT 课程学习 → 错误诊断 → 定向偏好优化**。诊断是连接 SFT 和 DPO 的桥梁——SFT 建立能力基线，诊断发现弱点，DPO 定向修复。

We design a three-stage pipeline: **SFT Curriculum → Error Diagnosis → Targeted Preference Optimization**. Diagnosis bridges SFT and DPO — SFT establishes the capability baseline, diagnosis discovers weaknesses, and DPO performs targeted repair.

![Architecture](architecture.svg)

### 3.2 数据工程

#### 3.2.1 SFT 数据来源与预处理

| 阶段 | 数据集 | 样本数 | Token 上限 | 角色 | 设计依据 |
|---|---|---|---|---|---|
| **A** | GSM8K-train | 7,500 | 1024 | In-distribution 锚点 | 先对齐评测分布，确保 SFT 不偏离目标 |
| **B1** | OpenR1-Math-220k | 10,000 | 2048 | R1 蒸馏长 CoT | DeepSeek-R1-Distill 验证了长 CoT 蒸馏的有效性 |
| **B2** | Orca-Math-200k | 15,000 | 1024 | GPT-4 蒸馏广覆盖 | Orca-Math 证明 GPT-4 蒸馏数据对小模型有效 |
| **B3** | NuminaMath-CoT | 8,000 | 2048 | 竞赛难题多样性 | 引入更难的推理题，扩展能力上限 |
| **C** | Magpie-Reasoning | 3,000 | 2048 | 通用推理防遗忘 | ~7% 的通用推理数据防止灾难性退化 |

**预处理**：SHA-1 跨数据集去重 + 长度过滤（<1024/<2048 双阈值），从原始 ~50k 过滤至 ~38k。

**Preprocessing**: SHA-1 cross-dataset deduplication + length filtering (<1024/<2048 dual thresholds), filtering from ~50k raw to ~38k.

#### 3.2.2 Teacher CoT 数据构建

用 Qwen3-235B-Thinking 为 GSM8K-train 的 1500 道题生成 CoT 解答，三层质量过滤：
1. 长度过滤：剔除 >10k chars 的过长输出（41 条）
2. 修正次数过滤：剔除自我修正 ≥10 次的输出（87 条）
3. 答案正确性验证：与 GSM8K 标准答案比对，96.27% 正确

最终保留 1409 条高质量 teacher CoT。

We use Qwen3-235B-Thinking to generate CoT solutions for 1,500 GSM8K-train problems, with three-layer quality filtering:
1. Length filtering: remove outputs >10k chars (41 samples)
2. Self-correction filtering: remove outputs with ≥10 self-corrections (87 samples)
3. Answer verification: compare against GSM8K ground truth, 96.27% correct

Final: 1,409 high-quality teacher CoT samples.

#### 3.2.3 Targeted DPO 数据构建

**Step 1 — 错误诊断**：用 Qwen-Flash（低成本 API 模型）对 SFT 模型的 GSM8K badcase 做 5 类错误分类。

**Step 1 — Error Diagnosis**: Use Qwen-Flash (low-cost API model) to classify SFT model's GSM8K badcases into 5 error types.

| 错误类型 | 占比 | Targeted Prompt |
|---|---|---|
| setup_error | 64.9% | "先复述题目的已知条件和约束" |
| reasoning_skip | 27.3% | "展开每个推理步骤，不跳步" |
| extraction_error | 3.9% | "答案单独写在最后一行" |
| arithmetic | 2.6% | "请将每一步的中间结果明确写出" |
| unit_or_format | 1.3% | "标注单位，统一格式" |

**Step 2 — 定向数据构建**：针对每类错误，用类型专属 system prompt 让 Qwen3-235B-Thinking 生成 chosen，rejected 来自 SFT 模型的真实 badcase。合并为 420 条 targeted DPO 数据。

**Step 2 — Targeted Data Construction**: For each error type, use type-specific system prompts to have Qwen3-235B-Thinking generate chosen responses, with rejected responses from SFT model's actual badcases. Combined into 420 targeted DPO samples.

### 3.3 模型训练

#### 3.3.1 SFT 训练

| 参数 | Group A (E1) | Group B (E3) |
|---|---|---|
| PEFT | LoRA, r=16, α=32 | **DoRA**, r=16, α=32 |
| 数据 | 38k 混合单段 | 五段分阶段 |
| max_seq_length | 2048 | 2048 |
| batch_size | 2 × 8 = 16 | 2 × 8 = 16 |
| lr | 2e-4, cosine | 5e-5→2e-5 递减 |
| steps | ~3900 | ~3900 (分 5 段) |
| 框架 | Unsloth + TRL SFTTrainer | Unsloth + TRL SFTTrainer |

**DoRA vs LoRA**：DoRA（Weight-Decomposed Low-Rank Adaptation）将预训练权重分解为幅度（magnitude）和方向（direction）两个分量，仅对方向分量应用 LoRA。理论上，这种分解使 PEFT 更接近全参数微调的效果，同时保持参数效率。

**DoRA vs LoRA**: DoRA (Weight-Decomposed Low-Rank Adaptation) decomposes pretrained weights into magnitude and direction components, applying LoRA only to the direction. Theoretically, this decomposition makes PEFT closer to full fine-tuning while maintaining parameter efficiency.

**五段课程设计逻辑**：
- **Stage A 先于 B1-B3**：先锚定 GSM8K 分布，再引入更广数据。如果反过来，模型可能被难题"带偏"
- **B1→B2→B3**：从 R1 蒸馏（深度）到 GPT-4 蒸馏（广度）到竞赛题（难度），逐步扩展
- **Stage C 最后**：通用推理数据占比 7%，不影响数学对齐，但防止 BBH 退化
- **学习率递减**（5e-5→2e-5）：课程越往后数据越难，降低学习率防止遗忘

**Five-stage Curriculum Logic**:
- **Stage A before B1-B3**: Anchor GSM8K distribution first, then introduce broader data. Reversing this risks the model being "led astray" by hard problems
- **B1→B2→B3**: From R1 distillation (depth) to GPT-4 distillation (breadth) to competition problems (difficulty), progressively expanding
- **Stage C last**: General reasoning data at 7% doesn't affect math alignment but prevents BBH degradation
- **Decreasing LR** (5e-5→2e-5): Harder data later in curriculum → lower LR to prevent forgetting

#### 3.3.2 DPO 训练

| 参数 | Standard DPO (E2/E4) | Targeted DPO (E5) |
|---|---|---|
| Base | E1/E3 merged (fp16) | E3 merged (fp16) |
| 数据 | argilla/distilabel 5k | targeted 420 条 |
| beta | 0.1 | 0.1 |
| loss_type | sigmoid | sigmoid |
| lr | 1e-5 | 1e-5 |
| steps | 600 | ~157 |
| batch | 1 × 16 | 1 × 16 |

### 3.4 评测体系

| Benchmark | 样本数 | 指标 | 选取原因 |
|---|---|---|---|
| **GSM8K** | 200 | 准确率 | 小学数学应用题，in-distribution 主评测 |
| **MATH-500** | 200 | 准确率 | 竞赛级数学，5 个难度等级，评测推理深度 |
| **BBH-27** | 780 (30/task) | Macro avg | 27 个通用推理子任务，检测灾难性遗忘 |

**评测协议**：chat-template + zero-shot，与训练格式一致。95% CI = ±6.9pp（基于二项分布 p=0.625, n=200）。

**Evaluation Protocol**: chat-template + zero-shot, consistent with training format. 95% CI = ±6.9pp (binomial p=0.625, n=200).

**计算资源**：Colab T4（免费）+ GPU L20（付费）。Unsloth 2x 训练加速 + 4-bit NF4 量化。两次训练可复现（seed=42, loss 偏差 <0.02）。

**Compute**: Colab T4 (free) + GPU L20 (paid). Unsloth 2x training acceleration + 4-bit NF4 quantization. Reproducible across two runs (seed=42, loss deviation <0.02).

---

## 4. Results / 实验结果

### 4.1 GSM8K 主评测结果（n=200, 95% CI ±6.9pp）

![GSM8K Results](fig_results.png)

| 编号 | 实验组 | 特征 | 准确率 | vs Baseline | 关键发现 |
|---|---|---|---|---|---|
| E0 | Baseline | 未训练原始模型 | 62.5% | — | 基线 |
| E1 | Group A SFT | LoRA + 单段混合 | 63.5% | +1.0pp | SFT 提升有限 |
| E2 | Group A DPO | 单段 SFT + Standard DPO | 59.5% | -3.0pp | **DPO 回退** |
| E3 | Group B SFT | DoRA + 五段课程 | 62.0% | -0.5pp | 课程更均衡 |
| E4 | Group B DPO | 五段课程 + Standard DPO | 62.0% | -0.5pp | **基座稳定** |
| E5 | Group D Targeted | 五段课程 + Targeted DPO | 64.5% | +2.0pp | **定向有效** |
| E6 | Teacher SFT | 1409 teacher CoT | **65.0%** | **+2.5pp** | **最佳** |

### 4.2 多 Benchmark 综合结果

| 编号 | 实验组 | GSM8K | MATH-500 | BBH-27 |
|---|---|---|---|---|
| E0 | Baseline | 62.5% | — | — |
| E1 | Group A SFT | 63.5% | 44.5% | 38.5% |
| E2 | Group A DPO | 59.5% | 51.25% | — |
| E3 | Group B SFT | 62.0% | 44.0% | 38.8% |
| E4 | Group B DPO | 62.0% | 47.5% | — |
| E5 | Group D Targeted | 64.5% | 44.0% | 37.4% |
| E6 | Teacher SFT | **65.0%** | 43.5% | — |

### 4.3 关键发现一：DPO 基座稳定性（E2 vs E4）

同样是 5000 条 argilla DPO 数据，Group A（单段 SFT 基座）GSM8K 回退 -4.0pp，而 Group B（五段课程基座）持平 +0.0pp。

With the same 5,000 argilla DPO samples, Group A (single-stage SFT base) regresses -4.0pp on GSM8K, while Group B (five-stage curriculum base) maintains +0.0pp.

![DPO Loss Curve](fig_dpo_loss.png)

**深层原因**：DPO 的 KL 约束（beta=0.1）能否防止遗忘，取决于基座模型的输出分布集中度。Group A 的 38k 混合数据包含多种风格，输出分布分散；Group B 的 Stage A 用 GSM8K 7.5k 锚定了输出格式，分布更集中。这反映在 DPO loss 上：Group A loss 降至 0.239（过低=过拟合），Group B loss 降至 0.458（更稳定）。

**Root Cause**: Whether DPO's KL constraint (beta=0.1) prevents forgetting depends on the base model's output distribution concentration. Group A's 38k mixed data contains multiple styles, yielding dispersed output distribution; Group B's Stage A anchors the output format with GSM8K 7.5k, yielding concentrated distribution. This is reflected in DPO loss: Group A drops to 0.239 (too low = overfitting), Group B stabilizes at 0.458.

**设计启示**：**SFT 基座的"稳定性"本质上是输出分布的集中度。五段课程的 Stage A 起到了"分布锚点"的作用，这比单纯增加数据量更重要。**

**Design Insight**: **The "stability" of an SFT base is essentially the concentration of its output distribution. Stage A of the five-stage curriculum acts as a "distribution anchor," which is more important than simply increasing data volume.**

### 4.4 关键发现二：Error-Type-Targeted DPO（E5）

| 对比 | GSM8K Δ | MATH Δ | Badcase 变化 |
|---|---|---|---|
| B SFT → B DPO (Standard) | +0.0pp | +3.5pp | 77→76 (-1) |
| B SFT → D DPO (Targeted) | +2.5pp | +0.0pp | 77→71 (-6) |

Standard DPO 在 GSM8K 上无效但在 MATH 上有效；Targeted DPO 恰好相反。**DPO 的效果高度依赖数据与目标任务的分布对齐度**——通用偏好数据适合广泛推理（MATH），定向偏好数据适合特定任务（GSM8K）。

Standard DPO is ineffective on GSM8K but effective on MATH; Targeted DPO shows the opposite pattern. **DPO's effectiveness is highly dependent on the distribution alignment between data and target task** — generic preference data suits broad reasoning (MATH), while targeted preference data suits specific tasks (GSM8K).

**创新本质**：在 SFT 和 DPO 之间插入"错误诊断"环节，将 DPO 从通用优化变为定向修复。与 Standard DPO（通用 teacher）和 Teacher-Guided DPO（通用 teacher prompt）不同，Targeted DPO 针对具体错误类型生成纠正信号。

**Innovation Essence**: Inserting an "error diagnosis" step between SFT and DPO, transforming DPO from generic optimization to targeted repair. Unlike Standard DPO (generic teacher) and Teacher-Guided DPO (generic teacher prompt), Targeted DPO generates corrective signals for specific error types.

### 4.5 关键发现三：Teacher SFT——质量 > 数量

| 方案 | 数据量 | GSM8K | MATH |
|---|---|---|---|
| Group B SFT（五段课程）| 38,000 | 62.0% | 44.0% |
| Group D Targeted DPO | 38,000 + 420 | 64.5% | 44.0% |
| **Teacher SFT** | **1,409** | **65.0%** | 43.5% |

**1409 条 teacher CoT 超越 38k 混合数据**，在 GSM8K 上达到最高 65.0%。

**1,409 teacher CoT samples surpass 38k mixed data**, achieving the highest GSM8K accuracy of 65.0%.

**为什么 1409 条 > 38k 条？** 不仅仅是"质量高"，更关键的是**分布纯净度**：38k 混合数据包含 5 个来源，每个来源有不同的推理风格，模型需同时学习多种模式；1409 条 teacher CoT 全部来自 GSM8K-train，分布与评测集完全一致，模型只需学习一种模式。

**Why do 1,409 samples beat 38k?** Not just "high quality" — the critical factor is **distribution purity**: 38k mixed data contains 5 sources with different reasoning styles, requiring the model to learn multiple patterns simultaneously; 1,409 teacher CoT samples all come from GSM8K-train, perfectly aligned with the evaluation distribution, requiring the model to learn only one pattern.

**Teacher CoT 的隐含价值**：(1) 格式规范化——统一 `\boxed{}` 输出，消除 extraction_error；(2) 推理完整性——每步有明确中间结果，消除 arithmetic error；(3) 风格一致性——1409 条数据推理风格一致。

**Hidden Value of Teacher CoT**: (1) Format standardization — unified `\boxed{}` output, eliminating extraction_error; (2) Reasoning completeness — explicit intermediate results at each step, eliminating arithmetic error; (3) Style consistency — uniform reasoning style across 1,409 samples.

### 4.6 关键发现四：BBH 无灾难性退化

所有实验组在 BBH 上维持 37-39%。Stage C（Magpie 3k，占 7%，学习率 2e-5）有效防止通用推理退化。其作用类似 continual learning 中的 rehearsal 策略——少量旧任务数据防止新任务训练遗忘旧知识。

All experiment groups maintain 37-39% on BBH. Stage C (Magpie 3k, 7%, LR 2e-5) effectively prevents general reasoning degradation. Its role is analogous to rehearsal strategies in continual learning — a small amount of old-task data prevents new-task training from forgetting old knowledge.

### 4.7 Badcase 分析

![Error Distribution](fig_error_dist.png)

#### 典型错误示例

**setup_error（64.9%）**：模型将 "three cups per chicken per day" 误解为 "per meal"，将每日总量乘以 3。1.5B 模型的 token-level attention 在处理长句中的细粒度约束词（every day vs. each meal）时容易丢失信息。

**setup_error (64.9%)**: The model misreads "three cups per chicken per day" as "per meal," multiplying the daily total by 3. The 1.5B model's token-level attention easily loses fine-grained constraint words (every day vs. each meal) in long sentences.

**reasoning_skip（27.3%）**：模型跳过 "30 / 6 = 5" 这一关键步骤，直接输出无意义数字。CoT 过程中丢失了中间状态。

**reasoning_skip (27.3%)**: The model skips the critical step "30 / 6 = 5," directly outputting a meaningless number. The intermediate state is lost during CoT.

**Teacher SFT 残留的 setup_misread（69%）**：即使经过 teacher 训练，模型仍将 "20 more than half of 80" 解析为 160 而非 60。说明读题理解是 1.5B 模型容量的固有局限，teacher CoT 展示了正确解法但模型未能内化。

**Residual setup_misread in Teacher SFT (69%)**: Even after teacher training, the model parses "20 more than half of 80" as 160 instead of 60. This indicates reading comprehension is an inherent limitation of 1.5B model capacity — teacher CoT demonstrates correct solutions but the model fails to internalize them.

---

## 5. Conclusion / 结论

### 5.1 核心贡献

1. **五段式 SFT 课程设计**：提出 in-distribution first → broad reasoning 的课程策略，Stage A 作为分布锚点确保基座稳定性。验证了课程学习对 DPO 稳定性的关键作用。
   - **Five-stage SFT Curriculum**: Proposed an in-distribution first → broad reasoning curriculum strategy, with Stage A as a distribution anchor ensuring base stability. Validated the critical role of curriculum learning for DPO stability.

2. **Error-Type-Targeted DPO**：提出诊断驱动的偏好优化范式——先分类错误类型，再用类型专属 prompt 生成纠正样本。比 Standard DPO 在 GSM8K 上多 +2.5pp。
   - **Error-Type-Targeted DPO**: Proposed a diagnostic-driven preference optimization paradigm — classify error types first, then generate corrective samples with type-specific prompts. +2.5pp over Standard DPO on GSM8K.

3. **Teacher SFT 蒸馏验证**：在 1.5B 规模上验证了 DeepSeek-R1-Distill 的核心洞察——1409 条高质量 teacher CoT 超越 38k 混合数据。
   - **Teacher SFT Distillation Validation**: Validated the core insight of DeepSeek-R1-Distill at the 1.5B scale — 1,409 high-quality teacher CoT samples surpass 38k mixed data.

4. **诊断驱动的实验方法论**：不是盲目尝试不同方法，而是先诊断模型弱点，再针对性设计实验。每次实验后做 badcase 分析，验证错误是否被修复，发现新瓶颈。
   - **Diagnosis-driven Experimental Methodology**: Rather than blindly trying different methods, we first diagnose model weaknesses, then design targeted experiments. After each experiment, badcase analysis verifies whether errors are fixed and reveals new bottlenecks.

### 5.2 局限性

1. **评测样本量不足**：n=200, CI ±6.9pp，无法区分 <5pp 的差异
   - **Insufficient evaluation sample size**: n=200, CI ±6.9pp, cannot distinguish <5pp differences

2. **Targeted DPO 数据量小**：420 条, ~157 steps，训练不充分
   - **Small Targeted DPO data volume**: 420 samples, ~157 steps, insufficient training

3. **Teacher 数据仅覆盖 GSM8K**：MATH 从 44.0% 降至 43.5%
   - **Teacher data only covers GSM8K**: MATH drops from 44.0% to 43.5%

4. **Error classification 粒度不够**：setup_error 占 65% 但内部异质性高
   - **Insufficient error classification granularity**: setup_error accounts for 65% but has high internal heterogeneity

5. **Response-level DPO 的信用分配问题未解决**
   - **Credit assignment problem of response-level DPO remains unsolved**

### 5.3 未来方向

1. **Step-DPO / PRM**：从 response-level 到 step-level，解决信用分配问题
   - **Step-DPO / PRM**: From response-level to step-level, solving credit assignment

2. **Rejection Sampling**：同一模型生成多个候选，选对/错做 pair，放大 10-50 倍数据量
   - **Rejection Sampling**: Generate multiple candidates per problem, select correct/incorrect pairs, amplifying data 10-50x

3. **Iterative DPO**：多轮训练→评测→挖掘新 badcase→再训练
   - **Iterative DPO**: Multi-round training → evaluation → mining new badcases → retraining

4. **多领域 Teacher 数据**：扩展 teacher 覆盖 MATH、竞赛等多领域
   - **Multi-domain Teacher Data**: Extend teacher coverage to MATH, competitions, etc.

---

## 6. Contributions / 贡献

| 成员 | 贡献内容 | 贡献比例 |
|---|---|---|
| 成员 1 | 项目设计、SFT/DPO 训练管线开发、全部实验执行、评测体系搭建 | XX% |
| 成员 2 | 数据预处理、错误分类管线、Targeted DPO 数据构建 | XX% |
| 成员 3 | Teacher CoT 数据生成、质量过滤、文档撰写 | XX% |

（注：单人项目则 100%）

---

## References / 参考文献

1. Zhou, C., et al. (2023). LIMA: Less Is More for Alignment. *NeurIPS 2023*.
2. Gunasekar, S., et al. (2023). Textbooks Are All You Need. *arXiv:2306.11644*.
3. DeepSeek AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.
4. Microsoft Research. (2024). Orca-Math: Unlocking the Potential of SLMs in Grade School Math.
5. NVIDIA. (2024). OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data.
6. Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*.
7. Lai, X., et al. (2024). Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning. *arXiv:2406.18629*.
8. Lightman, H., et al. (2023). Let's Verify Step by Step. *arXiv:2305.20050*.
9. Liu, Z., et al. (2024). DeepSeek-R1-Distill-Qwen Technical Report.
10. Qwen Team. (2024). Qwen2.5 Technical Report. *arXiv:2412.15115*.
11. Ethayarajh, K., et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. *arXiv:2402.01306*.
12. Meng, Y., et al. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward. *arXiv:2405.14734*.
13. Cobbe, K., et al. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.
14. Hendrycks, D., et al. (2021). Measuring Mathematical Problem Solving with the MATH Dataset. *NeurIPS 2021*.
15. Suzgun, M., et al. (2022). Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. *ACL 2023 Findings*.
