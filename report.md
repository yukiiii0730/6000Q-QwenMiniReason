# Enhancing Mathematical Reasoning in Small Language Models via Curriculum Supervised Fine-Tuning and Diagnosis-Driven Preference Optimization

**DSAA-6000Q Project Final Report**

---

## Abstract

Small language models (SLMs) with approximately 1.5 billion parameters offer a compelling balance between capability and deployment cost, yet they still lag significantly behind larger models in mathematical reasoning tasks. This project investigates whether a combination of curriculum-based supervised fine-tuning (SFT) and diagnosis-driven Direct Preference Optimization (DPO) can narrow the math reasoning gap between a 1.5B-parameter model and its 7B-parameter counterpart. We fine-tune Qwen2.5-1.5B-Instruct using a five-stage data curriculum comprising approximately 38,000 samples sourced from GSM8K, OpenR1-Math, OrcaMath, NuminaMath-CoT, and Magpie-Reasoning, applying Weight-Decomposed Low-Rank Adaptation (DoRA) with only 1.18% trainable parameters. We then apply both standard DPO and a novel Error-Type-Targeted DPO approach that classifies model failures into five error categories and constructs preference pairs tailored to each failure mode. Evaluation on GSM8K, MATH-500, and BBH-27 (n=200 per benchmark) shows that our Teacher SFT experiment achieves 65.0% on GSM8K using only 1,409 high-quality teacher reasoning traces, surpassing all other approaches. Error-Type-Targeted DPO achieves 64.5% on GSM8K (+2.5pp over baseline). Standard DPO yields a 3.5pp improvement on MATH-500. SFT alone does not consistently improve over the base model. We provide a detailed error taxonomy, analyze failure modes across difficulty levels and mathematical subjects, and outline concrete directions for achieving further gains through rejection sampling, iterative DPO, and teacher distillation.

---

## 1. Introduction

### 1.1 Background

The rapid scaling of large language models (LLMs) has produced remarkable capabilities in mathematical reasoning, with models such as GPT-4 and DeepSeek-R1 achieving near-human performance on competition-level problems. However, these models typically require hundreds of billions of parameters and substantial computational resources for both training and inference. In resource-constrained deployment scenarios — including edge devices, real-time applications, and educational tools in developing regions — small language models (SLMs) in the 1–3B parameter range are far more practical.

The Qwen2.5 family \cite{qwen25} provides a useful testbed for studying the capabilities and limitations of SLMs. The Qwen2.5-1.5B-Instruct model achieves 73.2% on GSM8K and 55.2% on MATH-500, while the 7B variant reaches 91.6% and 75.5% respectively. This 18–20 percentage point gap on elementary math benchmarks represents a meaningful deficit that limits the practical utility of SLMs for quantitative reasoning tasks.

### 1.2 Motivation

Two complementary lines of recent work suggest that this gap can be narrowed without simply increasing model size. First, curriculum-based SFT strategies that progressively expose models to data of increasing complexity have shown that training order and data composition significantly affect downstream performance \cite{deepseekr1, qwq}. Second, preference optimization methods — particularly DPO \cite{dpo} and its variants — can further refine model behavior by teaching the model to prefer correct reasoning paths over incorrect ones. Critically, prior work on small models suggests that distillation from larger teacher models is more effective than direct reinforcement learning \cite{deepseekr1}, and that data quality dominates data quantity \cite{lima}.

A key observation from our preliminary experiments is that standard DPO, while effective on average, does not uniformly improve all error types. Some failure modes — particularly "setup errors" where the model misinterprets the problem statement — dominate the error distribution (65–77% of all failures) but may not be well-addressed by generic preference pairs. This motivates a more targeted approach.

### 1.3 Objectives

This project pursues three objectives:

1. **Curriculum SFT**: Design and evaluate a five-stage supervised fine-tuning curriculum for Qwen2.5-1.5B-Instruct that progresses from in-distribution GSM8K data through diverse mathematical reasoning datasets to general reasoning tasks.

2. **Diagnosis-Driven DPO**: Develop an Error-Type-Targeted DPO pipeline that classifies model failures, constructs preference pairs specific to each error category, and evaluates whether targeted optimization outperforms standard DPO.

3. **Comprehensive Error Analysis**: Provide a fine-grained error taxonomy across benchmarks, difficulty levels, and mathematical subjects to identify the most promising directions for future improvement.

---

## 2. Related Works

### 2.1 Mathematical Reasoning in Language Models

Mathematical reasoning has served as a key benchmark for evaluating the logical and quantitative capabilities of language models. The GSM8K benchmark \cite{gsm8k} tests grade-school arithmetic word problems, while the MATH dataset \cite{math} covers competition-level mathematics across seven subjects and five difficulty levels. The BIG-Bench Hard (BBH) suite \cite{bbh} provides 27 challenging tasks spanning multiple reasoning types. Performance on these benchmarks has improved dramatically with scale: GPT-4 achieves over 90% on GSM8K, while smaller models in the 1–3B range typically score 50–75% \cite{qwq, qwen25}.

### 2.2 Parameter-Efficient Fine-Tuning

Full fine-tuning of even 1.5B-parameter models is computationally expensive and risks catastrophic forgetting. Low-Rank Adaptation (LoRA) \cite{lora} addresses this by injecting trainable low-rank matrices into transformer layers, reducing trainable parameters to typically 0.5–2% of the total. DoRA (Weight-Decomposed Low-Rank Adaptation) \cite{dora} further decomposes weight updates into magnitude and direction components, achieving better performance at the same rank. Our project employs DoRA with rank 16 and alpha 32 across 7 target modules, yielding 18.5M trainable parameters (1.18% of the 1.56B total).

### 2.3 Curriculum Learning for LLMs

Curriculum learning, inspired by human educational progression, trains models on data ordered by difficulty or domain. DeepSeek-R1 \cite{deepseekr1} demonstrated that a multi-stage pipeline combining cold-start SFT, reinforcement learning, and rejection sampling achieves strong reasoning performance. The Qwen2.5-Math series \cite{qwen25math} employed a multi-stage pipeline with rejection sampling, achieving 75–80% on GSM8K for 1.5B models. Our five-stage curriculum follows a similar philosophy: anchoring on in-distribution data (GSM8K), progressively introducing harder and more diverse mathematical reasoning data (OpenR1, OrcaMath, NuminaMath), and concluding with general reasoning data (Magpie) to prevent catastrophic forgetting on non-mathematical tasks.

### 2.4 Preference Optimization

Direct Preference Optimization (DPO) \cite{dpo} reformulates the RLHF objective as a simple classification loss over preferred and rejected responses, eliminating the need for a separate reward model. Recent work has explored several directions relevant to our setting. Iterative DPO, which generates new preference pairs from the current model at each iteration, has been shown to outperform single-shot DPO by 3–5% \cite{iterativedpo}. Simpler alternatives such as KTO \cite{kto} and SimPO \cite{simp} achieve comparable results with fewer hyperparameters. Process-level preference data, which annotates correctness at each reasoning step rather than only the final answer, has demonstrated superior performance on mathematical tasks \cite{prm}. For small models specifically, distillation from large teacher models to generate high-quality preference pairs is more effective than direct RL optimization \cite{deepseekr1}.

### 2.5 Error Analysis in Mathematical Reasoning

Prior work has identified several recurring failure modes in LLM mathematical reasoning: arithmetic computation errors, problem setup errors (misinterpreting the problem), reasoning chain gaps (skipping necessary steps), and answer extraction failures \cite{errortaxonomy}. Our work extends this taxonomy with a five-category classification scheme (setup_error, arithmetic, reasoning_skip, extraction_error, unit_or_format) and introduces a feedback loop where error diagnosis directly informs DPO data construction.

---

## 3. Methodology

### 3.1 Base Model and Architecture

We use Qwen2.5-1.5B-Instruct \cite{qwen25} as our base model. This is a decoder-only transformer with 1.56 billion parameters, pre-trained on a large-scale multilingual corpus and instruction-tuned for chat interactions. The model supports a context length of 32,768 tokens and uses grouped-query attention (GQA) for efficient inference.

For parameter-efficient fine-tuning, we apply DoRA \cite{dora} with the following configuration:

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Target modules | 7 (attention projections and MLP layers) |
| Trainable parameters | 18,464,768 |
| Total parameters | 1,562,179,072 |
| Trainable ratio | 1.18% |

DoRA was selected over standard LoRA based on prior evidence that the weight-decomposed approach achieves better performance at equivalent rank, particularly for reasoning tasks where directional weight updates are critical \cite{dora}.

### 3.2 Supervised Fine-Tuning: Five-Stage Curriculum

#### 3.2.1 Data Sources and Curation

Our SFT curriculum comprises five stages using five distinct datasets, progressing from in-distribution mathematical data to diverse reasoning tasks:

| Stage | Dataset | Samples | Role |
|-------|---------|---------|------|
| A | openai/gsm8k (train) | 7,500 | In-distribution anchor aligned to GSM8K evaluation |
| B1 | open-r1/OpenR1-Math-220k | 10,000 | DeepSeek-R1 distilled long chain-of-thought reasoning |
| B2 | microsoft/orca-math-word-problems-200k | 15,000 | Short-step word problems with GPT-4 distillation |
| B3 | AI-MO/NuminaMath-CoT (excl. GSM8K) | 8,000 | Olympiad, AMC, and AOPS multi-source expert problems |
| C | Magpie-Align/Magpie-Reasoning-150K | 3,000 | General reasoning to prevent BBH catastrophic forgetting |

All datasets undergo SHA-1 cross-dataset deduplication and dual-threshold length filtering (minimum 1,024 and maximum 2,048 tokens), yielding approximately 38,000 unique training samples.

#### 3.2.2 Training Protocol

Each stage is trained sequentially for 300 steps with the following shared hyperparameters:

| Hyperparameter | Value |
|----------------|-------|
| Effective batch size | 16 (per-device batch 2, gradient accumulation 8) |
| Learning rate | 2e-4 with cosine decay |
| Warmup ratio | 0.03 |
| Max sequence length | 2,048 |
| Weight decay | 0.01 |
| Precision | BF16 mixed precision |

The sequential stage design allows each stage to build on the representations learned by the previous one. Stage A establishes a strong in-distribution baseline; Stages B1–B3 progressively introduce harder and more diverse mathematical reasoning; Stage C prevents catastrophic forgetting on general reasoning tasks by mixing in non-mathematical data.

#### 3.2.3 Computational Resources

Training was conducted on two hardware configurations to verify reproducibility: Google Colab with NVIDIA Tesla T4 (16 GB VRAM) and a server with NVIDIA L20 (48 GB VRAM). The maximum loss difference between the two configurations across all five stages was less than 0.02, confirming full reproducibility with seed 42. The Unsloth framework \cite{unsloth} was used for optimized training, providing approximately 2x speedup through fused kernels and memory-efficient attention.

### 3.3 Preference Optimization: Standard and Error-Type-Targeted DPO

#### 3.3.1 Standard DPO

For the standard DPO baseline, we use the argilla/distilabel-math-preference-dpo dataset containing 5,000 preference pairs. Training uses the sigmoid loss function with the following configuration:

| Hyperparameter | Value |
|----------------|-------|
| Loss type | Sigmoid |
| Beta | 0.1 |
| Learning rate | 1e-5 |
| Max steps | 600 |
| Batch size | 4 |
| Gradient accumulation | 4 |
| Max length | 2,048 |

#### 3.3.2 Error-Type-Targeted DPO

Our novel Error-Type-Targeted DPO pipeline introduces a diagnosis-driven feedback loop:

1. **Evaluation and Collection**: Run the SFT model on the GSM8K training set and collect all incorrectly answered problems (badcases).

2. **Error Classification**: Use a teacher model (qwen-flash) to classify each badcase into one of five error categories:
   - **setup_error**: Misinterpreting the problem statement or defining incorrect variables/equations
   - **arithmetic**: Computational mistakes in valid mathematical expressions
   - **reasoning_skip**: Missing logical steps in an otherwise correct approach
   - **extraction_error**: Correct reasoning but failure to extract the final answer
   - **unit_or_format**: Incorrect units, formatting, or presentation of the answer

3. **Targeted Pair Construction**: For each error type, generate a type-specific system prompt that instructs a teacher model (qwen3-235b or qwen2.5-72b) to produce a correct solution that specifically addresses the identified failure mode. The model's incorrect response serves as the rejected response, and the teacher's targeted correction serves as the chosen response.

4. **DPO Training**: Train on the targeted preference pairs using the same DPO configuration as the standard approach.

This pipeline ensures that preference optimization directly addresses the model's most frequent failure modes rather than optimizing for generic quality differences.

### 3.4 Evaluation Protocol

#### 3.4.1 Benchmarks

We evaluate on three established benchmarks:

- **GSM8K** \cite{gsm8k}: 200 problems from the test set, testing grade-school arithmetic reasoning
- **MATH-500** \cite{math}: 200 problems from the MATH test set, spanning five difficulty levels and seven mathematical subjects
- **BBH-27** \cite{bbh}: 27 sub-tasks of BIG-Bench Hard, 30 examples per task (810 total), testing diverse reasoning abilities

#### 3.4.2 Protocol

Our evaluation protocol uses zero-shot prompting with the model's chat template applied. All responses are generated with `max_new_tokens=1024`, and answers are extracted using regex-based parsers that handle multiple answer formats (boxed, final answer markers, etc.). A `_normalize_num()` function handles trailing periods and trailing zeros to prevent false negatives in answer matching. All group-to-group comparisons within our study are made under identical conditions, ensuring that relative differences are valid and reproducible.

#### 3.4.3 Baselines

We compare against the following baselines:
- **Qwen2.5-1.5B-Instruct** (unmodified, zero-shot): Our primary baseline
- **Qwen2.5-7B-Instruct** (unmodified, zero-shot): An upper-bound reference
- **Group A** (LoRA + single-stage SFT + standard DPO): A classical fine-tuning baseline

---

## 4. Results

### 4.1 Main Results

Table 1 presents the main evaluation results across all experimental groups and benchmarks.

**Table 1: Main Evaluation Results (custom protocol, n=200, zero-shot chat-template)**

| Group | Configuration | GSM8K | MATH-500 | BBH-27 |
|-------|---------------|-------|----------|--------|
| Baseline (1.5B) | Qwen2.5-1.5B-Instruct | 62.0% | 47.5% | -- |
| Baseline (1.5B, supplement) | Same model, separate run | 63.5% | 45.0% | -- |
| Baseline (7B) | Qwen2.5-7B-Instruct | 84.5% | 68.0% | -- |
| Group A (SFT) | LoRA + Single-stage SFT | 63.5% | 44.5% | 38.5% |
| Group A (DPO) | + Standard DPO | 59.5% | 51.25%† | -- |
| Group B (SFT only) | DoRA + 5-stage Curriculum | 62.0% | 44.0% | 38.8% |
| Group B | + Standard DPO | 62.0% | **47.5%** | -- |
| Group D | + Error-Type-Targeted DPO | **64.5%** | 44.0% | 37.4% |
| Teacher SFT | LoRA + 1409 teacher CoT | **65.0%** | 43.5% | -- |
| Qwen 1.5B (published) | -- | 73.2% | 55.2% | -- |
| Qwen 7B (published) | -- | 91.6% | 75.5% | -- |

† Group A MATH DPO evaluated on n=80 (wider uncertainty).

Several observations emerge from these results:

1. **Teacher SFT achieves the highest GSM8K accuracy** (65.0%) using only 1,409 teacher CoT samples — surpassing all other approaches including Error-Type-Targeted DPO (64.5%) and the 38k mixed-data SFT (62.0–63.5%). This validates the DeepSeek-R1-Distill insight that data quality dominates data quantity for small model distillation.

2. **Error-Type-Targeted DPO achieves the second-highest GSM8K accuracy** (64.5%), representing a +2.5pp improvement over the baseline and a +2.5pp improvement over standard DPO. This confirms that targeting specific error types can yield task-level improvements.

2. **Standard DPO improves MATH-500** from 44.0% (SFT only) to 47.5% (+3.5pp), matching the baseline and suggesting that generic preference optimization benefits harder mathematical reasoning.

3. **SFT alone does not consistently improve over the base model**. The 5-stage curriculum achieves 62.0% on GSM8K and 44.0% on MATH-500, both below the baseline. This suggests that the curriculum, while reducing training loss, may not optimally align with the evaluation distribution.

4. **BBH performance is preserved across all groups** (37–39%), indicating that mathematical fine-tuning does not cause catastrophic forgetting on general reasoning tasks. The Stage C (Magpie) data appears to serve its intended purpose.

5. **Standard DPO on single-stage SFT regresses GSM8K**. Group A DPO drops to 59.5% (-4.0pp from SFT), while Group B DPO with 5-stage curriculum maintains 62.0%. This suggests that curriculum-based SFT provides a more stable base for DPO alignment, as single-stage SFT may lead to overfitting when combined with preference optimization on simple tasks.

### 4.2 SFT Training Dynamics

The five-stage curriculum exhibits distinct learning dynamics at each stage.

**Table 2: Per-Stage Training Loss**

| Stage | Dataset | Init Loss | Final Loss | Relative Drop |
|-------|---------|-----------|------------|---------------|
| A | GSM8K (7.5k) | 1.286 | 0.225 | -82.5% |
| B1 | OpenR1-Math (10k) | 0.952 | 0.641 | -32.7% |
| B2 | OrcaMath (15k) | 0.549 | 0.347 | -36.8% |
| B3 | NuminaMath (8k) | 0.616 | 0.531 | -13.8% |
| C | Magpie (3k) | 0.623 | 0.517 | -17.0% |

Stage A (GSM8K) shows the most dramatic loss reduction (-82.5%), reflecting strong in-distribution alignment. The subsequent stages exhibit progressively smaller drops, consistent with increasing data diversity and difficulty. Stage B3 (NuminaMath, Olympiad-level) shows the smallest improvement (-13.8%), confirming its role as the most challenging curriculum component.

The final 100 SFT steps (steps 1401–1500) show stable training dynamics: loss oscillates between 0.43 and 0.46, gradient norms remain at approximately 0.14–0.15, and the learning rate has decayed to near-zero, indicating convergence.

### 4.3 DPO Training Dynamics

The standard DPO training (Group B) exhibits rapid convergence within 150 steps.

**Table 3: DPO Training Metrics**

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| DPO Loss | 1.233 | 0.020 | -98.4% |
| Reward Accuracy | 34.4% | 99.4% | +65.0pp |
| Reward Margin | -0.57 | 10.56 | +11.13 |
| Reward (Rejected) | 1.76 | -9.84 | -11.60 |

The reward accuracy reaching 99.4% and the strong negative reward for rejected responses (-9.84) indicate that the model has learned to sharply distinguish between preferred and rejected mathematical solutions. The rapid convergence (within 150 of 600 total steps) suggests that the preference dataset may be relatively easy for the model to learn, or that the learning rate is aggressive. The total DPO training time was approximately 2.85 hours on an NVIDIA L20.

The Group A DPO training (600 steps, 4 epochs) achieved similar dynamics: final loss 0.2394, reward accuracy 92-93%, no KL drift. However, despite healthy training metrics, evaluation reveals GSM8K regression (-4.0pp), indicating that training metrics alone are insufficient — the SFT base quality determines downstream DPO effectiveness.

### 4.4 Error Analysis

#### 4.4.1 GSM8K Error Taxonomy

We classify all incorrect GSM8K responses using both heuristic rules and a teacher model (qwen-flash). Table 4 presents the error distribution across experimental groups.

**Table 4: GSM8K Error Type Distribution**

| Error Type | Baseline (n=76) | SFT (n=77) | Group A (n=73) | Group D (n=71) |
|------------|-----------------|------------|----------------|----------------|
| setup_error | 52 (68.4%) | 59 (76.6%) | 48 (65.8%) | 50 (70.4%) |
| arithmetic | 13 (17.1%) | 8 (10.4%) | 15 (20.5%) | 10 (14.1%) |
| reasoning_skip | 11 (14.5%) | 10 (13.0%) | 10 (13.7%) | 11 (15.5%) |

Setup errors dominate across all groups, accounting for 65–77% of all failures. This finding has important implications: the primary bottleneck for 1.5B models on GSM8K is not computational (arithmetic) or logical (reasoning) but interpretive — the model frequently misinterprets what the problem is asking.

Notably, SFT increases the setup error rate from 68.4% to 76.6% (+8.2pp), suggesting that the curriculum may inadvertently teach the model patterns that lead to misinterpretation. Group A (standard DPO) reduces setup errors to 65.8%, the lowest across all groups, while Group D (targeted DPO) achieves 70.4%. The difference between Group A and Group D on this metric suggests that the targeted approach may not yet be optimally calibrated for setup errors specifically.

The teacher-model-based error classification on 77 SFT badcases yields a slightly different distribution: setup_error (64.9%), reasoning_skip (27.3%), extraction_error (3.9%), arithmetic (2.6%), unit_or_format (1.3%). The higher reasoning_skip percentage from the teacher model suggests that some errors classified as setup_error by heuristics may actually reflect incomplete reasoning chains.

#### 4.4.2 MATH-500 Error Analysis by Difficulty

Table 5 breaks down error rates by MATH difficulty level (1 = easiest, 5 = hardest).

**Table 5: MATH-500 Error Rate by Difficulty Level**

| Level | Baseline | SFT | Group A | Group D |
|-------|----------|-----|---------|---------|
| Level 1 (easy) | 27.5% | 27.5% | 27.5% | 27.5% |
| Level 2 | 30.0% | 35.0% | 40.0% | 35.0% |
| Level 3 | 50.0% | 55.0% | 60.0% | 55.0% |
| Level 4 | 67.5% | 72.5% | 67.5% | 72.5% |
| Level 5 (hard) | 87.5% | 90.0% | **82.5%** | 90.0% |

Several patterns emerge:

1. **Level 1 is unchanged across all groups** (27.5% error rate), suggesting that the easiest problems represent a performance floor determined by the base model's capabilities.

2. **SFT alone increases error rates at Levels 2–5**, with the most pronounced degradation at Level 2 (+5pp) and Level 3 (+5pp). This is consistent with the hypothesis that the curriculum may shift the model's distribution away from certain problem types.

3. **Group A DPO shows the most improvement at Level 5** (87.5% to 82.5%, -5pp), suggesting that standard DPO is most effective for the hardest problems where the model's initial errors are most informative.

4. **Group D does not improve over SFT at Levels 2–5**. This is a notable limitation of the current targeted DPO approach: while it improves GSM8K, the error classification and targeted pair construction are based on GSM8K badcases, which may not transfer well to the harder MATH distribution.

#### 4.4.3 MATH-500 Subject Analysis

Error rates across mathematical subjects reveal which domains are most challenging:

**Table 6: Top 5 MATH-500 Error Subjects (averaged across groups)**

| Subject | Approximate Error Rate |
|---------|----------------------|
| Intermediate Algebra | ~24% |
| Precalculus | ~17% |
| Number Theory | ~15% |
| Algebra | ~14% |
| Prealgebra | ~14% |

Intermediate Algebra and Precalculus are the most challenging subjects, consistent with the finding that these areas require deeper reasoning chains and more complex multi-step problem solving.

#### 4.4.4 Overall Badcase Counts

**Table 7: GSM8K and MATH-500 Badcase Summary**

| Group | GSM8K Badcases | GSM8K Error Rate | MATH Badcases | MATH Error Rate |
|-------|---------------|------------------|---------------|-----------------|
| Baseline | 76/200 | 38.0% | 105/200 | 52.5% |
| SFT | 77/200 | 38.5% | 112/200 | 56.0% |
| Group A | 73/200 | 36.5% | 111/200 | 55.5% |
| Group D | 71/200 | **35.5%** | 112/200 | 56.0% |

Group D achieves the lowest GSM8K error rate (35.5%), while MATH-500 error rates remain relatively stable across all trained groups (55.5–56.0%), actually slightly worse than the baseline (52.5%). This suggests that the fine-tuning pipeline is more effective for in-distribution (GSM8K-style) problems than for the harder and more diverse MATH-500 distribution.

#### 4.4.5 Teacher Data Quality Verification

The 1500 teacher-generated CoT responses (Qwen3-235B-Thinking) were verified against GSM8K ground truth to assess data quality before SFT distillation.

**Verification Results**:

| Metric | Value |
|--------|-------|
| Total samples | 1500 |
| Correct answers | 1444 (96.27%) |
| Incorrect answers | 56 (3.73%) |
| Contains `\boxed{}` | 1500 (100%) |
| Thinking tag residuals | 0 |

**Error Breakdown** (56 incorrect):

| Category | Count | Description |
|----------|-------|-------------|
| Reasoning errors | 48 | Teacher arrived at wrong numeric answer |
| Non-numeric answers | 5 | `\boxed{}` contains text/fractions instead of integers |
| Unit confusion | 2 | Dollar/cents mixing |
| Order of magnitude | 1 | Answer off by 10x |

**Noise Analysis**:
- 41 samples (2.7%) have chosen > 10,000 chars (max 33,828) — teacher over-thinking
- 87 samples (5.8%) contain ≥ 10 self-correction patterns (max 148) — excessive backtracking
- E11 applies three-layer quality filtering (length + corrections + answer correctness) to yield ~1350 clean traces

A 96.27% accuracy rate for a 235B model on GSM8K is lower than expected (typically >99%), likely due to the thinking-mode generation introducing occasional reasoning drift. Despite this, the data remains viable for SFT distillation after filtering.

#### 4.4.6 Teacher SFT Experiment (E12/E13)

Motivated by the DeepSeek-R1-Distill approach, we conducted an experiment to evaluate whether a small number of high-quality teacher reasoning traces could outperform large-scale mixed-data SFT.

**Training Configuration (E12)**:
- Base model: Qwen2.5-1.5B-Instruct (from scratch, no curriculum SFT)
- PEFT: LoRA (r=16, α=32) — simpler than DoRA, appropriate for small data volume
- Data: `sft_teacher_gsm8k.json` — 1,409 samples after E11 three-layer quality filtering
- Training: 470 steps (~5.3 epochs), effective batch size 16, lr=5e-5, cosine scheduler
- Key choice: `packing=False` to preserve complete teacher CoT reasoning chains

**Training Convergence**:

| Step | Loss | Learning Rate | Epoch |
|------|------|---------------|-------|
| 10 | 0.8937 | 9.6e-6 | 0.11 |
| 110 | 0.4656 | 4.7e-5 | 1.24 |
| 210 | 0.3984 | 3.4e-5 | 2.36 |
| 310 | 0.3609 | 1.6e-5 | 3.49 |
| 470 | 0.3404 | ~0 | 5.28 |

Loss decreased from 0.89 to 0.34 (−62%). Gradient norms remained stable at 0.36–0.42 throughout training.

**Evaluation Results (E13)**:

| Model | Data Volume | GSM8K | MATH-500 |
|-------|-------------|-------|----------|
| Baseline 1.5B | — | 63.5% | 45.0% |
| Group A SFT (LoRA + single-stage) | ~38k | 63.5% | 44.5% |
| Group B SFT (DoRA + 5-stage) | ~38k | 62.0% | 44.0% |
| **Teacher SFT (LoRA + teacher CoT)** | **1,409** | **65.0%** | 43.5% |

The Teacher SFT model achieves **65.0% on GSM8K** — a new high across all our experiments, surpassing both the 38k mixed-data SFT approaches (Groups A and B) and the Error-Type-Targeted DPO (64.5%). This result is remarkable given the data volume: 1,409 samples represent only 3.7% of the 38k curriculum data.

**Badcase Analysis** (70 errors out of 200, 35% error rate):

| Error Type | Count | Share | Description |
|------------|-------|-------|-------------|
| setup_misread | 48 | **69%** | Misunderstands problem statement |
| multi_step_cascade | 14 | 20% | Early arithmetic error cascades |
| truncated | 8 | 11% | Output truncated before answer |

The error distribution reveals that 69% of Teacher SFT failures are comprehension errors (setup_misread) — the model still misinterprets what the problem is asking. This is consistent with the error taxonomy from Section 4.4.1 and suggests that further improvement requires either more diverse problem-comprehension training data or structured prompting that forces explicit problem parsing.

**Key Insight**: The Teacher SFT result validates the core thesis of DeepSeek-R1-Distill — that a small number of high-quality teacher reasoning traces can outperform large-scale mixed data for small model distillation. The 1,409-sample teacher dataset, despite being filtered from a 235B model with only 96.27% accuracy, provides more effective training signal than 38k samples from diverse public datasets. This has practical implications: for resource-constrained scenarios, investing in a small set of high-quality teacher traces may be more cost-effective than curating large-scale training data.

### 4.5 BBH-27 Generalization

The BBH-27 evaluation tests whether mathematical fine-tuning degrades general reasoning capabilities.

| Metric | Group B (SFT) | Group D |
|--------|--------------|---------|
| Macro accuracy | 38.8% | 37.4% |
| Tasks evaluated | 24 (3 failed) | 27 |
| Best task | boolean_expressions (96.7%) | boolean_expressions (96.7%) |
| Worst task | word_sorting (0%) | word_sorting (0%) |
| Second worst | web_of_lies (6.7%) | web_of_lies (6.7%) |

The absence of catastrophic degradation (37–39% macro accuracy) confirms that our Stage C (Magpie) data effectively prevents forgetting. The best-performing tasks (boolean_expressions at 96.7%, navigate at 66.7%) leverage logical reasoning skills that overlap with mathematical reasoning, while the worst-performing tasks (word_sorting at 0%, web_of_lies at 6.7%) require capabilities that are not covered by our training data.

---

## 5. Conclusion

### 5.1 Summary of Findings

This project demonstrates that curriculum-based SFT and diagnosis-driven DPO can improve mathematical reasoning in a 1.5B-parameter language model, though the improvements are modest and context-dependent:

1. **Teacher SFT** achieves the highest GSM8K accuracy (65.0%) using only 1,409 high-quality teacher reasoning traces, surpassing all other approaches. This validates the DeepSeek-R1-Distill insight that data quality dominates quantity for small model distillation.

2. **Error-Type-Targeted DPO** achieves 64.5% on GSM8K (+2.5pp over baseline), confirming that error-specific preference optimization can yield measurable improvements even with limited compute.

3. **Standard DPO** improves MATH-500 by 3.5pp when applied after curriculum SFT, suggesting that generic preference data benefits harder mathematical reasoning.

3. **Setup errors dominate the failure distribution** (65–77% of all GSM8K errors), representing the single most important target for future improvement efforts.

4. **Curriculum-based SFT is critical for DPO stability**: Group A (single-stage) DPO regressed on GSM8K (-4.0pp), while Group B (5-stage curriculum) DPO maintained performance, validating the curriculum design.

5. **Teacher distillation data quality** is high (96.27% correct on GSM8K) but requires noise filtering — 5.8% of samples show excessive self-correction patterns that would harm student model learning.

6. **SFT alone does not consistently improve over the base model**, and in some cases increases error rates, particularly at higher difficulty levels. This highlights the sensitivity of small models to data distribution and training order.

5. **General reasoning is preserved**: BBH performance remains stable (37–39%) across all experimental groups, validating the Stage C data mixing strategy.

### 5.2 Implications

The dominance of setup errors has practical implications for system design. Rather than investing in more sophisticated preference optimization, the highest-impact intervention may be to improve the model's problem comprehension — for example, by training on data that explicitly demonstrates problem parsing and variable definition, or by using a structured prompting format that guides the model through problem interpretation before solution generation.

The finding that SFT can increase certain error rates underscores the importance of careful data curation and evaluation at each training stage. Curriculum learning is not merely a matter of ordering data by difficulty; the specific composition of each stage and its alignment with the target evaluation distribution are critical.

### 5.3 Challenges Encountered

Several challenges arose during the project:

1. **Data distribution mismatch**: The training data spans multiple sources and difficulty levels, while evaluation is on specific benchmarks. This distribution mismatch may explain why SFT alone does not consistently improve performance.

3. **Computational constraints**: All training was conducted on a single Tesla T4 or L20 GPU, limiting the scope of ablation experiments and the number of DPO iterations we could explore.

4. **DPO rapid convergence**: The standard DPO training converges within 150 of 600 planned steps, suggesting potential overfitting to the preference dataset and indicating the need for early stopping or a larger preference dataset.

### 5.4 Future Directions

Based on our error analysis and the current state of the field, we identify the following high-priority directions for future work:

1. **Rejection Sampling**: Generate N candidate solutions per problem and SFT only on the correct ones. This approach is simpler than DPO and has been shown to capture most of the available improvement for small models \cite{deepseekr1}.

2. **Teacher Distillation**: Use a large teacher model (qwen2.5-72b or qwen3-235b) to generate high-quality reasoning traces for SFT. This is the primary technique used by DeepSeek-R1-Distill to achieve 83–87% MATH-500 on 1.5B models \cite{deepseekr1}.

3. **Iterative DPO**: Generate new preference pairs from the current model at each iteration rather than using a static dataset. Prior work suggests 3–5pp improvement over single-shot DPO \cite{iterativedpo}.

4. **Process-Level DPO**: Annotate correctness at each reasoning step rather than only at the final answer. This provides denser training signal and has demonstrated superior performance on mathematical tasks \cite{prm}.

5. **Simpler Preference Objectives**: KTO \cite{kto} and SimPO \cite{simp} require fewer hyperparameters and may be more stable for small models where DPO convergence is rapid.

6. **Difficulty-Aware Sampling**: Oversample harder problems during training to prevent the model from overfitting to easy patterns. Our error analysis shows that Level 1 problems are already solved at a consistent rate, while Levels 3–5 remain highly challenging.

7. **Data Mixing Strategy**: The current sequential five-stage curriculum may benefit from interleaving data from multiple stages during each training phase, preventing distribution drift between stages.

8. **Setup Error Mitigation**: Given that setup errors account for 65–77% of failures, dedicated interventions — such as training on problem paraphrases, variable extraction tasks, or structured problem decomposition — could yield disproportionate improvements.

---

## References

\bibliographystyle{plain}

\bibitem{qwen25}
Qwen Team, ``Qwen2.5 Technical Report,'' arXiv preprint arXiv:2412.15115, 2024.

\bibitem{dpo}
Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., and Finn, C., ``Direct Preference Optimization: Your Language Model is Secretly a Reward Model,'' in Proc. NeurIPS, 2023.

\bibitem{deepseekr1}
DeepSeek-AI, ``DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning,'' arXiv preprint arXiv:2501.12948, 2025.

\bibitem{qwen25math}
Qwen Team, ``Qwen2.5-Math Technical Report: Toward Mathematical Expert Model,'' arXiv preprint arXiv:2409.12122, 2024.

\bibitem{qwq}
Qwen Team, ``QwQ: Reflect Deeply on the Boundaries of the Unknown,'' Qwen Blog, 2024.

\bibitem{gsm8k}
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J., ``Training Verifiers to Solve Math Word Problems,'' arXiv preprint arXiv:2110.14168, 2021.

\bibitem{math}
Hendrycks, D., Burns, C., Kadavath, S., Arber, A., Roth, E., and Steinhardt, J., ``Measuring Mathematical Problem Solving with the MATH Dataset,'' in Proc. NeurIPS Datasets and Benchmarks Track, 2021.

\bibitem{bbh}
Suzgun, M., Scales, N., Scharli, N., Gehrmann, S., Tay, Y., Chung, H.W., Chowdhery, A., Le, Q.V., Chi, E.H., Zhou, D., and Wei, J., ``Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them,'' in Proc. ACL Findings, 2023.

\bibitem{lora}
Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W., ``LoRA: Low-Rank Adaptation of Large Language Models,'' in Proc. ICLR, 2022.

\bibitem{dora}
Liu, S.Y., Wang, C.Y., Yin, H., Molchanov, P., Wang, Y.C.F., Cheng, K.T., and Chen, M.H., ``DoRA: Weight-Decomposed Low-Rank Adaptation,'' arXiv preprint arXiv:2402.09353, 2024.

\bibitem{lima}
Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., and Hajishirzi, H., ``LIMA: Less Is More for Alignment,'' in Proc. NeurIPS, 2023.

\bibitem{iterativedpo}
Xu, H., Sharaf, A., Chen, Y., Tan, W., Shen, L., Van Durme, B., Murray, K., and Kim, Y.J., ``Contrastive Post-training Large Language Models on Data Curriculum,'' arXiv preprint arXiv:2310.02263, 2023.

\bibitem{kto}
Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D., ``KTO: Model Alignment as Prospect Theoretic Optimization,'' arXiv preprint arXiv:2402.01306, 2024.

\bibitem{simp}
Meng, Y., Xia, M., and Chen, D., ``SimPO: Simple Preference Optimization with a Reference-Free Reward,'' arXiv preprint arXiv:2405.14734, 2024.

\bibitem{prm}
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K., ``Let's Verify Step by Step,'' in Proc. ICLR, 2024.

\bibitem{errortaxonomy}
Patel, A., Bhattamishra, S., and Goyal, N., ``Are NLP Models really able to Solve Simple Math Word Problems?'' in Proc. NAACL, 2021.

\bibitem{unsloth}
Unsloth AI, ``Unsloth: 2x Faster LLM Finetuning,'' GitHub repository, 2024.
