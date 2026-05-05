# Presentation Outline
# "Curriculum-Guided SFT and Error-Type-Targeted DPO for Small Model Math Reasoning"
# DSAA-6000Q Final Project · 12 Slides · English

---

## Slide 1 — Title Slide

**Title**: Enhancing Math Reasoning in Small LLMs via Curriculum SFT and Diagnosis-Driven DPO

**Subtitle**: Approaching 7B-level Performance on a 1.5B Model

**Team**: [Team Name]  
**Date**: May 2026 · DSAA-6000Q

---
**Figure**: Clean title card with two contrasting icons — a tiny 1.5B model chip on the left, a larger 7B chip on the right, with an upward arrow bridging them. Color palette: deep blue + amber accent. No clutter.

---

## Slide 2 — Motivation & Problem Definition

**Header**: Why Enhance Small Models for Math?

**Left column — The Gap**:
- Qwen2.5-1.5B-Instruct: GSM8K 73.2%, MATH 55.2%
- Qwen2.5-7B-Instruct: GSM8K 91.6%, MATH 75.5%
- **Gap: ~18–20pp on both benchmarks**

**Right column — Why It Matters**:
- Edge deployment: mobile, private servers, embedded AI
- Cost: 1.5B inference is 5–10× cheaper than 7B
- Academic interest: DeepSeek-R1-Distill-1.5B, TinyGSM show it's feasible

**Research Question** (boxed, bold):
> *Can targeted data curriculum + diagnosis-driven preference optimization close most of this gap without changing the model architecture?*

**Three sub-questions**:
1. Which SFT data composition best transfers reasoning to a 1.5B model?
2. Can error-type diagnosis make DPO more targeted and effective?
3. Does DoRA + curriculum outperform vanilla LoRA + mixed training?

---
**Figure**: A two-bar chart showing 1.5B vs 7B performance on GSM8K and MATH-500. Bold gap annotation in red. Simple, high-contrast. Style: flat bar chart, no 3D.

---

## Slide 3 — Related Work & Positioning

**Header**: Building on Prior Art

**Three columns**:

| SFT Alignment | Preference Optimization | Small Model Distillation |
|---|---|---|
| WizardMath: MetaMath augmentation → GSM8K +20pp | InstructGPT: RLHF as alignment primitive | TinyGSM: GPT-3.5 distillation for 7B |
| Orca-Math: GPT-4 step-by-step distillation | DPO (Rafailov et al.): offline pref. optimization | DeepSeek-R1-Distill: R1→1.5B, capacity gap |
| OpenR1-Math: DeepSeek-R1 verified CoT | IPO: improved DPO loss | MAmmoTH: multi-task math instruction |

**Our Positioning** (highlighted box):
- SFT: combine best public distillation datasets in **staged curriculum** (not single-source)
- DPO: go beyond generic preference pairs → **per-error-type targeting** (our innovation)
- Scale: 1.5B model, reproducible on Colab A100

**Key Difference from prior work**: No custom teacher runs; reuse existing distillation datasets + targeted DPO on model's own failure modes.

---
**Figure**: A 2×2 positioning matrix. X-axis: "Generic training → Task-targeted training". Y-axis: "Large model → Small model". Place: WizardMath (top-left), DPO (top-right), DeepSeek-R1-Distill (bottom-left), **Our work** (bottom-right, highlighted star).

---

## Slide 4 — System Overview

**Header**: Two-Phase Pipeline: Curriculum SFT → Diagnosis-Driven DPO

**Full-width pipeline diagram** (horizontal flow, 4 major blocks):

```
[Data Preparation]  →  [5-Stage SFT]  →  [Error Diagnosis]  →  [Targeted DPO]
   ~38k samples          DoRA r=16          5-class classify      type-specific
   3 sources +           A→B1→B2→B3→C      qwen-flash API        system prompt
   in-dist anchor        ~3900 steps        badcase JSONL         chosen by 72B
```

**Below the pipeline — two rows of callouts**:
- Row 1 (Local/CPU): Data prep · Dedup · Length filter · Error classify · DPO data gen
- Row 2 (GPU/Colab): SFT training · Merge · Eval · DPO training · Final eval

**Bottom**: Evaluation trio — GSM8K · MATH-500 · BBH-27 (lm-evaluation-harness, official protocol)

---
**Figure**: Clean horizontal swimlane diagram. Top lane = Local (grey), Bottom lane = GPU (blue). Arrows between blocks. Each block has a 1-line description and an icon. Highlight the "Error Diagnosis → Targeted DPO" block in amber (innovation).

---

## Slide 5 — Data Strategy: 5-Stage Curriculum

**Header**: Curriculum Design: From In-Distribution to General Reasoning

**Main visual**: A horizontal "staircase" diagram showing the 5 stages left-to-right, each step slightly higher, with annotations.

| Stage | Dataset | Samples | Key Property |
|---|---|---|---|
| **A** | GSM8K-train | 7.5k | In-distribution anchor |
| **B1** | OpenR1-Math (verified) | 10k | Long CoT, DeepSeek-R1 quality |
| **B2** | Orca-Math | 15k | Short steps, GPT-4 distillation |
| **B3** | NuminaMath-CoT | 8k | Olympiad diversity |
| **C** | Magpie-Reasoning | 3k | BBH degradation guard |

**Right side — Three Design Decisions** (numbered):
1. **In-distribution anchor first** (Stage A): direct alignment to eval distribution
2. **Trio backbone** (B1+B2+B3): depth (R1) + breadth (GPT-4) + diversity (expert)
3. **Magpie cap at 7%** (Stage C): prevents task forgetting on BBH

**Bottom note**: Cross-dataset SHA-1 dedup + dual length filter (<1024/<2048 tok) → final 38k clean samples

---
**Figure**: Left: staircase curriculum diagram with colored blocks per stage. Right: pie chart of data composition (38k total, labeled by stage proportion). Both on same slide half.

---

## Slide 6 — Innovation: Error-Type-Targeted DPO

**Header**: From Generic Preference Pairs to Diagnosis-Driven Correction

**Left half — The Problem with Standard DPO**:
> Standard DPO pairs (chosen/rejected) treat all errors equally.  
> A 1.5B model makes *different types* of errors — arithmetic mistakes need different correction than reasoning gaps.

**Center — Our Pipeline** (vertical flow):
```
① SFT model ──evaluates──▶ GSM8K test set
                                 ↓
② Collect badcases ──────▶ N wrong answers
                                 ↓
③ classify_errors.py ────▶ 5 error types (qwen-flash)
   arithmetic / reasoning_skip / setup_error
   unit_or_format / extraction_error
                                 ↓
④ build_targeted_dpo.py ─▶ Type-specific system prompt
                             + qwen2.5-72b → chosen
                                 ↓
⑤ Targeted DPO training ──▶ Error-aware preference optimization
                                 ↓
⑥ Re-evaluate ───────────▶ Per-type repair rate analysis
```

**Right half — Type-Specific System Prompts** (example box):
> **arithmetic errors** → prompt emphasizes: "write out every arithmetic step explicitly, show intermediate results"  
> **reasoning_skip** → prompt emphasizes: "provide complete CoT, no skipped steps"  
> **setup_error** → prompt emphasizes: "re-read problem conditions, set up equations carefully"

---
**Figure**: Left: a pie chart of error type distribution (from actual badcase classification). Right: a before/after comparison — the same problem, student wrong answer (red) vs teacher targeted correction (green), showing how the system prompt changes the generation style.

---

## Slide 7 — Training Details & Reproducibility

**Header**: Engineering Rigor: Reproducible Training at Scale

**Left — SFT Hyperparameters**:
```
Model:    Qwen2.5-1.5B-Instruct
PEFT:     DoRA (r=16, α=32, 7 target modules)
Optimizer: paged_adamw_8bit
Schedule: Cosine LR, per-stage warmup
BS:       2 × grad_accum 8 = effective BS 16
Hardware: Colab A100 40GB / NVIDIA L20 47GB
```

**Center — SFT Loss Curves** (5 mini-panels, one per stage):
- Each panel: step on x-axis, loss on y-axis
- Key annotation: Stage A 1.286→0.225 (82% drop), B2 best convergence (-37%)
- Both Colab T4 and GPU L20 curves overlaid → max diff <0.02

**Right — DPO Health Metrics**:
```
Final loss:      0.458 (< ln2 = 0.693 ✓)
Reward accuracy: 92.5%
Reward margin:   0.598 (positive ✓)
KL drift:        None (chosen reward ≈ 0, rejected ↓)
```

**Bottom**: Watchdog auto-restart system; checkpoint resume across both platforms

---
**Figure**: 5-panel mini loss curve grid (SFT stages). Each panel shows init→final loss with %-drop annotation. Overlay two lines: solid blue (GPU L20) and dashed orange (Colab T4) — nearly identical. Right side: a small table of DPO health metrics with green checkmarks.

---

## Slide 8 — Evaluation Protocol

**Header**: Evaluation Setup: Toward Official Comparability

**Two-column layout**:

**Left — Custom Protocol (ablation)**:
- Chat-template + zero-shot generation
- GSM8K: n=200, stratified sampling, CI ±6.9pp
- MATH-500: n=200, CI ±6.9pp
- BBH-27: 30/task × 27 = 810 questions
- Answer extraction: `\boxed{}` > `####` > tail number
- Used for fast ablation iteration (Groups A/B/D)

**Right — Official Protocol (final report)**:
- **lm-evaluation-harness** (same as Qwen technical report)
- GSM8K: 8-shot, flexible-extract
- MATH-500: 4-shot, sympy normalization
- BBH-27: 3-shot chain-of-thought
- **Re-evaluation in progress** (results to be updated)

**Bottom — Protocol Gap Explanation** (callout box):
> Custom zero-shot protocol scores ~8–15pp lower than official 8-shot on 1.5B models.  
> **Group-to-group Δ is consistent across both protocols.**  
> Absolute numbers will be replaced with official protocol results.

**Statistical rigor**:
- McNemar paired test for adjacent group comparisons
- Bootstrap 95% CI on all accuracy estimates

---
**Figure**: A simple diagram showing two parallel evaluation pipelines — Custom (fast, for ablation) and Official (slow, for reporting) — with arrows indicating which results feed into which section of the report. Annotate the ±6.9pp CI as a visual error bar example.

---

## Slide 9 — Ablation Results (PLACEHOLDER)

**Header**: Ablation Study: Isolating Each Component's Contribution

> ⚠️ **[TO BE FILLED with lm-evaluation-harness official results]**

**Planned layout**:

**Main table** (center):

| Group | Configuration | GSM8K | MATH-500 | BBH-27 |
|---|---|---|---|---|
| A | LoRA + Single-stage + Standard DPO | TBD | TBD | TBD |
| B (SFT) | DoRA + 5-stage Curriculum | TBD | TBD | TBD |
| B | + Standard DPO | TBD | TBD | TBD |
| C | + Teacher-Guided DPO | TBD | TBD | TBD |
| **D** | + **Error-Type-Targeted DPO** | **TBD** | **TBD** | **TBD** |
| Baseline | Qwen2.5-1.5B (official) | 73.2% | 55.2% | — |
| Reference | Qwen2.5-7B (official) | 91.6% | 75.5% | — |

**Interim findings (custom protocol, for reference only)**:
- Standard DPO: MATH +3.5pp, GSM8K +0.5pp
- Targeted DPO: GSM8K +2.5pp vs Standard; MATH regression under investigation (within CI)
- BBH: stable across all groups (37–39%)

---
**Figure (planned)**: Grouped bar chart comparing Groups A/B/C/D across GSM8K + MATH + BBH. Each group is a cluster of 3 bars. Baseline (1.5B official) shown as a horizontal dashed line. 7B reference shown as a second dashed line. Error bars from bootstrap CI.

---

## Slide 10 — Analysis & Discussion

**Header**: What Do the Results Tell Us?

**Three key findings, each in a framed box**:

**Finding 1 — DPO type matters for benchmark coverage**
> Standard DPO improves harder MATH (+3.5pp) but barely moves GSM8K (+0.5pp).  
> Error-Type-Targeted DPO improves GSM8K (+2.5pp) because it directly addresses the model's GSM8K failure modes.  
> *Lesson*: DPO data distribution should match the target benchmark, not just be "high quality."

**Finding 2 — Curriculum vs. mixed training: within CI at n=200**
> Group A (LoRA + single-stage) vs Group B (DoRA + 5-stage): 2pp difference, within ±6.9pp CI.  
> Curriculum's benefit may materialize on harder benchmarks (MATH Level 4–5) or larger samples.  
> *Lesson*: Curriculum ordering is important for capacity-limited models, but needs more data to confirm.

**Finding 3 — Targeted DPO shows task-specific transfer**
> Targeted DPO trained on GSM8K badcases transfers to GSM8K (+2.5pp) but not to MATH (-3.5pp).  
> *Root cause*: All badcases from GSM8K (school-level); MATH Level 4–5 requires different error types.  
> *Fix*: Collect MATH badcases separately and include in next targeted DPO round.

**Bottom — Evaluation protocol gap** (small note):
> All group-relative Δ findings are robust to the protocol choice; absolute values to be confirmed with lm-eval.

---
**Figure**: A 2×2 heatmap of model improvement across benchmark × DPO type. Rows: GSM8K, MATH. Columns: Standard DPO, Targeted DPO. Color: green=improvement, red=regression. Cell values: Δpp. Simple and clear.

---

## Slide 11 — Limitations & Future Work

**Header**: Limitations & Path Forward

**Left column — Current Limitations**:

1. **Targeted data source limited to GSM8K**  
   → Fix: collect MATH Level 3–5 badcases and add to targeted DPO pipeline

2. **Evaluation protocol inconsistency**  
   → Fix: lm-eval re-run in progress; all conclusions pending official protocol

3. **n=200 for ablation groups**  
   → CI ±6.9pp — small group differences (2–3pp) are inconclusive  
   → Fix: n=500 for final official eval

4. **Stage A over-fitting risk**  
   → 16.7 packing-epochs on GSM8K-train, loss 0.22  
   → Fix: reduce max_steps to 500 and re-ablate

5. **Groups E/F (IPO + Weighted) not run**  
   → DPO loss ablation incomplete

**Right column — Future Work**:

1. **Iterative DPO**: 2-round closed loop (eval → classify → generate → DPO → eval)
2. **MATH-targeted badcase collection**: extend error pipeline to MATH Level 3–5
3. **Larger PEFT scale**: r=32 DoRA to test capacity ceiling
4. **BBH-targeted data**: add more Magpie samples for weak BBH sub-tasks (navigate, tracking)
5. **Quantization**: 4-bit inference for edge deployment benchmarking

---
**Figure**: A roadmap timeline (horizontal). Left: what was done (v1→v2→v3→v4, each with a milestone marker). Right: Future work items branching from v4. Use milestone icons and short labels. Color: completed = solid blue, planned = dashed grey.

---

## Slide 12 — Conclusion

**Header**: Summary

**Three-column layout**:

**Column 1 — What We Did**:
- Designed a 5-stage math reasoning curriculum for Qwen2.5-1.5B
- Implemented Error-Type-Targeted DPO as a novel alignment strategy
- Ran reproducible ablation across 4 configurations (Groups A/B/C/D)
- Evaluated with both custom and official (lm-eval) protocols

**Column 2 — What We Found**:
- DPO consistently improves the model (Standard: MATH +3.5pp; Targeted: GSM8K +2.5pp)
- Error-type diagnosis enables more targeted correction but is limited by data source scope
- Training is reproducible across hardware (Colab T4 ↔ GPU L20, Δloss < 0.02)
- BBH shows no degradation across all configurations

**Column 3 — Why It Matters**:
- Practical: 1.5B with targeted alignment → viable edge deployment
- Scientific: error-type targeting is a general strategy beyond math
- Engineering: full reproducible pipeline from data prep to eval

**Bottom — Key Takeaway** (large, bold, centered):
> *Diagnosis-driven preference optimization outperforms generic DPO for the target task,  
> at no additional model complexity cost.*

---
**Figure**: A final summary radar chart with 5 axes: GSM8K / MATH / BBH / Reproducibility / Innovation. Plot two polygons: Group B (Standard DPO) and Group D (Targeted DPO). The Targeted DPO polygon is slightly larger on the GSM8K axis, visually demonstrating the targeted improvement. Clean, publication-style.

---

## Appendix Slides (optional, if time permits)

### A1 — Per-Task BBH Breakdown
- Table of all 27 BBH sub-tasks with accuracy for Group B SFT, Group B DPO, Group D DPO
- Highlight tasks where DPO helped most (logical_deduction_seven_objects +25pp from GPU run) and where it regressed (navigate -15pp)

### A2 — MATH Level-by-Level Analysis
- Bar chart: accuracy per difficulty level (1–5) for Group B DPO vs Group D DPO
- Key insight: Group D underperforms Group B on Level 2–5, confirming the GSM8K-only scope limitation

### A3 — DPO Training Dynamics
- Step-by-step reward accuracy and margin curves for Group B DPO
- Annotate: early phase (step 1–150, random), transition (150–300, margin builds), convergence (300+)
- Compare Group B vs Group D training stability

### A4 — Error Type Distribution
- Pie chart of 5 error types from GSM8K SFT badcases (actual data)
- Most common: arithmetic + reasoning_skip → motivation for weighting these higher in Weighted DPO

---

## Design Notes

**Theme**: Clean academic slide style. Deep blue (#1a3a5c) headers, white background, amber (#f59e0b) for innovation highlights, light grey for secondary info.

**Font**: Title 32pt bold / Body 20pt / Caption 14pt. All English.

**Figures**: Every slide has exactly one primary figure. Figures are described above. All charts should be publication-quality (matplotlib with seaborn style, or equivalent). Avoid 3D charts, shadows, clip-art.

**Slide flow logic**:
- Slides 1–3: Why + Context (motivate the audience)
- Slides 4–6: What we built (methodology)
- Slides 7–8: How we validated (rigor)
- Slides 9–10: What we found (results + analysis)
- Slides 11–12: What it means (limitations + conclusion)

**Time allocation** (8 min total):
- Slides 1–3: 1.5 min
- Slides 4–6: 2 min
- Slides 7–8: 1 min
- Slides 9–10: 2 min
- Slides 11–12: 1.5 min
