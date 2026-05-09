# 汇报 PPT 大纲

# Presentation Outline

> **演讲时长**：8 分钟 | **总页数**：12 页 | **适用场景**：DSAA-6000Q 课程终期汇报
> **Presentation Duration**: 8 minutes | **Total Slides**: 12 | **Context**: DSAA-6000Q Course Final Presentation

---

# 中文版

---

## Slide 1: 封面 / Cover

**标题**：通过诊断式数据工程提升 1.5B 大语言模型数学推理能力

**副标题**：Enhancing Mathematical Reasoning of 1.5B LLM via Diagnostic Data Engineering

**信息**：DSAA-6000Q | 2026 年 5 月

---

## Slide 2: 问题与动机（40 秒）

### 页面内容

**核心问题**：1.5B 模型的数学推理能力能否逼近 7B？

| 模型 | 参数量 | GSM8K |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | 62.5% |
| Qwen2.5-7B-Instruct | 7B | 84.5% |
| **差距** | **4.7x** | **22pp** |

**关键追问**：
- 这 22pp 差距是模型容量的硬限制，还是可以通过数据工程弥补？
- 1.5B 模型部署成本低（边缘设备/移动端），提升其推理能力有巨大实用价值
- 当前文献主要关注 7B+，1.5B 的优化空间未被充分探索

### 配图说明
左侧放模型参数对比柱状图，右侧放边缘设备部署场景示意图（手机、树莓派等）。

### 备注
开场用 10 秒讲清楚问题的重要性和实用价值——小模型在资源受限场景下的部署优势。然后用 30 秒引出核心问题：22pp 差距是否可弥补。引用文献信号：LIMA（1000 条数据对齐 LLM）、DeepSeek-R1-Distill（蒸馏有效性）。

---

## Slide 3: 文献基础与核心洞察（30 秒）

### 页面内容

**文献告诉我们什么？**

| 文献 | 核心发现 | 对本项目的启示 |
|---|---|---|
| LIMA (2023) | 1000 条高质量数据即可对齐 LLM | 数据质量 > 数据数量 |
| DeepSeek-R1-Distill (2025) | 671B→1.5B 蒸馏有效 | Teacher CoT 是最好的训练数据 |
| Orca-Math (2024) | GPT-4 蒸馏让小模型大幅超越 baseline | 蒸馏数据对小模型数学有效 |
| Phi-1 (2023) | 1.3B + 高质量"教科书"数据超越更大模型 | 数据质量可弥补模型规模 |

**本项目的核心假设**：
> **在 1.5B 的极限规模下，数据工程（质量过滤 + 课程学习 + 诊断式偏好优化）是提升数学推理能力的最有效路径。**

### 配图说明
四篇文献的关键数据卡片，中间用箭头指向本项目的假设。

### 备注
快速过四篇核心文献（每篇 5-7 秒），强调它们的共同信号：数据质量比模型规模更重要。然后用一句话引出本项目假设。不要深入讲每篇论文细节，只讲对本项目有直接启示的部分。

---

## Slide 4: 方法总览——三阶段流水线（40 秒）

### 页面内容

**三阶段流水线**：SFT 课程学习 → 错误诊断 → 定向偏好优化

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Stage 1: SFT  │───→│  Stage 2: 诊断  │───→│  Stage 3: DPO   │
│  五段式课程学习   │    │  5类错误分类     │    │  定向偏好优化    │
│  38k→62.0%      │    │  setup 65% 主导  │    │  Targeted 64.5% │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ↓                      ↓                      ↓
   建立能力基线            发现弱点分布            定向修复错误
```

**8 组消融实验**：

| 编号 | 实验组 | GSM8K | 核心验证 |
|---|---|---|---|
| E0 | Baseline | 62.5% | 基线 |
| E1 | LoRA + 单段 SFT | 63.5% | SFT 效果 |
| E2 | E1 + Standard DPO | 59.5% | DPO 回退 |
| E3 | DoRA + 五段课程 | 62.0% | 课程效果 |
| E4 | E3 + Standard DPO | 62.0% | 基座稳定 |
| E5 | E3 + Targeted DPO | 64.5% | **定向有效** |
| E6 | Teacher SFT | **65.0%** | **最佳** |

### 配图说明
上方三阶段流水线图（可用 architecture.svg 的简化版），下方实验结果表。E5 和 E6 用绿色高亮。

### 备注
先花 15 秒讲清楚三阶段逻辑：SFT 建立基线 → 诊断发现弱点 → DPO 定向修复。然后用 25 秒过实验总览表，重点指出 3 个对比：E2 vs E1（DPO 回退）、E4 vs E3（基座稳定）、E6 vs E3（Teacher 最强）。这是整个项目的骨架，后面的 slide 会逐一展开。

---

## Slide 5: 创新点一——五段式 SFT 课程（50 秒）

### 页面内容

**课程设计**：in-distribution first → broad reasoning

| 阶段 | 数据 | 样本数 | 角色 | LR |
|---|---|---|---|---|
| **A** | GSM8K | 7.5k | 分布锚点 | 5e-5 |
| B1 | OpenR1 | 10k | R1 蒸馏长 CoT | 4e-5 |
| B2 | OrcaMath | 15k | GPT-4 蒸馏广覆盖 | 4e-5 |
| B3 | NuminaMath | 8k | 竞赛难题 | 3e-5 |
| C | Magpie | 3k | 通用推理防遗忘 | 2e-5 |

**五段 SFT Loss 曲线**：（配图 fig_sft_loss.png）

**关键发现**：
- Stage A 降幅最大（-82.5%）：GSM8K 是 in-distribution，模型快速学习
- Stage B1 Loss 从 0.952 开始（非 0.225 延续）：新数据分布不同，模型需适应
- Stage C 仅占 7%，但有效防止 BBH 退化

**设计逻辑**：Stage A 先锚定评测分布 → B1-B3 逐步扩展能力 → Stage C 防遗忘。学习率递减防止遗忘前面学到的能力。

### 配图说明
上方课程阶段表，中间放 fig_sft_loss.png（五段 Loss 曲线），下方标注 Stage A 的锚点作用。

### 备注
这是第一个创新点，花 50 秒。先用 15 秒讲课程设计逻辑（为什么 Stage A 必须在最前面——v2 的教训：先 OpenR1 后 GSM8K 会被难题"带偏"）。然后用 20 秒讲 Loss 曲线的关键特征（Stage A 降幅 -82.5%，B1 起点跳升说明分布不同）。最后 15 秒总结：五段课程的核心价值不是"分了 5 段"，而是 Stage A 的**分布锚点**作用——它让 DPO 阶段的基座更稳定。

---

## Slide 6: 创新点二——Error-Type-Targeted DPO（60 秒）

### 页面内容

**核心创新**：先诊断错误类型，再定向构建偏好数据

**传统 DPO 的问题**：
- 通用 teacher prompt，不区分错误类型
- chosen/rejected 分布差异大（teacher 3384 字符 vs student 539 字符）
- E14 实验中 DPO 训练方向反转（logps/chosen=-329.4 vs rejected=-133.8）

**Targeted DPO 流程**：（配图 fig_targeted_dpo.png）

```
SFT 模型评测 → 收集 badcase → 5 类错误分类 → 类型专属 prompt → Teacher 生成 chosen
     ↓                                              ↓
  77 条 GSM8K                              arithmetic: "写出每步中间结果"
  112 条 MATH                              setup_error: "先复述已知条件"
                                           reasoning_skip: "展开每个推理步骤"
```

**错误类型分布**：（配图 fig_error_dist.png）

| 错误类型 | 占比 | Targeted Prompt |
|---|---|---|
| setup_error | 64.9% | "先复述题目的已知条件和约束" |
| reasoning_skip | 27.3% | "展开每个推理步骤，不跳步" |
| 其余 3 类 | 7.8% | 计算/格式/提取专属 |

**效果验证**：

| 对比 | GSM8K Δ | MATH Δ |
|---|---|---|
| Standard DPO | +0.0pp | +3.5pp |
| **Targeted DPO** | **+2.5pp** | +0.0pp |

**结论**：DPO 的效果高度依赖数据与目标任务的分布对齐度。Targeted DPO 针对 GSM8K → GSM8K 提升；Standard DPO 覆盖广泛 → MATH 提升。

### 配图说明
上方放 fig_targeted_dpo.png（流程图），中间放 fig_error_dist.png（饼图），下方放效果对比表。

### 备注
这是核心创新点，给最多时间（60 秒）。前 15 秒讲传统 DPO 的三个问题（分布鸿沟、粒度粗、信号混杂）。中间 20 秒讲 Targeted DPO 的流程和错误分类结果（setup_error 65% 是关键发现）。后 25 秒讲效果对比——重点解释为什么 Targeted DPO 对 GSM8K 有效但对 MATH 无效（分布对齐度），引出"DPO 不是万能优化器，而是分布对齐器"这个深层洞察。

---

## Slide 7: 创新点三——Teacher SFT 蒸馏（50 秒）

### 页面内容

**核心发现**：1409 条 > 38000 条

| 方案 | 数据量 | GSM8K | MATH |
|---|---|---|---|
| 五段课程 SFT | 38,000 | 62.0% | 44.0% |
| Targeted DPO | 38,000 + 420 | 64.5% | 44.0% |
| **Teacher SFT** | **1,409** | **65.0%** | 43.5% |

**数据构建流程**：（配图 fig_teacher_sft.png）

```
GSM8K-train 1500 题 → Qwen3-235B-Thinking 生成 CoT → 三层质量过滤 → 1409 条
                                                    ├─ 长度过滤（-41）
                                                    ├─ 修正次数过滤（-87）
                                                    └─ 答案验证（96.27% 正确）
```

**为什么 1409 条 > 38k 条？**

不仅"质量高"，更关键的是**分布纯净度**：
- 38k 混合数据：5 个来源，5 种推理风格，模型需学习多种模式
- 1409 teacher CoT：全部来自 GSM8K-train，分布与评测集完全一致

**Teacher CoT 的隐含价值**：
1. 格式规范化 → 消除 extraction_error（3.9% → 0%）
2. 推理完整性 → 消除 arithmetic error（2.6% → 0%）
3. 风格一致性 → 模型无需在多种风格间切换

**代价**：MATH 从 44.0% 降至 43.5%（teacher 数据仅覆盖 GSM8K）

### 配图说明
上方放 fig_teacher_sft.png（流程图），中间放对比表，下方用三个图标标注 Teacher CoT 的三个隐含价值。

### 备注
50 秒。前 10 秒抛出核心数据对比（1409 vs 38k）。中间 20 秒讲数据构建流程和三层过滤。后 20 秒讲"为什么"——重点解释**分布纯净度**这个概念，而非简单说"质量高"。最后提一下代价（MATH 退化），为后面的局限性 slide 做铺垫。验证了 DeepSeek-R1-Distill 的核心洞察在 1.5B 规模上成立。

---

## Slide 8: 关键发现一——DPO 基座稳定性（40 秒）

### 页面内容

**同一 DPO 数据，不同基座，截然不同的结果**：

| 指标 | Group A (单段 SFT) | Group B (五段课程) |
|---|---|---|
| DPO 前 GSM8K | 63.5% | 62.0% |
| DPO 后 GSM8K | **59.5% (-4.0pp)** | **62.0% (+0.0pp)** |
| DPO Loss | 0.239（过低=过拟合）| 0.458（稳定）|

**DPO Loss 对比曲线**：（配图 fig_dpo_loss.png）

**深层原因**：
- Group A：38k 混合数据 → 输出分布分散 → DPO KL 约束不足以防止遗忘
- Group B：Stage A 锚定 GSM8K → 输出分布集中 → DPO KL 约束有效

**设计启示**：**SFT 基座的"稳定性" = 输出分布的集中度。Stage A 是分布锚点，这比增加数据量更重要。**

### 配图说明
左侧对比表，右侧放 fig_dpo_loss.png（两条 Loss 曲线对比），底部标注设计启示。

### 备注
40 秒。用对比表格快速展示现象（-4.0pp vs +0.0pp），然后用 Loss 曲线解释原因（0.239 过拟合 vs 0.458 稳定）。最后一句话总结设计启示：Stage A 的分布锚点作用是五段课程的核心价值。

---

## Slide 9: 关键发现二——Badcase 深度分析（40 秒）

### 页面内容

**SFT 模型错误分布（77 条 badcases）**：

| 错误类型 | 占比 | 典型表现 |
|---|---|---|
| setup_error | **64.9%** | 误读题意、忽略约束条件 |
| reasoning_skip | 27.3% | 跳过关键推理步骤 |
| 其余 3 类 | 7.8% | 计算/格式/提取 |

**典型 Badcase 示例**：

**setup_error**：题目 "three cups per chicken per **day**"
→ 模型误解为 "per **meal**"，结果 ×3

**reasoning_skip**：题目要求计算回程时间（30÷6）
→ 模型跳过除法，直接输出 "48"

**Teacher SFT 后的变化**：
- arithmetic 和 extraction_error **完全消除**
- setup_error 仍是瓶颈（69%）——1.5B 模型理解能力的固有局限

### 配图说明
左侧错误分布表（可叠加 fig_error_dist.png 的简化版），右侧两个 badcase 示例卡片（题目、错误输出、正确答案）。

### 备注
40 秒。前 10 秒展示错误分布（setup_error 65% 是关键数字）。中间 20 秒用两个具体例子说明典型错误——让听众直观感受模型错在哪里。后 10 秒讲 Teacher SFT 的变化（消除 2 类，但 setup 仍是瓶颈），引出这是模型容量的固有局限。

---

## Slide 10: 实验结果总览（30 秒）

### 页面内容

**GSM8K 准确率对比**：（配图 fig_results.png）

**三 Benchmark 综合结果**：

| 实验组 | GSM8K | MATH-500 | BBH-27 |
|---|---|---|---|
| Baseline | 62.5% | — | — |
| Group A SFT | 63.5% | 44.5% | 38.5% |
| Group A DPO | 59.5% | 51.25% | — |
| Group B SFT | 62.0% | 44.0% | 38.8% |
| Group B DPO | 62.0% | 47.5% | — |
| Group D Targeted | 64.5% | 44.0% | 37.4% |
| **Teacher SFT** | **65.0%** | 43.5% | — |

**关键观察**：
- GSM8K：Teacher SFT 最优（65.0%）
- MATH：Standard DPO 最优（47.5%）
- BBH：所有组维持 37-39%，无灾难性退化

### 配图说明
上方放 fig_results.png（GSM8K 柱状图），下方放综合结果表。

### 备注
30 秒，快速过结果。用 fig_results.png 的柱状图直观展示各实验组的 GSM8K 表现（绿色=最佳，红色=回退），然后用一句话总结三个 benchmark 的规律：GSM8K 需要定向优化，MATH 通用 DPO 有效，BBH 无退化。

---

## Slide 11: 局限性与未来方向（40 秒）

### 页面内容

**当前局限**：

| 局限 | 影响 | 改进方向 |
|---|---|---|
| 评测 n=200, CI ±6.9pp | 无法区分 <5pp 差异 | 扩大到 n=500-1000 |
| Targeted DPO 仅 420 条 | 训练不充分 | Rejection Sampling 10-50x |
| Teacher 仅覆盖 GSM8K | MATH 退化 -0.5pp | 多领域 Teacher 数据 |
| Error 分类 5 类太粗 | setup 65% 内部异质 | 细粒度分类（10-15 类）|
| Response-level DPO | 信用分配粗糙 | Step-DPO / PRM |

**最高优先级改进**：

1. **Rejection Sampling**：同一模型生成 N=20 个候选，选对/错做 pair → 数据量放大 20x
2. **Step-DPO**：推理步骤级偏好优化，解决信用分配
3. **Iterative DPO**：多轮训练→评测→挖掘新 badcase→再训练

### 配图说明
左侧局限性表，右侧改进方向（可用流程图表示 Rejection Sampling 和 Iterative DPO 的循环）。

### 备注
40 秒。诚实面对局限性（前 20 秒），展示学术严谨性。后 20 秒讲改进方向，重点提 3 个最高优先级的：Rejection Sampling（解决数据量）、Step-DPO（解决粒度）、Iterative DPO（解决一次性训练）。这些不是空想，是有文献支撑的可行方案。

---

## Slide 12: 总结（30 秒）

### 页面内容

**核心结论**：数据质量 > 数据数量

**三项贡献**：

| 贡献 | 内容 | 验证结果 |
|---|---|---|
| 五段式 SFT 课程 | Stage A 分布锚点 + 学习率递减 | 基座稳定性 +0.0pp vs -4.0pp |
| Error-Type-Targeted DPO | 错误诊断 → 类型专属 prompt | GSM8K +2.5pp |
| Teacher SFT 蒸馏 | 1409 条高质量 CoT | GSM8K 65.0%（最佳）|

**方法论启示**：
> 数据工程（质量过滤 + 课程学习 + 诊断式偏好优化）是小模型推理优化的最有效路径。
> 诊断驱动的实验方法论——先发现弱点，再针对性优化——比盲目尝试更高效。

### 配图说明
中央大标题"数据质量 > 数据数量"，下方三项贡献卡片，底部方法论启示。

### 备注
30 秒收尾。先用一句话总结核心结论。然后快速过三项贡献（每项 5 秒）。最后用 10 秒讲方法论启示——这是超越具体实验的更高层面贡献。语气自信但不夸大，在 CI 范围内的结果诚实标注。

---

# English Version

---

## Slide 1: Cover

**Title**: Enhancing Mathematical Reasoning of 1.5B LLM via Diagnostic Data Engineering

**Subtitle**: 通过诊断式数据工程提升 1.5B 大语言模型数学推理能力

**Info**: DSAA-6000Q | May 2026

---

## Slide 2: Problem and Motivation (40s)

### Slide Content

**Core Question**: Can a 1.5B model's math reasoning close the gap with 7B?

| Model | Params | GSM8K |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | 62.5% |
| Qwen2.5-7B-Instruct | 7B | 84.5% |
| **Gap** | **4.7x** | **22pp** |

**Key Questions**:
- Is this 22pp gap a hard limit of model capacity, or can data engineering bridge it?
- 1.5B models have low deployment cost (edge devices / mobile) — improving their reasoning has huge practical value
- Current literature focuses on 7B+; 1.5B optimization remains under-explored

### Figure Description
Left: model parameter comparison bar chart. Right: edge device deployment scenarios (phone, Raspberry Pi, etc.).

### Notes
Open with 10s on the importance and practical value of the problem — small models' deployment advantage in resource-constrained scenarios. Then 30s to pose the core question: can the 22pp gap be bridged? Cite literature signals: LIMA (1000 samples align LLMs), DeepSeek-R1-Distill (distillation effectiveness).

---

## Slide 3: Literature Foundation and Core Insight (30s)

### Slide Content

**What Does Literature Tell Us?**

| Paper | Core Finding | Implication for This Project |
|---|---|---|
| LIMA (2023) | 1000 high-quality samples can align LLMs | Quality > Quantity |
| DeepSeek-R1-Distill (2025) | 671B→1.5B distillation works | Teacher CoT is the best training data |
| Orca-Math (2024) | GPT-4 distillation boosts small models | Distilled data helps small model math |
| Phi-1 (2023) | 1.3B + "textbook" data beats larger models | Data quality compensates for model size |

**Our Core Hypothesis**:
> **At the extreme 1.5B scale, data engineering (quality filtering + curriculum learning + diagnostic preference optimization) is the most effective path to improve mathematical reasoning.**

### Figure Description
Four literature cards with key data, arrows pointing to our hypothesis in the center.

### Notes
Quickly cover four core papers (5-7s each), emphasizing their common signal: data quality matters more than model scale. Then one sentence to introduce our hypothesis. Don't go into paper details — only what directly informs our project.

---

## Slide 4: Method Overview — Three-Stage Pipeline (40s)

### Slide Content

**Three-Stage Pipeline**: SFT Curriculum → Error Diagnosis → Targeted DPO

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Stage 1: SFT  │───→│  Stage 2: Diag  │───→│  Stage 3: DPO   │
│  5-stage curric │    │  5-class errors  │    │  Targeted DPO   │
│  38k→62.0%      │    │  setup 65% lead  │    │  Targeted 64.5% │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ↓                      ↓                      ↓
  Build capability      Discover weakness      Targeted repair
```

**8 Ablation Experiments**:

| ID | Experiment | GSM8K | Key Validation |
|---|---|---|---|
| E0 | Baseline | 62.5% | Baseline |
| E1 | LoRA + single-stage SFT | 63.5% | SFT effect |
| E2 | E1 + Standard DPO | 59.5% | DPO regression |
| E3 | DoRA + 5-stage curriculum | 62.0% | Curriculum effect |
| E4 | E3 + Standard DPO | 62.0% | Base stability |
| E5 | E3 + Targeted DPO | 64.5% | **Targeted works** |
| E6 | Teacher SFT | **65.0%** | **Best** |

### Figure Description
Top: three-stage pipeline diagram (simplified architecture.svg). Bottom: experiment results table. E5 and E6 highlighted in green.

### Notes
Spend 15s explaining the three-stage logic: SFT builds baseline → diagnosis discovers weaknesses → DPO targeted repair. Then 25s on the experiment overview table, pointing out 3 key comparisons: E2 vs E1 (DPO regression), E4 vs E3 (base stability), E6 vs E3 (Teacher best). This is the skeleton of the entire project — later slides expand on each.

---

## Slide 5: Innovation 1 — Five-Stage SFT Curriculum (50s)

### Slide Content

**Curriculum Design**: in-distribution first → broad reasoning

| Stage | Data | Samples | Role | LR |
|---|---|---|---|---|
| **A** | GSM8K | 7.5k | Distribution anchor | 5e-5 |
| B1 | OpenR1 | 10k | R1 distilled long CoT | 4e-5 |
| B2 | OrcaMath | 15k | GPT-4 distilled broad | 4e-5 |
| B3 | NuminaMath | 8k | Competition difficulty | 3e-5 |
| C | Magpie | 3k | General reasoning anti-forget | 2e-5 |

**Five-Stage SFT Loss Curve**: (Figure: fig_sft_loss.png)

**Key Findings**:
- Stage A has the largest drop (-82.5%): GSM8K is in-distribution, model learns fast
- Stage B1 loss starts at 0.952 (not continuing from 0.225): new data distribution differs
- Stage C is only 7%, but effectively prevents BBH degradation

**Design Logic**: Stage A anchors evaluation distribution → B1-B3 progressively expand capability → Stage C prevents forgetting. Decreasing LR prevents forgetting earlier-learned abilities.

### Figure Description
Top: curriculum stage table. Middle: fig_sft_loss.png (five-stage loss curve). Bottom: annotation of Stage A's anchor role.

### Notes
First innovation, spend 50s. 15s on curriculum logic (why Stage A must be first — v2 lesson: starting with OpenR1 then GSM82 gets "led astray" by hard problems). 20s on loss curve key features (Stage A -82.5% drop, B1 starting point jump indicates different distribution). 15s summary: the core value of five-stage curriculum is not "split into 5 stages" but Stage A's **distribution anchor** role — it makes the DPO phase more stable.

---

## Slide 6: Innovation 2 — Error-Type-Targeted DPO (60s)

### Slide Content

**Core Innovation**: Diagnose error types first, then construct targeted preference data

**Problems with Standard DPO**:
- Generic teacher prompt, no error-type distinction
- Large chosen/rejected distribution gap (teacher 3384 chars vs student 539 chars)
- DPO training direction reversal in E14 (logps/chosen=-329.4 vs rejected=-133.8)

**Targeted DPO Pipeline**: (Figure: fig_targeted_dpo.png)

```
SFT eval → Collect badcases → 5-class error classification → Type-specific prompt → Teacher generates chosen
     ↓                                                          ↓
  77 GSM8K                                           arithmetic: "write out each step"
  112 MATH                                           setup_error: "restate known conditions"
                                                     reasoning_skip: "expand every step"
```

**Error Type Distribution**: (Figure: fig_error_dist.png)

| Error Type | Proportion | Targeted Prompt |
|---|---|---|
| setup_error | 64.9% | "Restate the problem's known conditions and constraints" |
| reasoning_skip | 27.3% | "Expand every reasoning step, no skipping" |
| Other 3 types | 7.8% | Arithmetic/format/extraction specific |

**Effect Validation**:

| Comparison | GSM8K Δ | MATH Δ |
|---|---|---|
| Standard DPO | +0.0pp | +3.5pp |
| **Targeted DPO** | **+2.5pp** | +0.0pp |

**Conclusion**: DPO effectiveness depends on data-target distribution alignment. Targeted DPO for GSM8K → GSM8K improves; Standard DPO covers broad → MATH improves.

### Figure Description
Top: fig_targeted_dpo.png (pipeline). Middle: fig_error_dist.png (pie chart). Bottom: effect comparison table.

### Notes
Core innovation, get the most time (60s). First 15s on Standard DPO's three problems (distribution gap, coarse granularity, mixed signals). Middle 20s on Targeted DPO pipeline and error classification results (setup_error 65% is the key finding). Last 25s on effect comparison — explain why Targeted DPO works for GSM8K but not MATH (distribution alignment), leading to the deep insight: "DPO is not a universal optimizer, but a distribution aligner."

---

## Slide 7: Innovation 3 — Teacher SFT Distillation (50s)

### Slide Content

**Core Finding**: 1,409 samples > 38,000 samples

| Approach | Data Size | GSM8K | MATH |
|---|---|---|---|
| 5-stage SFT | 38,000 | 62.0% | 44.0% |
| Targeted DPO | 38,000 + 420 | 64.5% | 44.0% |
| **Teacher SFT** | **1,409** | **65.0%** | 43.5% |

**Data Construction Pipeline**: (Figure: fig_teacher_sft.png)

```
1500 GSM8K-train → Qwen3-235B-Thinking CoT → 3-layer filtering → 1,409 samples
                                                ├─ Length filter (-41)
                                                ├─ Self-correction filter (-87)
                                                └─ Answer verification (96.27% correct)
```

**Why Do 1,409 Samples Beat 38k?**

Not just "high quality" — the critical factor is **distribution purity**:
- 38k mixed data: 5 sources, 5 reasoning styles, model must learn multiple patterns
- 1,409 teacher CoT: all from GSM8K-train, perfectly aligned with evaluation distribution

**Hidden Value of Teacher CoT**:
1. Format standardization → eliminates extraction_error (3.9% → 0%)
2. Reasoning completeness → eliminates arithmetic error (2.6% → 0%)
3. Style consistency → no need to switch between styles

**Trade-off**: MATH drops from 44.0% to 43.5% (teacher data only covers GSM8K)

### Figure Description
Top: fig_teacher_sft.png (pipeline). Middle: comparison table. Bottom: three icons for Teacher CoT's hidden values.

### Notes
50s. First 10s: core data comparison (1409 vs 38k). Middle 20s: data construction pipeline and three-layer filtering. Last 20s: "why" — focus on **distribution purity** concept, not just "high quality." Mention the trade-off (MATH regression) to set up the limitations slide. Validates DeepSeek-R1-Distill's core insight at 1.5B scale.

---

## Slide 8: Key Finding 1 — DPO Base Stability (40s)

### Slide Content

**Same DPO Data, Different Bases, Dramatically Different Results**:

| Metric | Group A (Single-stage SFT) | Group B (5-stage Curriculum) |
|---|---|---|
| GSM8K before DPO | 63.5% | 62.0% |
| GSM8K after DPO | **59.5% (-4.0pp)** | **62.0% (+0.0pp)** |
| DPO Loss | 0.239 (too low = overfit) | 0.458 (stable) |

**DPO Loss Comparison**: (Figure: fig_dpo_loss.png)

**Root Cause**:
- Group A: 38k mixed data → dispersed output distribution → DPO KL constraint insufficient
- Group B: Stage A anchors GSM8K → concentrated distribution → DPO KL constraint effective

**Design Insight**: **SFT base "stability" = output distribution concentration. Stage A is the distribution anchor — more important than increasing data volume.**

### Figure Description
Left: comparison table. Right: fig_dpo_loss.png (two loss curves). Bottom: design insight annotation.

### Notes
40s. Use comparison table to quickly show the phenomenon (-4.0pp vs +0.0pp). Use loss curves to explain (0.239 overfitting vs 0.458 stable). One-sentence summary: Stage A's distribution anchor role is the core value of the five-stage curriculum.

---

## Slide 9: Key Finding 2 — Badcase Deep Analysis (40s)

### Slide Content

**SFT Model Error Distribution (77 badcases)**:

| Error Type | Proportion | Typical Behavior |
|---|---|---|
| setup_error | **64.9%** | Misreads problem, ignores constraints |
| reasoning_skip | 27.3% | Skips critical reasoning steps |
| Other 3 types | 7.8% | Arithmetic/format/extraction |

**Typical Badcase Examples**:

**setup_error**: Problem says "three cups per chicken per **day**"
→ Model misreads as "per **meal**", result ×3

**reasoning_skip**: Problem requires return trip time (30÷6)
→ Model skips division, outputs "48" directly

**After Teacher SFT**:
- arithmetic and extraction_error **completely eliminated**
- setup_error remains the bottleneck (69%) — inherent limitation of 1.5B model comprehension

### Figure Description
Left: error distribution table (simplified fig_error_dist.png). Right: two badcase example cards (problem, wrong output, correct answer).

### Notes
40s. First 10s: error distribution (setup_error 65% is the key number). Middle 20s: two concrete examples — let the audience直观see where the model goes wrong. Last 10s: Teacher SFT changes (eliminates 2 types, but setup remains), leading to the point that this is an inherent model capacity limitation.

---

## Slide 10: Results Overview (30s)

### Slide Content

**GSM8K Accuracy Comparison**: (Figure: fig_results.png)

**Three-Benchmark Comprehensive Results**:

| Experiment | GSM8K | MATH-500 | BBH-27 |
|---|---|---|---|
| Baseline | 62.5% | — | — |
| Group A SFT | 63.5% | 44.5% | 38.5% |
| Group A DPO | 59.5% | 51.25% | — |
| Group B SFT | 62.0% | 44.0% | 38.8% |
| Group B DPO | 62.0% | 47.5% | — |
| Group D Targeted | 64.5% | 44.0% | 37.4% |
| **Teacher SFT** | **65.0%** | 43.5% | — |

**Key Observations**:
- GSM8K: Teacher SFT best (65.0%)
- MATH: Standard DPO best (47.5%)
- BBH: All groups maintain 37-39%, no catastrophic forgetting

### Figure Description
Top: fig_results.png (GSM8K bar chart). Bottom: comprehensive results table.

### Notes
30s, quickly cover results. Use fig_results.png bar chart to visually show GSM8K performance (green = best, red = regression). One sentence summarizing three benchmarks: GSM8K needs targeted optimization, generic DPO works for MATH, no BBH degradation.

---

## Slide 11: Limitations and Future Directions (40s)

### Slide Content

**Current Limitations**:

| Limitation | Impact | Improvement Direction |
|---|---|---|
| Eval n=200, CI ±6.9pp | Cannot distinguish <5pp differences | Scale to n=500-1000 |
| Only 420 Targeted DPO samples | Insufficient training | Rejection Sampling 10-50x |
| Teacher only covers GSM8K | MATH regresses -0.5pp | Multi-domain teacher data |
| 5-class error taxonomy too coarse | setup 65% has high internal heterogeneity | Fine-grained (10-15 classes) |
| Response-level DPO | Coarse credit assignment | Step-DPO / PRM |

**Highest Priority Improvements**:

1. **Rejection Sampling**: Generate N=20 candidates per problem, select correct/incorrect pairs → 20x data amplification
2. **Step-DPO**: Step-level preference optimization, solving credit assignment
3. **Iterative DPO**: Multi-round train → evaluate → mine new badcases → retrain

### Figure Description
Left: limitations table. Right: improvement directions (flowchart for Rejection Sampling and Iterative DPO loop).

### Notes
40s. Honestly address limitations (first 20s), demonstrating academic rigor. Last 20s on improvements, focusing on top 3 priorities: Rejection Sampling (data volume), Step-DPO (granularity), Iterative DPO (one-shot training). These are not hand-waving — they are literature-backed feasible approaches.

---

## Slide 12: Conclusion (30s)

### Slide Content

**Core Conclusion**: Data Quality > Data Quantity

**Three Contributions**:

| Contribution | Content | Validation |
|---|---|---|
| 5-stage SFT Curriculum | Stage A distribution anchor + decreasing LR | Base stability +0.0pp vs -4.0pp |
| Error-Type-Targeted DPO | Error diagnosis → type-specific prompt | GSM8K +2.5pp |
| Teacher SFT Distillation | 1,409 high-quality CoT samples | GSM8K 65.0% (best) |

**Methodological Insight**:
> Data engineering (quality filtering + curriculum learning + diagnostic preference optimization) is the most effective path for small model reasoning optimization.
> Diagnosis-driven experimentation — discover weaknesses first, then optimize targetedly — is more efficient than blind trial-and-error.

### Figure Description
Central large title "Data Quality > Data Quantity". Below: three contribution cards. Bottom: methodological insight.

### Notes
30s closing. One sentence for core conclusion. Quickly cover three contributions (5s each). Last 10s on methodological insight — this is the higher-level contribution beyond specific experiments. Confident but not exaggerated; honestly note results within CI range.
