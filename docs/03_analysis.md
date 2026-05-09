# 实验分析

## 1. Badcase 深度分析

### 1.1 SFT 模型错误分布（GSM8K, E3: 77 条 badcases）

| 错误类型 | 数量 | 占比 | 含义 |
|---|---|---|---|
| setup_error | 50 | 64.9% | 读错题意、列错方程 |
| reasoning_skip | 21 | 27.3% | 跳步、缺少中间推理 |
| extraction_error | 3 | 3.9% | 推理正确但答案提取错误 |
| arithmetic | 2 | 2.6% | 计算错误 |
| unit_or_format | 1 | 1.3% | 单位/格式错误 |

**关键发现**：setup_error 占近 2/3，说明 1.5B 模型的主要瓶颈是**理解题意**，而非计算或推理能力。

### 1.2 典型 Badcase 示例

#### 典型错误 1：setup_error — 误读题意条件

**题目**（GSM8K）：
> Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of 15 cups. In the afternoon, she gives 25 cups. How many cups does she need for the final meal?

**标准答案**：20 杯（每只鸡每天 3 杯 × 20 只 = 60 杯，60 - 15 - 25 = 20）

**模型错误输出**（E3 SFT）：
> 每只鸡每天需要 3 × 3 = 9 杯……总计 20 只鸡需要 180 杯……

**错误根因**：模型将 "three cups per chicken per day" 误解为 "three cups per chicken per meal"，将每日总量乘以 3。这是典型的**条件误读**——模型未能区分 "per day" 和 "per meal" 的语义差异。1.5B 模型的 token-level attention 在处理长句子中的细粒度约束词（every day vs. each meal）时容易丢失信息。

#### 典型错误 2：setup_error — 百分比基数混淆

**题目**（GSM8K）：
> In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop. What percentage enrolled in hip-hop?

**标准答案**：60%

**模型错误输出**（E3 SFT）：
> 步骤 1：contemporary = 20 × 20% = 4 人
> 步骤 2：jazz = 20 × 25% = 5 人
> 步骤 3：hip-hop = 20 - 4 - 5 = 11 人 → 55%

**错误根因**：模型将 "25% of the **remaining**" 中的 "remaining" 忽略，直接对总数 20 取 25%。正确应为 25% × (20-4) = 4 人，hip-hop = 12 人 = 60%。这类错误反映了模型在处理**嵌套约束**（先取子集、再对子集操作）时的能力不足。

#### 典型错误 3：reasoning_skip — 推理链不完整

**题目**（GSM8K）：
> Tom's ship travels at 10 mph from 1 to 4 PM, then returns at 6 mph. How long does it take to get back?

**标准答案**：5 小时（距离 = 10 × 3 = 30 英里，回程 = 30 / 6 = 5 小时）

**模型错误输出**（E3 SFT）：
> 去程距离 = 10 × 3 = 30 英里。回程时间 = 48。  ← 直接给出错误数字

**错误根因**：模型跳过了 "30 / 6 = 5" 这一关键步骤，直接输出一个无意义的数字。reasoning_skip 的本质是模型在 CoT 过程中**丢失了中间状态**，常见于需要多步计算的问题。

#### 典型错误 4：Teacher SFT 仍存在的 setup_misread

**题目**（GSM8K）：
> Gunter asks friends to guess jelly beans. One says 80. Another says 20 more than half the first. A third says 25% more than the first. What is their average?

**标准答案**：80（猜测值：80, 60, 100，平均 = 240/3 = 80）

**模型错误输出**（E6 Teacher SFT）：
> 第二人："20更多于一半" → 80/2 + 20 = **160** ← 应为 60
> 平均 = (80 + 160 + 100) / 3 ≈ 113

**错误根因**：模型将 "20 more than half of 80" 错误解析为 "half of (80+20)" 或类似结构，得到 160 而非 60。**即使经过 Teacher SFT 训练，setup_error 仍是主要瓶颈（69%）**，说明这类错误是 1.5B 模型理解能力的固有局限——teacher CoT 展示了正确解法，但模型未能内化为自己的理解能力。

### 1.3 Teacher SFT 模型错误分布（E6: 70 条 badcases）

| 错误类型 | 数量 | 占比 | 变化趋势 |
|---|---|---|---|
| setup_misread | 48 | 69% | 仍是主要瓶颈 |
| multi_step_cascade | 14 | 20% | 级联错误（前步错→后步全错）|
| truncated | 8 | 11% | 输出被截断（max_tokens 不足）|
| arithmetic | 0 | 0% | **已消除** |
| extraction_error | 0 | 0% | **已消除** |

**对比 E3 的变化**：Teacher SFT 完全消除了 arithmetic 和 extraction_error 两类错误，说明 teacher CoT 的高质量输出格式（规范的 `\boxed{}`、清晰的中间步骤）对这两类错误有直接纠正效果。但 setup_misread 从 64.9% 升至 69%（占比上升），说明读题理解是模型容量问题，非数据质量能解决。

### 1.4 MATH-500 按难度分析

| 难度 | Baseline | SFT | Group A DPO | Group D |
|---|---|---|---|---|
| Level 1 | 27.5% | 27.5% | 27.5% | 27.5% |
| Level 2-4 | — | 错误率上升 | 部分改善 | 无显著变化 |
| Level 5 | 87.5% | — | 82.5% (-5pp) | — |

**发现**：DPO 对高难度题（Level 5）改善最明显，但对低难度题无影响。

### 1.5 BBH 弱项子任务

| 子任务 | 准确率 | 状态 |
|---|---|---|
| word_sorting | 0.0% | 完全失败 |
| web_of_lies | 6.7% | 接近随机 |
| boolean_expressions | 96.7% | 最强 |
| geometric_shapes | — | 加载失败（已知 issue）|

---

## 2. 关键发现与设计洞察

### 2.1 DPO 不稳定性的根源：基座分布集中度

| 指标 | Group A SFT → A DPO | Group B SFT → B DPO |
|---|---|---|
| GSM8K | 63.5% → 59.5% (**-4.0pp**) | 62.0% → 62.0% (**+0.0pp**) |
| DPO Loss | 1.233 → 0.239（-80.6%） | 1.233 → 0.458（-62.9%）|
| Reward Acc | 34.4% → 93% | 34.4% → 92.5% |

**表面现象**：同样是 5000 条 argilla DPO 数据，Group A 回退而 Group B 持平。

**深层原因**：DPO 的 KL 约束（beta=0.1）能否防止遗忘，取决于基座模型的输出分布集中度。Group A（单段 SFT）的输出分布更分散——38k 混合数据包含竞赛题、通用推理、应用题等多种风格，模型没有形成统一的输出模式。DPO 的梯度信号轻易将分布拉偏。

Group B（五段课程）的 Stage A 先用 GSM8K 7.5k 锚定了输出格式和推理风格，后续阶段在此基础上扩展。基座的输出分布更集中（loss 更难下降：0.458 vs 0.239），DPO 的 KL 约束足以防止遗忘。

**设计启示**：**SFT 基座的"稳定性"本质上是输出分布的集中度**。五段课程的 Stage A 起到了"分布锚点"的作用，这比单纯增加数据量更重要。

### 2.2 Targeted DPO 的定向有效性与局限

| 对比 | GSM8K Δ | MATH Δ | Badcase 变化 |
|---|---|---|---|
| B SFT → B DPO (Standard) | +0.0pp | +3.5pp | 77→76 (-1) |
| B SFT → D DPO (Targeted) | +2.5pp | +0.0pp | 77→71 (-6) |

**关键观察**：Standard DPO 在 GSM8K 上无效但在 MATH 上有效；Targeted DPO 恰好相反。这不是巧合——**DPO 的效果高度依赖数据与目标任务的分布对齐度**。

- argilla DPO 数据（5000 条）覆盖广泛数学推理，与 MATH-500 的分布更接近 → MATH +3.5pp
- Targeted DPO 数据（420 条）全部来自 GSM8K badcase，与 GSM8K 完全 in-distribution → GSM8K +2.5pp
- 两者在对方目标上均无提升甚至回退

**设计启示**：**DPO 不是万能优化器，而是分布对齐器**。它的效果取决于偏好数据与目标任务的匹配度。"通用 DPO" 对特定任务的效果有限，"定向 DPO" 对其他任务无效果。这意味着理想的 DPO 策略应该是**多领域定向 DPO**——为每个目标领域构建专属偏好数据。

### 2.3 Teacher SFT：质量 > 数量的深层机制

| 方案 | 数据量 | GSM8K | MATH | BBH |
|---|---|---|---|---|
| Group B SFT（五段课程）| 38k | 62.0% | 44.0% | 38.8% |
| Group D Targeted DPO | 38k + 420 | 64.5% | 44.0% | 37.4% |
| **Teacher SFT** | **1409** | **65.0%** | 43.5% | — |

**为什么 1409 条 > 38k 条？** 不仅仅是"质量高"，更关键的是**分布纯净度**：

- 38k 混合数据包含 5 个来源（GSM8K、OpenR1、OrcaMath、NuminaMath、Magpie），每个来源有不同的推理风格、输出长度、语言偏好。模型需要同时学习多种模式，导致 GSM8K 特定的模式学习不充分。
- 1409 条 teacher CoT 全部来自 GSM8K-train，分布与评测集完全一致。模型只需学习一种模式——GSM8K 的解题风格。

**Teacher CoT 的隐含价值**：
1. **格式规范化**：teacher 输出统一使用 `\boxed{}`，消除了 extraction_error（E3 的 3.9% → E6 的 0%）
2. **推理完整性**：teacher CoT 的每一步都有明确的中间结果，消除了 arithmetic error（E3 的 2.6% → E6 的 0%）
3. **风格一致性**：所有 1409 条数据的推理风格一致，模型无需在多种风格间切换

**代价**：MATH 从 44.0% 降至 43.5%（-0.5pp），因为 teacher 数据完全不覆盖代数、几何、数论等领域。

### 2.4 BBH 无灾难性退化的设计保障

所有实验组在 BBH 上维持 37-39%，说明 Stage C（Magpie 3k，占 7%）有效防止通用推理退化。

**为什么 7% 的通用数据就能防遗忘？** 关键在于 Stage C 的**位置**（在课程最后）和**学习率**（2e-5，最低）。Stage C 的作用不是"教会"模型通用推理，而是在数学训练后"提醒"模型保留通用能力。类似 continual learning 中的 rehearsal 策略——少量旧任务数据防止新任务训练遗忘旧知识。

---

## 3. 方案设计缺陷分析

### 3.1 DPO 数据构建的分布鸿沟

**问题**：本项目 DPO 的 chosen 来自 teacher（Qwen3-235B），rejected 来自 student（1.5B SFT）。两者在语言、风格、推理深度上存在系统性差异，导致 DPO 信号被非推理因素稀释。

**具体表现**：
- Teacher chosen 平均 3384 字符，Student rejected 平均 539 字符——长度差异 6 倍
- Teacher 用英文长 CoT，Student 用中文短输出——语言偏好信号与推理质量信号混杂
- E14 实验中 logps/chosen=-329.4 vs logps/rejected=-133.8，模型认为 chosen 比 rejected 更不可能——DPO 训练方向反转

**根本原因**：DPO 假设 chosen 和 rejected 来自相似分布，仅在质量上有差异。但 teacher-student 范式打破了这一假设。模型学到的不是"如何推理更好"，而是"如何生成更像 teacher 风格的文本"。

**启示**：偏好数据的构建应尽量控制 chosen/rejected 的风格差异，聚焦于推理正确性的差异。Rejection Sampling（同一模型生成多个候选，选对/错的做 pair）是更合理的数据构建方式。

### 3.2 Error Classification 粒度不足

**问题**：5 类错误分类（arithmetic / reasoning_skip / setup_error / unit_or_format / extraction_error）中，setup_error 独占 64.9%，但其内部异质性极高。

**具体分析**：
- "读错一个数字" 和 "完全误解题意" 被归为同一类，但所需的纠正策略完全不同
- reasoning_skip（27.3%）也存在类似问题：跳一步和跳三步的严重程度差异很大
- 5 类分类中 3 类占比 <5%（arithmetic 2.6%, unit_or_format 1.3%, extraction_error 3.9%），数据不足以支撑有效的定向训练

**影响**：Targeted DPO 的类型专属 prompt 只能对 setup_error 做粗粒度纠正，无法区分"读题错误"和"建模错误"，限制了定向优化的效果。

**改进方向**：
- 采用更细粒度的错误分类（如 10-15 类），或用连续分数替代离散分类
- 引入 **Step-Level Error Localization**：不仅分类错误类型，还定位到具体哪一步出错
- 对低频错误类型做数据增强，而非仅依赖自然分布

### 3.3 Response-Level DPO 的信用分配困境

**问题**：当前 DPO 将整个 response 作为 chosen/rejected，无法区分推理链中哪些步骤是关键的。

**具体表现**：
- 一个 2000 字的 CoT 中，可能只有 1 个关键步骤导致错误，但整个 response 被标记为 rejected
- DPO 的梯度信号均匀分布在整个序列上，关键错误步骤的纠正信号被大量正确步骤稀释
- 这解释了为什么 Standard DPO（Group A/B）对 GSM8K 几乎无效（+0.0pp / -4.0pp）

**对比**：MATH 上 DPO 有效（+3.5pp ~ +6.75pp），因为 MATH 题目更难，错误更分散在整个推理链中，response-level 的信号更有效。

**改进方向**：
- **Step-DPO**（Lai et al., 2024）：在推理步骤级别构建偏好对，精确定位错误步骤
- **Process Reward Model (PRM)**：训练逐步奖励模型，为每个推理步骤打分
- **Credit Assignment**：用 teacher 模型标注 student response 中哪些步骤正确/错误，仅对错误步骤构建 DPO pair

### 3.4 Teacher SFT 的 GSM8K-MATH 权衡

**问题**：Teacher SFT 在 GSM8K 上达到最高（65.0%），但在 MATH 上却是最低（43.5%）。

**分析**：
- Teacher 数据（1409 条）全部来自 GSM8K-train，与 GSM8K eval 完全 in-distribution
- MATH-500 涵盖代数、几何、数论等 GSM8K 未覆盖的领域
- Teacher SFT 实质上是一种极端的 in-distribution 对齐，代价是泛化能力

**对比**：
- Group B SFT（38k 混合数据）：GSM8K 62.0%, MATH 44.0%——更均衡
- Group B DPO：GSM8K 62.0%, MATH 47.5%——MATH 最佳
- Teacher SFT：GSM8K 65.0%, MATH 43.5%——GSM8K 最佳但 MATH 最差

**启示**：数据的多样性与专精性存在 trade-off。纯 GSM8K teacher 数据在 GSM8K 上最优，但损害了更广泛数学推理能力。理想方案是 **teacher 数据覆盖多领域**（GSM8K + MATH + 竞赛），而非仅 GSM8K。

### 3.5 Targeted DPO 的数据规模瓶颈

**问题**：Targeted DPO 仅 420 条有效数据（v1），按错误类型分每类仅 ~84 条。

**影响**：
- 420 条数据对应的 DPO 训练仅 ~157 steps，模型难以充分学习
- 对比 Standard DPO（5000 条, 600 steps），Targeted DPO 的训练量不足
- GSM8K 仅 +2.5pp（在 CI 范围内），可能只是数据量不足导致的欠拟合

**根本矛盾**：Targeted DPO 的数据来源是 SFT 模型的 badcase（77 条 GSM8K + 112 条 MATH），受限于评测样本量（n=200）。要获得更多 targeted 数据，需要更大的评测集或更高效的 badcase 挖掘策略。

**改进方向**：
- **Rejection Sampling**：对每道题生成 N 个候选（N=10~50），选错误答案作为 rejected，正确答案作为 chosen，数据量可放大 10-50 倍
- **Iterative DPO**：每轮 DPO 后重新评测，挖掘新的 badcase，逐步扩大 targeted 数据池
- **跨任务迁移**：用 GSM8K 的错误类型知识指导 MATH 的数据构建

---

## 4. 方法论改进方向

### 4.1 从 Response-Level 到 Step-Level（最高优先级）

**现状**：DPO 在整个 response 级别做偏好优化，信用分配粗糙。

**方案**：
- **Step-DPO**：将推理链切分为步骤，对每个错误步骤独立构建偏好对。需要 step-level 标注（可用 teacher 模型自动标注）。
- **Process Reward Model (PRM)**：训练一个逐步打分模型，在推理过程中提供细粒度反馈。Lightman et al. (2023) 证明 PRM 在 MATH 上 best-of-N 达 78.2%，远超 ORM 的 72.4%。
- **预期收益**：解决信用分配问题，DPO 信号不再被正确步骤稀释。

### 4.2 Rejection Sampling 扩大数据规模

**现状**：Targeted DPO 仅 420 条，训练不充分。

**方案**：对每道 GSM8K/MATH 题目，用 SFT 模型生成 N=20 个候选答案。正确答案中选最短的作为 chosen，错误答案中选最像正确的作为 rejected。预期可生成 5000+ 条高质量偏好数据。

**优势**：
- chosen/rejected 来自同一模型，风格一致，避免分布鸿沟
- 数据量可放大 10-50 倍
- 可按错误类型对 rejected 做分类，保留 Targeted DPO 的定向性

### 4.3 Iterative DPO 在线学习

**现状**：DPO 是一次性的（offline），训练后不再更新数据。

**方案**：多轮循环——DPO 训练 → 评测 → 挖掘新 badcase → 构建新 DPO 数据 → 再训练。每轮 badcase 会变化（旧错误被修复，新错误出现），数据池逐步扩大和更新。

**预期收益**：
- 持续挖掘模型的弱点并针对性优化
- 避免一次性 DPO 的数据固化问题
- 类似 RLHF 的在线学习效果，但更稳定

### 4.4 多领域 Teacher 数据扩展

**现状**：Teacher 数据仅来自 GSM8K，导致 MATH 能力退化。

**方案**：
- 扩展 teacher 数据到 MATH-500 训练集、NuminaMath、AMC/AIME 等
- 每个领域用对应的 teacher prompt 生成 CoT
- 保留 GSM8K 的 in-distribution 锚点，同时覆盖更广的数学推理领域

**预期收益**：在保持 GSM8K 优势的同时，恢复或提升 MATH 能力。

### 4.5 细粒度错误分类与专项训练

**现状**：5 类分类太粗，setup_error 占 65% 但内部异质性高。

**方案**：
- 将 setup_error 细分为：数字读取错误、条件遗漏、约束误解、问题类型误判
- 对每种子类型设计专项训练数据（如"先列出所有已知数字"、"复述题目的每个条件"）
- 用更多样的 teacher prompt 变体覆盖错误子类型

### 4.6 评测体系升级

**现状**：n=200, CI ±6.9pp，无法区分 <5pp 的差异。

**方案**：
- 扩大评测集到 n=500（CI ±4.4pp）或 n=1000（CI ±3.1pp）
- 引入 McNemar 检验和 Bootstrap CI
- 增加 per-error-type 评测：不只看总体准确率，还看每类错误的修复率
