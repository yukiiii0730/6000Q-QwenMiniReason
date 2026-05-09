# Qwen-Reasoning-Enhance: 整体设计方案

## 1. 核心发现：文献驱动的方法论基础

### 1.1 小模型数学推理的可行性

Qwen2.5-1.5B-Instruct 在 GSM8K 上的官方成绩为 73.2%（8-shot, lm-eval-harness），但实际 zero-shot 场景下仅约 62.5%。与 7B（84.5%）存在 ~22pp 差距。文献表明，这一差距主要来自**数据质量和训练策略**，而非模型容量本身：

- **LIMA**（Zhou et al., 2023）：仅 1000 条高质量数据即可对齐 LLM，证明数据质量远比数量重要
- **Phi-1 / Textbooks Are All You Need**（Gunasekar et al., 2023）：1.3B 模型用高质量"教科书"数据在代码任务上超越更大模型
- **DeepSeek-R1-Distill**（DeepSeek, 2025）：从 671B 蒸馏到 1.5B-70B，蒸馏模型在 MATH/AIME 上表现优异，验证了 teacher 蒸馏对小模型的有效性
- **Orca-Math**（Microsoft, 2024）：GPT-4 蒸馏数据让小模型在 GSM8K 上大幅超越 baseline

**结论**：1.5B 模型有充足的提升空间，关键在于数据工程。

### 1.2 SFT 阶段的方法选择

文献中提升小模型数学推理的 SFT 策略主要有三种：

**（a）数据质量过滤**
- OpenMathInstruct-2（NVIDIA, 2024）通过答案验证 + 难度过滤构建高质量 SFT 数据
- 本项目采用 SHA-1 跨数据集去重 + 长度过滤（<1024/<2048 双阈值），从原始 ~50k 过滤至 ~38k

**（b）课程学习（Curriculum Learning）**
- 多项 2024 年研究表明，按难度递增安排 SFT 数据顺序可提升最终性能
- Qwen2.5-Math 技术报告中采用渐进式训练策略
- 反直觉发现：部分研究（THUDM）发现 hard-to-easy 有时优于 easy-to-hard
- 本项目选择 **in-distribution first → broad reasoning** 的课程顺序，先锚定评测分布再扩展能力

**（c）Teacher 蒸馏**
- DeepSeek-R1-Distill 核心洞察：大模型的 CoT 推理过程本身就是最好的训练数据
- 关键不是模型大小，而是 CoT 质量——清晰、完整、无错误的推理链
- 本项目用 Qwen3-235B-Thinking 生成 teacher CoT，验证了这一洞察

### 1.3 DPO 阶段的方法选择

DPO（Rafailov et al., 2023）相比 RLHF 更简单稳定，但应用于数学推理时面临独特挑战：

**Standard DPO 的局限**：
- 静态 offline 数据无法适应模型训练过程中的变化
- response-level 粒度导致信用分配困难——一个 2000 字 CoT 中仅 1 步出错，但整个 response 被标为 rejected
- 在小模型上效果不稳定：Group A DPO 回退 -4.0pp

**更有效的 DPO 变体**：
- **Step-DPO**（Lai et al., 2024）：在推理步骤级别构建偏好对，解决信用分配问题
- **Process Reward Model**（Lightman et al., 2023）：逐步奖励优于结果奖励（PRM 78.2% vs ORM 72.4% on MATH）
- **Rejection Sampling**：同一模型生成多个候选，选对/错做 pair，避免 teacher-student 分布鸿沟
- **Iterative/Online DPO**：多轮训练→评测→挖掘新 badcase→再训练

**本项目的创新切入点**：在 response-level DPO 的框架内，引入**错误类型诊断**作为中间环节，用类型专属 prompt 生成更有针对性的 chosen，弥补粒度不足的问题。

---

## 2. 整体思路

### 2.1 三阶段流水线

```
SFT 课程学习 → 错误诊断 → 定向偏好优化
```

基于上述文献分析，本项目设计了三阶段流水线：

1. **SFT 阶段**：五段式课程，从 in-distribution 到 broad reasoning，建立推理基座
2. **诊断阶段**：对 SFT 模型做 badcase 分析，将错误分为 5 类，理解模型的弱点分布
3. **DPO 阶段**：基于错误类型构建定向偏好数据，做 targeted DPO

**设计逻辑**：SFT 建立能力基线 → 诊断发现弱点 → DPO 定向修复。诊断是连接 SFT 和 DPO 的桥梁，使 DPO 从"通用优化"变为"定向修复"。

### 2.2 五段式 SFT 课程设计

| 阶段 | 数据集 | 样本数 | Token 上限 | 角色 | 设计依据 |
|---|---|---|---|---|---|
| **A** | GSM8K-train | 7.5k | 1024 | In-distribution 锚点 | 先对齐评测分布，确保 SFT 不偏离目标 |
| **B1** | OpenR1-Math-220k | 10k | 2048 | R1 蒸馏长 CoT | DeepSeek-R1-Distill 验证了长 CoT 蒸馏的有效性 |
| **B2** | Orca-Math-200k | 15k | 1024 | GPT-4 蒸馏广覆盖 | Orca-Math 证明 GPT-4 蒸馏数据对小模型有效 |
| **B3** | NuminaMath-CoT | 8k | 2048 | 竞赛难题多样性 | 引入更难的推理题，扩展能力上限 |
| **C** | Magpie-Reasoning | 3k | 2048 | 通用推理防遗忘 | ~7% 的通用推理数据防止灾难性退化 |

**课程顺序的设计考量**：
- Stage A 先于 B1-B3：先锚定 GSM8K 分布，再引入更广的数据。如果反过来（先 OpenR1 后 GSM8K），模型可能被 AIME/Olympiad 难题"带偏"，反而损害简单应用题能力——这是 v2 的教训
- B1→B2→B3 的顺序：从 R1 蒸馏（深度）到 GPT-4 蒸馏（广度）到竞赛题（难度），逐步扩展
- Stage C 在最后：通用推理数据占比小（7%），放在最后不影响数学对齐，但能防止 BBH 退化
- 学习率递减（5e-5→2e-5）：课程越往后数据越难，降低学习率防止遗忘前面学到的能力

### 2.3 Error-Type-Targeted DPO 设计

传统 DPO 用通用 teacher prompt 生成 chosen，对所有错误类型一视同仁。本项目的创新是**先诊断再治疗**：

**Step 1 — 错误诊断**：用 Qwen-Flash（低成本 API 模型）对 SFT 模型的 GSM8K badcase 做 5 类错误分类。诊断结果揭示了模型的弱点分布：
- setup_error 64.9%：模型的主要瓶颈是理解题意，而非计算或推理
- reasoning_skip 27.3%：跳步问题次之
- 其余三类 <5%：计算、格式、提取问题相对较少

**Step 2 — 定向数据构建**：针对每类错误，用**类型专属 system prompt** 让 teacher（Qwen3-235B-Thinking）生成 chosen：
- arithmetic → "请将每一步的中间结果明确写出"——针对计算错误，强调中间过程
- setup_error → "先复述题目的已知条件和约束"——针对读题错误，强制先理解再解题
- reasoning_skip → "展开每个推理步骤，不跳步"——针对跳步，要求完整推理链
- extraction_error → "答案单独写在最后一行，后面不加其他内容"——针对提取错误，规范输出格式
- unit_or_format → "标注单位，统一格式"——针对格式问题

**Step 3 — DPO 训练**：用 targeted 数据做 DPO，beta=0.1, lr=1e-5。

**与 Standard DPO 的区别**：Standard DPO 的 chosen 由通用 teacher prompt 生成，不区分错误类型。Targeted DPO 的 chosen 针对具体错误类型做优化，纠正信号更精准。

### 2.4 Teacher SFT 蒸馏设计

独立于 Targeted DPO 的另一条路径：直接用 teacher CoT 做 SFT。

**数据构建流程**：
1. 用 Qwen3-235B-Thinking 为 GSM8K-train 的 1500 道题生成 CoT 解答
2. 三层质量过滤：
   - 长度过滤：剔除 >10k chars 的过长输出（41 条）
   - 修正次数过滤：剔除自我修正 ≥10 次的输出（87 条）
   - 答案正确性验证：与 GSM8K 标准答案比对
3. 最终保留 1409 条（96.27% 正确率）

**设计考量**：
- 为什么选 GSM8K-train 而非 MATH：GSM8K 与评测集完全 in-distribution，最大化 SFT 效果
- 为什么只 1500 条：受 API 成本限制，但结果证明 1409 条已足够
- 为什么用 Thinking 模型：thinking mode 产出的 CoT 更完整、更严谨

---

## 3. 核心创新点

### 3.1 Error-Type-Targeted DPO

**创新本质**：在 SFT 和 DPO 之间插入"错误诊断"环节，将 DPO 从通用优化变为定向修复。

**与现有工作的区别**：
- Standard DPO（argilla/distilabel）：通用偏好数据，不针对具体错误
- Teacher-Guided DPO：通用 teacher prompt，不区分错误类型
- **Error-Type-Targeted DPO**：先分类错误，再用类型专属 prompt 生成 chosen——诊断驱动

**理论依据**：不同类型错误需要不同的纠正策略。计算错误需要强调中间过程，读题错误需要强制先理解再解题。通用 prompt 无法同时覆盖所有错误类型。

### 3.2 五段式课程 SFT

**创新本质**：将单一 SFT 训练拆分为 5 个阶段，每阶段有明确的角色定位和训练策略。

**与现有工作的区别**：
- 传统 SFT：所有数据混合训练，无课程顺序
- 两阶段课程（v2）：仅区分"难"和"易"，粒度不够
- **五段课程**：从 in-distribution → 深度 → 广度 → 难度 → 泛化，每段有独立的学习率和步数

**关键设计**：Stage A 先锚定评测分布（GSM8K），再逐步引入更广数据。这解决了 v2 中"训练-测试分布错位"的问题。

### 3.3 诊断驱动的实验方法论

**创新本质**：不是盲目尝试不同 DPO 变体，而是先诊断模型弱点，再针对性设计实验。

**具体实践**：
1. 对每个 SFT 模型做 badcase 分析（5 类错误分类）
2. 根据错误分布设计 DPO 数据
3. 评测后再次做 badcase 分析，验证错误是否被修复
4. 发现新的瓶颈，指导下一轮实验

**价值**：这种方法论使实验迭代更高效——不是随机尝试，而是基于诊断的定向优化。

---

## 4. 评测体系设计

### 4.1 Benchmark 选取

| Benchmark | 选取原因 | 样本数 | 指标 |
|---|---|---|---|
| **GSM8K** | 小学数学应用题，in-distribution 主评测 | 200 | 准确率 |
| **MATH-500** | 竞赛级数学，5 个难度等级，评测推理深度 | 200 | 准确率 |
| **BBH-27** | 27 个通用推理子任务，检测灾难性遗忘 | 780（30/task）| Macro avg |

**选取考量**：
- GSM8K 是主目标：SFT 数据包含 GSM8K-train，评测 GSM8K 是直接对齐
- MATH-500 是推理标尺：比 GSM8K 难得多（涵盖代数、几何、数论等），能区分不同 SFT 策略的推理深度差异
- BBH-27 是退化检测：数学 SFT 可能损害通用推理能力，BBH 覆盖逻辑、常识、排序等多种推理类型

**为什么不选其他 benchmark**：
- AIME/AMC：太难，1.5B 模型几乎无法作答，无区分度
- ARC/HellaSwag：偏常识，与数学推理关系不大
- HumanEval：代码任务，超出本项目范围

### 4.2 评测协议设计

**协议选择**：chat-template + zero-shot。

**设计考量**：
- zero-shot 更贴近实际部署场景（用户不会给 few-shot 示例）
- 套 chat_template 确保评测格式与训练格式一致（Qwen2.5-Instruct 用 ChatML 格式）
- 与 Qwen 官方 lm-eval-harness（8-shot）存在系统性偏差，但组间相对比较仍然有效

**样本量选择**：n=200。
- 95% CI = ±6.9pp（基于二项分布 p=0.625 计算）
- 能检测 >7pp 的差异，<5pp 的差异无统计显著性
- 权衡：n=500（CI ±4.4pp）更可靠但评测时间翻倍，n=200 在 Colab GPU 时限内可行

**答案提取**：从模型输出中提取最终数字，与标准答案比对。
- GSM8K：提取 `####` 后的数字或最后一个数字
- MATH：提取 `\boxed{}` 中的内容，用 sympy 做符号等价判断
- 归一化处理：strip 空格、去除尾部句点、统一数字格式

### 4.3 Badcase 收集与分析

每次评测后自动收集 badcase（模型答错的题目），保存为 JSONL：
- question: 原始问题
- pred / pred_raw: 模型输出（提取后 / 原始）
- gt / gt_raw: 标准答案
- correct: 是否正确

badcase 是错误诊断和 Targeted DPO 数据构建的源头。

---

## 5. 迭代历程

### 5.1 v1：初版方案（失败）

**方案**：LoRA + 单段 SFT（NuminaMath-CoT）+ Standard DPO（orca-math）

**结果**：GSM8K 50% → 40%（DPO 回退 -10pp）

**设计层面的问题**：
1. **DPO 数据选择错误**：orca-math 没有 chosen/rejected 对，强行构建的偏好数据质量极差
2. **DPO 超参过激**：beta=0.3 过高（过强的 KL 约束），max_seq_length=1536 截断了 CoT 推理链
3. **SFT 数据单一**：仅用 NuminaMath（竞赛题），与 GSM8K（小学应用题）分布严重错位
4. **评测不足**：仅 50 题（CI ±14%），结果几乎无统计意义

**教训**：DPO 的成功前提是——(1) 高质量偏好数据；(2) 合适的超参；(3) SFT 基座稳定。三者缺一不可。

### 5.2 v2：DoRA + 课程 + 修复 DPO

**改进**：
- PEFT 从 LoRA 升级到 DoRA
- SFT 从单段改为两段课程（OpenR1 20k → Magpie 8k）
- DPO 数据改用 argilla/distilabel-math-preference-dpo（5k 有真实 chosen/rejected）
- DPO 超参修复：beta 0.3→0.1, seq_len 1536→2048, lr 3e-6→1e-5
- 评测样本量 50→200

**结果**：~55% GSM8K（显著提升）

**仍存在的设计问题**：
1. **训练-测试分布错位**：OpenR1-Math 主要是 AIME/Olympiad 级别，与 GSM8K（小学应用题）差距过大。模型学到了高难度推理，但简单应用题反而做不好
2. **BBH 评测不完整**：仅评了 1/27 子任务，无法判断是否退化
3. **缺少 MATH 评测**：无法区分不同策略的推理深度差异

### 5.3 v3：in-distribution 对齐 + 错误分类

**改进**：
- SFT 数据策略重构：废弃 OpenR1 作为主力，改用 GSM8K-train + MetaMathQA + Magpie（更 in-distribution）
- BBH 评测完善：27 子任务全量评测
- 评测脚本修复：套 chat_template, max_new_tokens=1024
- **创建错误分类管线**：classify_errors.py + build_targeted_dpo.py

**设计层面的突破**：
- 意识到 in-distribution 数据的重要性——先对齐评测分布，再扩展能力
- 错误分类管线的建立为 Targeted DPO 奠定基础

**仍存在的问题**：
1. 数据组合偏 in-distribution，缺乏推理深度
2. 缺少 MATH 评测
3. 错误分类尚未用于 DPO 数据构建

### 5.4 v4：三剑客课程 + Targeted DPO + Teacher SFT

**改进**：
- SFT 数据策略定型：五段式课程（GSM8K → OpenR1 → OrcaMath → NuminaMath → Magpie），平衡 in-distribution 和推理深度
- 加入 MATH-500 评测
- 实施 Error-Type-Targeted DPO（核心创新）
- 实施 Teacher SFT 蒸馏
- 6 组消融实验系统对比

**关键发现**：
1. **Teacher SFT 最高效**：1409 条 → 65.0%，超越 38k 混合数据（62.0%）
2. **Targeted DPO 定向有效**：GSM8K +2.5pp（针对 GSM8K badcase 设计）
3. **DPO 基座稳定性关键**：五段课程（62.0% 持平）vs 单段 SFT（59.5% 回退 -4.0pp）
4. **MATH 上 DPO 普遍有效**：Standard DPO +3.5pp, Group A DPO +6.75pp
5. **BBH 无灾难性退化**：Stage C（Magpie 3k, 7%）有效防遗忘

**v4 仍存在的设计局限**：
1. Targeted DPO 数据量小（420 条），训练不充分
2. Teacher SFT 仅覆盖 GSM8K，MATH 能力退化
3. Error classification 粒度不够（setup_error 占 65% 但内部异质性高）
4. Response-level DPO 的信用分配问题未解决
5. 评测样本量不足（n=200, CI ±6.9pp）
