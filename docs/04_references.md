# 参考文献

## 核心参考

### 基础模型

1. **Qwen2.5 Technical Report**
   - Qwen Team, Alibaba. 2024.
   - arXiv: 2412.15115
   - 本项目基座模型 Qwen2.5-1.5B-Instruct 的技术报告，涵盖多尺寸模型（0.5B-72B）在数学、代码、推理等任务上的评测。

2. **Qwen3 Technical Report**
   - Qwen Team, Alibaba. 2025.
   - 本项目 teacher 模型 Qwen3-235B-Thinking 的技术报告，支持 thinking mode 的 MoE 架构。

### 蒸馏与 SFT

3. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
   - DeepSeek AI. January 2025.
   - arXiv: 2501.12948
   - **直接相关**：验证了从大模型（671B）蒸馏推理能力到小模型（1.5B-70B）的有效性。本项目的 Teacher SFT 实验验证了同一洞察：1409 条 teacher CoT 超越 38k 混合数据。

4. **Orca-Math: Unlocking the Potential of SLMs in Grade School Math**
   - Microsoft Research. 2024.
   - **直接相关**：本项目 Stage B2 数据来源（Orca-Math-200k），验证了 GPT-4 蒸馏数据对小模型数学能力的提升。

5. **OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data**
   - NVIDIA. 2024.
   - 大规模数学 SFT 数据集，与本项目的数据策略（多源混合 + 质量过滤）理念一致。

6. **OpenR1-Math-220k**
   - Open-R1 Community. 2025.
   - 本项目 Stage B1 数据来源，DeepSeek-R1 蒸馏的高质量长 CoT 数学数据。

### 偏好优化

7. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model**
   - Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. 2023.
   - arXiv: 2305.18290
   - **核心方法**：本项目 DPO 训练的基础算法。

8. **Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning**
   - Xin Lai, Zhuotao Tian, et al. CUHK. June 2024.
   - arXiv: 2406.18629
   - **直接相关**：在推理步骤级别做 DPO，比本项目当前的 response-level DPO 更细粒度，是重要的改进方向。

9. **KTO: Model Alignment as Prospect Theoretic Optimization**
   - Ethayarajh et al. February 2024.
   - arXiv: 2402.01306
   - 基于 Kahneman-Tversky 前景理论的偏好优化，只需二元反馈（好/坏），不需要 paired data。本项目可选的 DPO 变体。

10. **SimPO: Simple Preference Optimization with a Reference-Free Reward**
    - Meng et al. May 2024.
    - arXiv: 2405.14734
    - 去除参考模型的简化 DPO，用平均 log probability 作为隐式奖励。内存效率更高。

11. **IPO: A General Theoretical Paradigm to Understand Learning from Human Feedback**
    - Azar et al. Google DeepMind. 2023.
    - arXiv: 2310.12036
    - Identity-mapping Preference Optimization，本项目 Group E 的可选方案。

### 过程奖励与验证

12. **Let's Verify Step by Step**
    - Hunter Lightman et al. OpenAI. 2023.
    - arXiv: 2305.20050
    - **关键参考**：Process Reward Model (PRM) vs Outcome Reward Model (ORM)，PRM 在 MATH 上 best-of-N 达 78.2%。本项目改进方向之一。

13. **Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations**
    - Wang et al. 2024.
    - arXiv: 2312.08935
    - 自动化过程奖励标注，降低 PRM 的人工成本。

14. **OmegaPRM: Improve Mathematical Reasoning in Language Models by Automated Process Supervision**
    - Luo et al. 2024.
    - arXiv: 2406.04808
    - 自动构建过程奖励模型的方法。

### PEFT 方法

15. **DoRA: Weight-Decomposed Low-Rank Adaptation**
    - Shih-Yang Liu et al. UC Berkeley / NVIDIA. 2024.
    - arXiv: 2402.09353
    - **直接相关**：本项目 SFT 的 PEFT 方法。将权重分解为 magnitude + direction，仅对 direction 做 LoRA，优于标准 LoRA。

16. **LoRA: Low-Rank Adaptation of Large Language Models**
    - Edward J. Hu et al. Microsoft. 2021.
    - arXiv: 2106.09685
    - 本项目 Group A 的 PEFT 方法，DoRA 的基础。

### 数据质量与课程学习

17. **LIMA: Less Is More for Alignment**
    - Chunting Zhou et al. Meta. 2023.
    - arXiv: 2305.11206
    - 1000 条高质量数据即可对齐 LLM，验证了"质量 > 数量"的核心理念。

18. **Textbooks Are All You Need**
    - Suriya Gunasekar et al. Microsoft. 2023.
    - arXiv: 2306.11644
    - Phi-1 用高质量"教科书"数据训练 1.3B 模型在代码任务上超越更大模型。与本项目的 Teacher SFT 理念一致。

19. **NuminaMath-CoT**
    - AI-MO Community. 2024.
    - 本项目 Stage B3 数据来源，竞赛/奥数级别的数学 CoT 数据。

20. **Magpie-Reasoning-150K**
    - Magpie-Align. 2024.
    - 本项目 Stage C 数据来源，通用推理数据，用于防止灾难性遗忘。

### 评测基准

21. **GSM8K: Training Verifiers to Solve Math Word Problems**
    - Karl Cobbe et al. OpenAI. 2021.
    - arXiv: 2110.14168
    - 本项目主要评测基准，小学数学应用题。

22. **Measuring Mathematical Problem Solving With the MATH Dataset**
    - Dan Hendrycks et al. UC Berkeley. 2021.
    - arXiv: 2103.03874
    - 本项目评测基准之一，竞赛级数学题。

23. **BIG-Bench Hard**
    - Suzgun et al. 2022.
    - arXiv: 2210.09261
    - 本项目评测基准之一，27 个困难子任务，用于检测通用推理能力退化。

### 迭代式训练

24. **Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (SPIN)**
    - Zixiang Chen et al. 2024.
    - arXiv: 2401.01335
    - 自我对弈式 DPO 迭代提升，与本项目的 iterative DPO 改进方向相关。

25. **Self-Rewarding Language Models**
    - Weizhe Yuan et al. Meta. 2024.
    - arXiv: 2401.10020
    - 模型自身作为奖励模型进行迭代 DPO 训练。

---

## 延伸阅读

- **Scaling Data-Constrained Language Models** — 数据量 vs 模型规模的 scaling law
- **Orca 2: Teaching Small Language Models How to Reason** — 小模型推理能力训练
- **WizardMath** — 数学 SFT + 进化式数据增强
- **MetaMathQA** — 数学数据增强（本项目 v3 曾使用）
- **Rejection Sampling Fine-Tuning** — 采样多个候选选最优
