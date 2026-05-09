# 项目总结

## 一句话总结

在 Qwen2.5-1.5B-Instruct 上，通过五段式 SFT 课程 + Error-Type-Targeted DPO + Teacher 蒸馏，将 GSM8K 从 62.5% 提升至 65.0%。核心发现：**数据质量 > 数据数量**——1409 条高质量 teacher CoT 超越 38k 混合数据。

---

## 项目价值

### 1. 方法论贡献

- **五段式 SFT 课程设计**：从 in-distribution 到 broad reasoning 的渐进式训练，配合学习率递减，有效平衡任务对齐与能力泛化
- **Error-Type-Targeted DPO**：先诊断错误类型（5 类分类），再用类型专属 prompt 生成纠正样本，比通用 DPO 更有针对性
- **Teacher SFT 蒸馏验证**：用实验证实 DeepSeek-R1-Distill 的核心洞察在 1.5B 规模上成立

### 2. 工程实践

- 完整的两层流水线（CPU 数据准备 + GPU 训练评测）
- Unsloth 2x 训练加速 + 4-bit 量化
- 断点续跑 + NF4 检测 + 自动化评测管线
- 两次训练完全可复现（seed=42, loss 偏差 <0.02）

### 3. 迭代过程

从 v1 的 DPO 灾难性回退（-10pp）到 v4 的 Teacher SFT 最高成绩（65.0%），经历了 4 个大版本的迭代。每次失败都产生了有价值的教训：

| 版本 | 教训 |
|---|---|
| v1 | DPO 数据质量是前提，超参需要仔细调优 |
| v2 | 训练-测试分布错位会抵消课程学习的收益 |
| v3 | 评测管线的每个环节都可能引入隐性 bug |
| v4 | 数据质量远比数据数量重要 |

---

## 关键数据

| 指标 | 值 |
|---|---|
| 最佳 GSM8K | 65.0%（Teacher SFT, 1409 条）|
| 最佳 MATH | 47.5%（Group B DPO）|
| BBH 无退化 | 37-39%（所有组）|
| SFT 训练数据 | ~38k（五源混合）|
| DPO 训练数据 | 420-5000 条（视方案）|
| 可训练参数 | 18.4M（1.18% of 1.56B）|
| 训练时间 | SFT ~3.9k steps, DPO ~600 steps |
| 评测样本量 | GSM8K/MATH 各 200, BBH 780 |

---

## 文档索引

| 文件 | 内容 |
|---|---|
| [01_design.md](01_design.md) | 整体设计方案、架构、技术选型 |
| [02_experiments.md](02_experiments.md) | 全部实验设计、参数配置、评测结果 |
| [03_analysis.md](03_analysis.md) | Badcase 分析、关键发现、改进方向 |
| [04_references.md](04_references.md) | 参考文献（25 篇核心论文）|
| [05_report.md](05_report.md) | 汇报材料（PPT 大纲 + 备注）|
| [architecture.svg](architecture.svg) | 系统架构图 |
| [fig_results.png](fig_results.png) | GSM8K 准确率对比 |
| [fig_sft_loss.png](fig_sft_loss.png) | 五段 SFT Loss 曲线 |
| [fig_dpo_loss.png](fig_dpo_loss.png) | DPO Loss 对比 |
| [fig_lora_dora.png](fig_lora_dora.png) | LoRA vs DoRA 对比 |
| [fig_teacher_sft.png](fig_teacher_sft.png) | Teacher SFT 流程 |
| [fig_targeted_dpo.png](fig_targeted_dpo.png) | Targeted DPO 流程 |
| [fig_error_dist.png](fig_error_dist.png) | 错误类型分布 |
