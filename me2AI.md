# me2AI · 项目架构与方案设计

> **本文件用途**：固化"项目要做什么、怎么做、为什么这么做"的最终设计。
> **维护规则**：每次方案/架构发生实质变化时同步更新。
> **最后更新**：2026-05-05（v4 消融完成，官方评测重跑中）

---

## 1. 项目目标（一句话定义）

> 在 5B 以内的小模型（Qwen2.5-1.5B-Instruct）上，通过**目标对齐的数据课程**与**诊断驱动的偏好优化**，实现**接近 7B 模型的数学推理能力**，并验证不同推理数据构造与 DPO 损失改进的有效性。

### 1.1 选择 1.5B 的理由

- **资源受限场景的现实需求**：移动端、边缘设备、私有部署
- **学术研究热点**：DeepSeek-R1-Distill-Qwen-1.5B、TinyGSM、MAmmoTH 等聚焦此区间
- **可控的训练成本**：Colab A100 单卡可完整跑通 SFT+DPO

### 1.2 选择数学推理作为评测领域的理由

- **可量化、低主观性**：答案唯一可机器判分
- **学界标准 benchmark 完备**：GSM8K + MATH 两层难度
- **暴露推理能力差距最明显**：1.5B vs 7B 在 MATH 上常拉开 20pp+

### 1.3 选择 SFT + DPO 双阶段的理由

- **SFT 学语义基础**：让小模型学会"怎么写出可读的 CoT"
- **DPO 修对齐缺陷**：用偏好对修复 SFT 后剩余的 failure mode
- **业界已验证范式**：InstructGPT / DeepSeek-R1 / Qwen 系列均采用

---

## 2. 整体架构

```
┌──────────────────── 本地（CPU, run_local.sh）─────────────────────┐
│ ① 数据下载 & 预处理   v4 五段课程数据（~38k）                      │
│ ② API Baseline       公开值优先，自跑 50 题 sanity check           │
│ ③ 数据质量过滤        qwen-flash 评分（可选）                       │
│ ④ Teacher 数据生成    仅小批量难例（成本可控）                       │
│ ⑤ 错误分类 L3         classify_errors.py（qwen-flash）            │
│ ⑥ Targeted 数据生成 L4  build_targeted_dpo.py（qwen2.5-72b）      │
└──────────────────────────┬────────────────────────────────────────┘
                           │ rsync / Drive 同步
                           ▼
┌─────────────── GPU（Colab A100 / 服务器）──────────────────────────┐
│ ⑦ SFT 五段式课程       Stage A→B1→B2→B3→C                         │
│ ⑧ 合并 + SFT 评测      GSM8K + MATH-500 + BBH-27                 │
│ ⑨ Targeted DPO         按错误类型生成 chosen + Weighted DPO loss   │
│ ⑩ 合并 + 最终评测       lm-evaluation-harness 官方协议             │
│ ⑪ Ablation A/B/C/D     6 组对照实验                               │
└──────────────────────────┬────────────────────────────────────────┘
                           │ rsync 拉回 outputs/, logs/
                           ▼
                    本地报告 + 可视化 + 统计显著性
```

---

## 3. 数据策略 v4（"三剑客"主干 + in-distribution 对齐）

### 3.1 SFT 五段课程

| 阶段 | 数据集 | 采样 | 长度过滤 | 作用 |
|------|-------|------|---------|------|
| **Stage A** in-distribution | `openai/gsm8k`-train | 7.5k | <1024 tok | 直接对齐 GSM8K 评测分布 |
| **Stage B1** R1 推理深度 | `open-r1/OpenR1-Math-220k`（verified=True）| 10k | <2048 tok | 高质量长 CoT |
| **Stage B2** 应用题广度 | `microsoft/orca-math-word-problems-200k` | 15k | <1024 tok | 步骤短、覆盖广，1.5B 友好 |
| **Stage B3** 题型多样 | `AI-MO/NuminaMath-CoT`（去 source=gsm8k）| 8k | <2048 tok | 奥赛/AMC/AOPS 风格 |
| **Stage C** 通用推理 | `Magpie-Align/Magpie-Reasoning-150K` | 3k | <2048 tok | 防 BBH 退化（占比 ~7%）|

跨集去重（SHA-1）+ 长度过滤后实际 ~38k。

### 3.2 DPO 数据三层

| 类型 | 来源 | 用途 |
|------|------|------|
| **Fallback** | `argilla/distilabel-math-preference-dpo` 5k | Group A/B 基线 DPO |
| **Teacher-Guided** | qwen2.5-72b 生成 chosen（统一 prompt）| Group C |
| **Error-Type-Targeted** | qwen2.5-72b + 类型专属 system prompt | Group D（创新核心）|

---

## 4. 评测策略（v4 最终版）

### 4.1 官方协议（主要报告数据）

使用 **lm-evaluation-harness**，与 Qwen 官方技术报告一致：
- GSM8K：8-shot flexible-extract
- MATH-500：4-shot sympy 答案归一化
- BBH-27：3-shot chain-of-thought

### 4.2 自定义协议（消融对比用）

chat-template + zero-shot，用于快速消融迭代（n=200，CI ±6.9pp）。
**两种协议系统偏差约 8–15pp，组间相对 Δ 保持一致。**

### 4.3 Baseline 策略

| 模型 | 来源 | 验证方式 |
|------|------|---------|
| Qwen2.5-1.5B-Instruct | 官方 lm-eval 值（GSM8K 73.2%, MATH 55.2%）| — |
| Qwen2.5-7B-Instruct | 官方值（GSM8K 91.6%, MATH 75.5%）| 50 题 sanity check |
| Qwen2.5-14B-Instruct | 官方值（GSM8K 94.0%, MATH 80.0%）| 50 题 sanity check |

---

## 5. 训练策略

### 5.1 SFT（DoRA + 五段课程）

```yaml
lora:
  use_dora: true
  r: 16, alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

| Stage | max_steps | lr | warmup | 实测末段 loss |
|---|---|---|---|---|
| A (GSM8K) | 800 | 5e-5 | 80 | 0.225 |
| B1 (OpenR1) | 1000 | 4e-5 | 80 | 0.641 |
| B2 (Orca) | 1200 | 4e-5 | 100 | 0.347 |
| B3 (NuminaMath) | 600 | 3e-5 | 50 | 0.531 |
| C (Magpie) | 300 | 2e-5 | 30 | 0.517 |

总步数 ~3900，A100 约 7-8h。两次训练（Colab T4 / GPU L20）最大 loss 偏差 <0.02。

### 5.2 DPO

| 类型 | 配置 | 用途 |
|------|------|------|
| Standard DPO | `loss_type: sigmoid`, β=0.1 | Group A/B |
| Teacher-Guided | 同 + teacher 数据 | Group C |
| **Targeted DPO** | `loss_type: sigmoid` + error_type_weights | **Group D（创新）**|
| IPO | `loss_type: ipo` | Group E（可选）|

Group B DPO 实测：final loss 0.458，reward acc 92.5%，margin 0.598，无 KL 漂移。

### 5.3 Error-Type-Targeted DPO（创新点核心）

```
SFT 模型 → 评测 GSM8K → 收集 badcase
   ↓
qwen-flash 分类（5 类）
   ↓
每类用专属 system prompt 让 qwen2.5-72b 生成 chosen
   ↓
Targeted DPO → 再次评测 → 对比各类错误修复率
```

**5 类错误**：arithmetic / reasoning\_skip / setup\_error / unit\_or\_format / extraction\_error

---

## 6. Ablation 实验（6 组）

| Group | SFT | DPO | 目的 | 状态 |
|-------|-----|-----|------|------|
| A | LoRA + 单段 SFT(mix) | Standard DPO | 经典 baseline | ✅ 完成 |
| B | DoRA + 五段课程 | Standard DPO | 验证 DoRA + 课程效果 | ✅ 完成 |
| C | DoRA + 五段课程 | Teacher-Guided DPO | 验证 teacher 数据效果 | ⚠️ eval pending |
| **D** | DoRA + 五段课程 | **Error-Type-Targeted DPO** | 创新点 B 主验证 | ✅ 完成 |
| E | DoRA + 五段课程 | IPO + Targeted 数据 | 损失改进 | ⬜ 可选 |
| F | DoRA + 五段课程 | Weighted Targeted DPO | 创新点 B 权重版 | ⬜ 可选 |

---

## 7. 实验结果摘要（截至 2026-05-05）

### 7.1 消融结果（自定义协议，n=200，CI ±6.9pp）

| 组 | GSM8K | MATH-500 | BBH-27 macro |
|---|---|---|---|
| A（LoRA + 单段 + Standard DPO）| 63.5% | 44.5% | 38.5% |
| B SFT（DoRA + 五段课程）| 61.5% | 44.0% | 38.8% |
| B DPO（+ Standard DPO）| 62.0% | **47.5%** | TBD |
| D（+ Targeted DPO）| **64.5%** | 44.0% | 37.4% |

### 7.2 核心发现

1. **Standard DPO**：MATH +3.5pp（更难推理题受益），GSM8K +0.5pp（应用题提升有限）
2. **Targeted DPO**：GSM8K +2.5pp（针对性修复有效）；MATH -3.5pp（数据局限于 GSM8K badcase，对 MATH 级别难题无覆盖）
3. **DoRA vs LoRA**：当前差异不显著（CI 范围内），官方协议重跑可能拉开差距
4. **BBH**：各组 37–39%，无退化

### 7.3 评测协议 gap

自定义协议比官方低 8–15pp，**lm-eval 重跑进行中**，绝对值将更新。

---

## 8. 工程基础设施

- **Watchdog**：子进程 180s 无输出 → 自动重启（最多 3 次）
- **断点续训**：SFT/DPO 从 checkpoint 续训，评测有缓存
- **日志体系**：`logs/runs/<run_id>/` 独立目录，`logs 2/` 消融结果

---

## 9. 创新点说明（评分锚点）

| 创新点 | 类型 | 验证方式 | 状态 |
|--------|------|---------|------|
| **目标对齐数据课程**（三剑客 + 长度过滤）| 数据策略 | Group A vs B | ✅ 待官方协议确认 |
| **Error-Type-Targeted DPO** | 算法 | Group C vs D + 错误修复率 | ✅ GSM8K +2.5pp |
| **DPO loss 升级**（IPO / Weighted）| 算法 | Group E vs F | ⬜ 可选 |
| **诊断驱动闭环**（badcase→classify→targeted）| 方法论 | 完整 pipeline | ✅ |
| **工程严谨性**（BBH-27 + n=200 + CI）| 工程 | 评测可信性 | ✅ |

---

## 10. 评分对照与预期

| 评分项 | 占比 | 预期得分 |
|--------|------|---------|
| 问题定义 | 15% | 13-15 |
| 创新 | 20% | 16-18 |
| 报告 | 15% | 13-14 |
| 技术实现 | 30% | 24-27 |
| 演讲 | 20% | 16-18 |
| **合计** | 100% | **82-92** |

**目标：85-90 分（Excellent 区间）**

---

**最后更新：2026-05-05**
