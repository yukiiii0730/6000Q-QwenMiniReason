# me2AI · 项目架构与方案设计

> **本文件用途**：固化"项目要做什么、怎么做、为什么这么做"的最终设计；不记录每次迭代过程（迭代过程在 `AI2AI.md`）。
> **维护规则**：每次方案/架构发生实质变化时同步更新本文；每次迭代结束后修订对应章节。

---

## 1. 项目目标（一句话定义）

> 在 5B 以内的小模型（Qwen2.5-1.5B-Instruct）上，通过**目标对齐的数据课程**与**诊断驱动的偏好优化**，实现**接近 7B 模型的数学推理能力**，并验证不同推理数据构造与 DPO 损失改进的有效性。

### 1.1 选择 1.5B 的理由

- **资源受限场景的现实需求**：移动端、边缘设备、私有部署
- **学术研究热点**：DeepSeek-R1-Distill-Qwen-1.5B、TinyGSM、MAmmoTH 等都聚焦此区间
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
┌──────────────────── 本地（CPU 即可，run_local.sh）────────────────────┐
│ ① 数据下载&预处理   v4 五段课程数据                                    │
│ ② API Baseline      公开值优先，自跑 50 题 sanity check                │
│ ③ 数据质量过滤      qwen-flash 评分（可选）                            │
│ ④ Teacher 数据生成  仅小批量难例（成本可控）                           │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │ rsync / Drive 同步 data/ logs/
                           ▼
┌─────────────── GPU（Colab A100 / 服务器，run_gpu.sh）──────────────────┐
│ ⑤ SFT 五段式课程     Stage A (GSM8K) → B1-B3 (三剑客) → C (Magpie)    │
│ ⑥ 合并 + SFT 评测    GSM8K + MATH-500 + BBH-27                        │
│ ⑦ 错误诊断          qwen-flash 把 badcase 归 5 类                     │
│ ⑧ Targeted DPO      按错误类型生成 chosen + Weighted DPO loss         │
│ ⑨ 合并 + 最终评测    + 统计显著性测试                                  │
│ ⑩ Iterative DPO     可选 1-2 轮闭环                                   │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │ rsync 拉回 outputs/merged/, results/
                           ▼
                    本地报告 + 可视化 + LaTeX
```

---

## 3. 数据策略 v4（"三剑客"主干 + in-distribution 对齐）

### 3.1 SFT 五段课程（先精后通）


| 阶段                          | 数据集                                        | 采样   | 长度过滤      | 来源             | 作用                 |
| --------------------------- | ------------------------------------------ | ---- | --------- | -------------- | ------------------ |
| **Stage A** in-distribution | `openai/gsm8k`-train                       | 7.5k | <1024 tok | 学界标准           | 直接对齐 GSM8K 评测分布    |
| **Stage B1** R1 推理深度        | `open-r1/OpenR1-Math-220k` (verified=True) | 10k  | <2048 tok | DeepSeek-R1 蒸馏 | 高质量长 CoT，证书已验证     |
| **Stage B2** 应用题广度          | `microsoft/orca-math-word-problems-200k`   | 15k  | <1024 tok | GPT-4 蒸馏       | 步骤短、覆盖广，1.5B 友好    |
| **Stage B3** 题型多样           | `AI-MO/NuminaMath-CoT`（去 source=gsm8k）     | 8k   | <2048 tok | 多源 expert      | 奥赛/AMC/AOPS 风格     |
| **Stage C** 通用推理            | `Magpie-Align/Magpie-Reasoning-150K`       | 3k   | <2048 tok | Llama-70B 蒸馏   | 防 BBH 退化（占比降到 ~7%） |


**总量约 43.5k**，跨集去重 + 长度过滤后实际 ~38k，1.5B context=2048 完全吃得下。

### 3.2 字段适配（解决三剑客字段不一致）

所有数据集 normalize 为统一 schema：

```json
{"instruction": "...", "input": "<问题>", "output": "<解答>", "source": "<dataset_tag>"}
```


| 数据集              | 原字段                                                       | 映射                                            |
| ---------------- | --------------------------------------------------------- | --------------------------------------------- |
| GSM8K            | `question` / `answer`                                     | `input` / `output`（answer 中的 `####` 改成 `答案：`） |
| OpenR1-Math      | `problem` / `generations[]` + `correctness_math_verify[]` | 选 verified=True 的 generation 为 `output`       |
| Orca-Math        | `question` / `answer`                                     | 直接映射                                          |
| NuminaMath-CoT   | `problem` / `solution` + `source`                         | 直接映射，source 透传用于过滤                            |
| Magpie-Reasoning | `instruction` / `response`                                | `instruction`/`output`（input 留空）              |


### 3.3 跨数据集去重

- 使用 problem 文本规范化（去标点、转小写、压缩空白）后的 SHA-1 hash 去重
- NuminaMath-CoT 的 `source=gsm8k` 子集 vs GSM8K-train **强制全部去除**
- 同集内重复也去除

### 3.4 蒸馏策略（务实版）

**核心原则：用现成的高质量蒸馏数据，不重复劳动**


| 任务                  | 不做                                    | 做                                                 |
| ------------------- | ------------------------------------- | ------------------------------------------------- |
| 大规模 SFT 数据          | ❌ 自己跑 235B teacher                    | ✅ 直接用 OpenR1-Math（已是 R1 蒸馏） + Orca-Math（GPT-4 蒸馏） |
| 错误分类                | —                                     | ✅ qwen-flash（便宜，¥0.5/300 条）                       |
| Targeted DPO chosen | ❌ 235B-thinking（贵且 long CoT 不适合 1.5B） | ✅ qwen2.5-72b-instruct（中等大小，sweet spot）           |
| Iterative DPO 难例    | ❌ 235B 大批量                            | ✅ qwen2.5-72b 小批量（500 条/轮）                        |


**为什么不用 235B-thinking 大规模生成**：

1. **Capacity gap**：teacher 比 student 大 150x 时，CoT 风格 + 长度差距让 1.5B 学不来
2. **DeepSeek-R1 论文证据**：他们在 1.5B 上做 RL 不如直接 SFT 中等长度蒸馏数据
3. **成本**：235B thinking 输出价 ¥0.06/1k，比 72B 贵 5x
4. **OpenR1-Math 已是 R1 蒸馏好的**，重复劳动

### 3.5 v4 比 v3 改进点


| 维度        | v3                 | v4                                       |
| --------- | ------------------ | ---------------------------------------- |
| 数据课程段数    | 2 段                | **5 段（三剑客主干）**                           |
| 主干数据来源    | MetaMathQA + GSM8K | **OpenR1 + Orca-Math + NuminaMath（三剑客）** |
| 长度过滤      | 无                  | **有（<1024/<2048 双档）**                    |
| 跨集去重      | 无                  | **有（hash 去重）**                           |
| Magpie 占比 | 30%                | **7%**（降低噪声）                             |


---

## 4. 评测策略

### 4.1 评测组合（按重要性）


| Benchmark      | 类型          | 量             | 作用                | 实现                      |
| -------------- | ----------- | ------------- | ----------------- | ----------------------- |
| **GSM8K** test | 小学应用题       | 1319 / 200    | 主指标①（应用题）         | `eval/gsm8k_eval.py`    |
| **MATH-500**   | 竞赛数学（5 级难度） | 500 全量        | **主指标②（真正的推理标尺）** | `eval/math_eval.py` ⭐   |
| BBH 27 tasks   | 通用推理        | 100/任务 = 2700 | 防退化辅指标            | `eval/bbh_full_eval.py` |
| MMLU-Math      | 多选数学        | ~250          | 防过拟合（可选）          | TBD                     |


### 4.2 严格评测协议

- **chat_template**：所有评测必须套 Qwen2.5 chat template
- **max_new_tokens=1024**：避免长 CoT 被截断
- **答案抽取多策略**：`\boxed{}` > `####` > "答案：" > 末段数字
- **N≥200**：GSM8K 和 BBH 子任务样本数 ≥100，避免 95% CI ±14% 的统计无意义
- **统计显著性**：相邻方法对比用 McNemar 配对检验 + bootstrap CI

### 4.3 Baseline 策略（省 90% API 成本）


| 模型                    | 来源                                 | 验证方式                 |
| --------------------- | ---------------------------------- | -------------------- |
| Qwen2.5-1.5B-Instruct | 自跑全量（200/500/2700）                 | —                    |
| Qwen2.5-7B-Instruct   | **官方公开值**（GSM8K 91.6%, MATH 75.5%） | 自跑 50 题 sanity check |
| Qwen2.5-14B-Instruct  | **官方公开值**（GSM8K 94.0%, MATH 80.0%） | 自跑 50 题 sanity check |
| Qwen3-235B-Thinking   | 不跑（贵且非主比较对象）                       | —                    |


引用：Qwen Team, "Qwen2.5 Technical Report", arXiv:2412.15115。

---

## 5. 训练策略

### 5.1 SFT（DoRA + 五段课程）

```yaml
lora:
  use_dora: true
  r: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

每段独立 lr/steps：


| Stage           | max_steps | learning_rate | warmup |
| --------------- | --------- | ------------- | ------ |
| A (GSM8K)       | 800       | 5e-5          | 80     |
| B1 (OpenR1)     | 1000      | 4e-5          | 80     |
| B2 (Orca)       | 1200      | 4e-5          | 100    |
| B3 (NuminaMath) | 600       | 3e-5          | 50     |
| C (Magpie)      | 300       | 2e-5          | 30     |


总步数 ~3900，A100 约 3-4h。

### 5.2 DPO（loss 升级 + 错误类型加权）

支持 4 种 loss：


| 类型           | 配置                                     | 用途                          |
| ------------ | -------------------------------------- | --------------------------- |
| Standard DPO | `loss_type: sigmoid`                   | baseline                    |
| **IPO**      | `loss_type: ipo`                       | 防 reward hacking            |
| **Hinge**    | `loss_type: hinge`                     | 鲁棒边界                        |
| **Weighted** | `loss_type: sigmoid` + 数据带 `weight` 字段 | Error-Type-Targeted 加权（创新点） |


`base_adapter_path: outputs/sft_merged`（合并后的完整模型，避免 adapter 直接 load 的歧义）。

### 5.3 Error-Type-Targeted DPO（创新点 B 主体）

```
SFT 模型 → 评测 GSM8K → 自动收集 badcase
   ↓
qwen-flash 分类到 5 类（arithmetic / reasoning_skip / setup_error / unit_or_format / extraction_error）
   ↓
按错误类型用专属 system prompt 让 Teacher 生成 chosen
（如 arithmetic 类强调"每步写出算式 → 中间结果"）
   ↓
Targeted DPO 训练（可选 weighted loss，对修复难度高的类型给更大 β）
   ↓
再次评测 → 对比每类错误的修复率（可视化为条形图）
```

---

## 6. Ablation 实验（v4 升级到 6 组）


| Group | SFT 配方             | DPO 配方                            | 目的              |
| ----- | ------------------ | --------------------------------- | --------------- |
| A     | LoRA + 单段 SFT(mix) | Vanilla DPO (distilabel, sigmoid) | 经典 baseline     |
| B     | DoRA + 五段课程        | Vanilla DPO (distilabel, sigmoid) | 验证 DoRA + 课程效果  |
| C     | DoRA + 五段课程        | Teacher-Guided DPO (统一 prompt)    | 验证 teacher 数据效果 |
| D     | DoRA + 五段课程        | **Error-Type-Targeted DPO**（创新核心） | 创新点 B 主验证       |
| E     | DoRA + 五段课程        | **IPO loss + Targeted 数据**        | 损失改进 vs 标准 DPO  |
| F     | DoRA + 五段课程        | **Weighted Targeted DPO**         | 创新点 B 权重版       |


每组评测 GSM8K + MATH-500 + BBH-27，记录 3 项指标 + Δ + 95% CI。

---

## 7. 工程基础设施

### 7.1 Watchdog（防卡死）

`scripts/watchdog_run.py`：子进程 stdout/stderr 超过 N 秒（默认 180s）无新输出 → kill 整个进程组 → 自动重启（最多 3 次）。所有长时步骤已套上。

### 7.2 断点续训

- SFT/DPO trainer 支持从 `outputs/.../checkpoint-XXX` 续训
- 评测结果有缓存：相同模型路径 + 配置不重跑

### 7.3 日志体系

- `logs/runs/<run_id>/` 每次运行独立目录
- `logs/gpu_latest` 软链最新一次 GPU 运行
- `logs/runs/<id>/.watchdog.json` 记录 watchdog 重试事件

---

## 8. 文件结构

```
6000Q-QwenMiniReason 2/
├── me2AI.md                ⭐ 本文件：架构与方案
├── AI2AI.md                ⭐ 迭代记录
├── README.md               ⭐ 最终用户文档
│
├── config/
│   ├── sft_config.yaml     ⭐ v4 五段课程
│   ├── dpo_config.yaml     ⭐ v4 含 loss_type
│   └── benchmark_models.yaml
│
├── data/processed/
│   ├── sft_train.json      五段 source 标签合并
│   ├── dpo_train.json      distilabel 兜底
│   ├── dpo_teacher_*.json  Teacher 生成
│   └── dpo_targeted_*.json Error-Type-Targeted（创新核心）
│
├── scripts/
│   ├── prepare_data.py     ⭐ v4 三剑客 + 长度过滤 + 去重
│   ├── sft_train.py        DoRA + 五段课程
│   ├── dpo_train.py        ⭐ v4 含 loss_type + 权重
│   ├── merge_lora.py
│   ├── classify_errors.py  错误分类（创新点 B 第 1 步）
│   ├── build_targeted_dpo.py  Targeted 数据生成（创新点 B 第 2 步）
│   ├── build_teacher_dpo.py   通用 Teacher 数据
│   ├── llm_quality_filter.py  数据质量评分
│   ├── iterative_dpo_loop.py  可选闭环
│   ├── run_ablation.py     ⭐ v4 升级到 6 组
│   ├── stats_significance.py  ⭐ McNemar + bootstrap CI
│   └── watchdog_run.py     ⭐ 防卡死
│
├── eval/
│   ├── gsm8k_eval.py       ⭐ chat_template + 健壮提取
│   ├── gsm8k_api_eval.py
│   ├── math_eval.py        ⭐ MATH-500 本地（v4 新增）
│   ├── math_api_eval.py    ⭐ MATH-500 API（v4 新增）
│   ├── bbh_eval.py         ⭐ chat_template + 健壮提取
│   ├── bbh_api_eval.py
│   ├── bbh_full_eval.py    BBH 27 任务 wrapper
│   ├── compare_table.py    ⭐ 优先官方 baseline
│   ├── visualize.py        ⭐ 雷达图 + 错误饼图 + ablation 条形图
│   └── published_baselines.json  Qwen 官方公开值
│
├── run_local.sh            CPU 阶段（数据/baseline/Teacher）
├── run_gpu.sh              GPU 阶段（SFT/DPO/评测）
└── run_train.sh            兼容入口（路由）
```

⭐ 标记为 v4 关键文件。

---

## 9. 创新点说明（评分锚点）


| 创新点                                              | 类型   | 验证方式                               | 报告章节                      |
| ------------------------------------------------ | ---- | ---------------------------------- | ------------------------- |
| **目标对齐数据课程**（v3→v4 三剑客 + 长度过滤）                   | 数据策略 | Group A vs B 对比 +5pp 预期            | Methodology + Results     |
| **Error-Type-Targeted DPO**（创新核心）                | 算法   | Group C vs D 对比 + 各类错误修复率饼图        | Methodology + Results（重点） |
| **DPO loss 升级**（IPO / Weighted）                  | 算法   | Group E vs F 对比                    | Methodology + Ablation    |
| **诊断驱动闭环**（badcase → classify → targeted → eval） | 方法论  | 完整 pipeline 案例 + before/after 错误分布 | Methodology               |
| **工程严谨性**（27 任务 BBH + 200 样本 + CI + Watchdog）    | 工程   | 评测可信性章节                            | Limitations               |


---

## 10. 评分对照与目标


| 评分项  | 占比  | 落地保证                                 |
| ---- | --- | ------------------------------------ |
| 问题定义 | 15% | §1（明确论证为什么 1.5B / 数学 / SFT+DPO）      |
| 创新   | 20% | §9 四点创新 + §6 六组 ablation             |
| 报告   | 15% | me2AI/AI2AI 提供素材，最后写 LaTeX           |
| 技术实现 | 30% | §3-§7 完整 pipeline + MATH-500 + 统计显著性 |
| 演讲   | 20% | 后期准备：可视化 + 错误案例 + Δ 表格               |


**目标总分：85-92（Excellent 区间）**

---

## 11. 运行指引（最终版）

详见 `README.md`。

---

**最后更新：2026-04-28（v4 落地阶段）**