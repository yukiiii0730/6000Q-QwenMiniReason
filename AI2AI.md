# AI2AI · 迭代记录

> **本文件用途**：按时间顺序记录每次迭代的"做了什么、为什么改、得到什么"。
> **维护规则**：每次运行/重大修改结束后追加新条目，不删旧条目。最新条目在底部。

---

## 索引

- [v1 · 初版（LoRA + 单段 SFT）](#v1--初版lora--单段-sft)
- [v2 · 重构（DoRA + 课程 SFT + Teacher DPO）](#v2--重构dora--课程-sft--teacher-dpo)
- [v3 · in-distribution 修正](#v3--in-distribution-修正)
- [v4 · 三剑客主干 + MATH-500 + DPO loss 升级](#v4--三剑客主干--math-500--dpo-loss-升级)

---

## v1 · 初版（LoRA + 单段 SFT）

**时间**：项目初期 → 2026-04 中

### 方案

- **SFT 数据**：NuminaMath-CoT 单数据集，单段训练
- **DPO 数据**：`microsoft/orca-math-word-problems-200k`（误用——它没有 chosen/rejected）
- **PEFT**：LoRA r=16，`use_dora=false`
- **评测**：GSM8K 50 题 + BBH 单子任务（boolean_expressions）50 题

### 实测结果（README 之前隐藏数据）


| 模型                               | GSM8K              | BBH (boolean)  |
| -------------------------------- | ------------------ | -------------- |
| Qwen2.5-1.5B-Instruct (Baseline) | 46.0%              | 80.0%          |
| + SFT only                       | 50.0% (+4.0pp)     | 86.0% (+6.0pp) |
| **+ SFT + DPO**                  | **40.0% (-6.0pp)** | 84.0% (-2.0pp) |
| Qwen2.5-7B (参考)                  | 82.0%              | 78.0%          |
| Qwen2.5-14B (参考)                 | 64.0%              | 68.0%          |


### 发现的问题

1. ❌ **DPO 让 GSM8K 退步 6pp**：`beta=0.3` 过高、`max_seq_length=1536` 截断 CoT、orca-math 没有真实 chosen/rejected
2. ❌ **README 数据虚高**（声称 +14pp 实际 -6pp），需要诚实修正
3. ❌ **50 题样本量不足**：95% CI ±14%，几乎无统计意义
4. ❌ **BBH 只评了 1/27 子任务**：不能代表 BBH 真实水平
5. ❌ **没有 MATH 评测**：GSM8K 只是入门，缺真正的推理标尺

---

## v2 · 重构（DoRA + 课程 SFT + Teacher DPO）

**时间**：2026-04 中

### 主要改动

- ✅ **PEFT 升级**：LoRA → **DoRA**（Weight-Decomposed LoRA，性能提升 1-3pp）
- ✅ **SFT 课程化**：单段 → **两段课程**（Stage 1a: OpenR1-Math 20k → Stage 1b: Magpie 8k）
- ✅ **DPO 数据修复**：废弃 orca-math（SFT 数据），改用 `argilla/distilabel-math-preference-dpo`（真实 chosen/rejected）
- ✅ **DPO 超参修复**：`beta` 0.3→0.1，`max_seq_length` 1536→2048，`lr` 3e-6→1e-5
- ✅ **Teacher-Guided DPO**：用 `qwen3-235b-a22b-thinking-2507` 生成 chosen + student badcase 当 rejected
- ✅ **Iterative DPO 框架**：评测 → 收 badcase → Teacher 补 → 再 DPO 闭环
- ✅ **本地/GPU 分工**：`run_local.sh`（CPU 阶段）+ `run_gpu.sh`（GPU 阶段）+ `run_train.sh`（路由）
- ✅ **样本量提升**：50 → **200**

### v2 仍存在的问题（v3 反思发现）

1. ❌ **训-测分布严重错位**：OpenR1-Math（AIME/Olympiad）vs GSM8K（小学应用题）
  - 1.5B 学到了"奥赛模板"用不到 GSM8K 上
2. ❌ **BBH 评测仍是单子任务**（v2 没改）
3. ❌ **iterative_dpo_loop.py bug**：传 `--adapter_path` 给只接受 `--model_path` 的 eval 脚本
4. ❌ **DPO base_adapter_path 指向 `outputs/sft`**（adapter 目录），与 4-bit 量化模型加载有歧义
5. ❌ **GSM8K/BBH eval 不套 chat_template**：Qwen2.5-Instruct 强依赖 chat 模板，会导致评分虚低

---

## v3 · in-distribution 修正

**时间**：2026-04-27 ~ 04-28

### 主要改动

#### 评测健壮化（P0 全修）

- ✅ `**eval/bbh_full_eval.py`**：BBH 27 子任务全量 wrapper（修复 v2 只评 1/27）
- ✅ `**eval/gsm8k_eval.py` / `eval/bbh_eval.py**`：套 `apply_chat_template` + `max_new_tokens=1024` + 健壮答案提取（`\boxed{}` / `####` / "答案：" / 末段数字 多策略）
- ✅ **iterative_dpo_loop.py**：先 `merge_lora.py` 再 `--model_path` 评测（修 bug）
- ✅ **DPO base_adapter_path 改 `outputs/sft_merged`**（完整模型，避免 adapter 直接 load 歧义）
- ✅ `run_gpu.sh` SFT→DPO 衔接处自动 merge

#### 数据策略 v3（in-distribution 对齐）

- ✅ 废弃 OpenR1-Math 当 Stage 1a 主力（错位严重）
- ✅ 改用 **MetaMathQA + GSM8K-train + Magpie**：
  - Stage 1a-i: GSM8K-train 7.5k（in-distribution）
  - Stage 1a-ii: MetaMathQA 15k（GSM8K+MATH 增强）
  - Stage 1b: Magpie 8k

#### 创新点 B：Error-Type-Targeted DPO（创新核心）

- ✅ `**scripts/classify_errors.py`**：用 qwen-flash 把 student 的 GSM8K badcase 自动归类到 5 类（arithmetic / reasoning_skip / setup_error / unit_or_format / extraction_error）
- ✅ `**scripts/build_targeted_dpo.py**`：按错误类型用**专属 system prompt** 让 Teacher 生成 chosen，rejected 用 student 真实错误
- ✅ `**scripts/run_ablation.py`**：4 组对照实验（A: LoRA / B: DoRA+课程 / C: Teacher-Guided / D: Error-Type-Targeted）

#### 工程基础设施

- ✅ `**scripts/watchdog_run.py**`：通用进程监控，子进程 stdout/stderr 超过 N 秒（默认 180s）无新输出 → kill 进程组 + 自动重启（最多 3 次）
- ✅ 已嵌入 `run_local.sh` / `run_gpu.sh` 所有长时步骤

### v3 评估反思（用户提出的关键问题）

1. ❓ **数据组合是否符合"1.5B → 7B 数学推理"目标**？
  - MetaMathQA + GSM8K-train 太偏 in-distribution 而缺乏推理深度，无法逼近 7B
2. ❓ **GSM8K + BBH 是不是数学推理的标准评测组合**？
  - **不是**。学界标准是 GSM8K + **MATH/MATH-500**，BBH 是通用推理不算数学
3. ❓ **235B 蒸馏给 1.5B 真的有用吗**？
  - **效果不佳**（capacity gap）。DeepSeek-R1 论文也证实 1.5B 上 RL 不如直接 SFT 中等长度蒸馏数据
4. ❓ **多数据集格式如何统一处理**？
  - 三剑客字段差异巨大（problem/question/query/generations），需要严格 normalize + 长度过滤 + 去重

→ 触发 v4 重构。

---

## v4 · 三剑客主干 + MATH-500 + DPO loss 升级

**时间**：2026-04-28 起

### 设计目标

1. 数据策略对齐学界共识："三剑客"（NuminaMath-CoT + Orca-Math + OpenR1）+ in-distribution 锚点（GSM8K-train）
2. 评测组合升级：加入 **MATH-500** 作为真正的推理标尺
3. DPO 算法创新：增加 **IPO / Hinge / Weighted** 损失函数对照
4. Baseline 务实：**引用 Qwen 官方公开值** + 自跑 50 题 sanity check（省 90% API 钱）
5. 文档化：me2AI（架构）+ AI2AI（迭代）+ README（最终用户）三件套

### 计划改动

#### Phase 1: 文档化（本次先行）

- `me2AI.md` 创建
- `AI2AI.md` 创建（v1-v3 历史回顾 + v4 计划）

#### Phase 2: 数据策略 v4

- `scripts/prepare_data.py` 重写：v4 五段课程 + 三剑客字段适配 + 长度过滤 + 跨集去重
- `config/sft_config.yaml`：五段课程配置（A/B1/B2/B3/C）

#### Phase 3: MATH-500 评测

- `eval/math_eval.py`：本地 MATH-500 评测（含 LaTeX/分数答案匹配）
- `eval/math_api_eval.py`：API 版本
- `run_local.sh` / `run_gpu.sh`：集成 MATH-500

#### Phase 4: DPO loss 升级

- `scripts/dpo_train.py`：增加 `loss_type` 切换（sigmoid/ipo/hinge）+ Error-Type-Targeted 数据带 `weight` 字段支持
- `config/dpo_config.yaml`：增加 `loss_type` 字段
- `scripts/run_ablation.py`：升级到 6 组（加 IPO + Weighted-DPO）

#### Phase 5: Baseline 务实化

- `eval/published_baselines.json`：Qwen 官方公开值
- `eval/compare_table.py`：优先用官方值，自跑结果作为 sanity check 补充

#### Phase 6: 可视化与统计

- `eval/visualize.py` 升级：雷达图 + 错误分布饼图 + ablation 条形图
- `scripts/stats_significance.py`：McNemar 配对检验 + bootstrap CI

#### Phase 7: 本地实跑

- 下载 v4 数据集（三剑客 + GSM8K-train + Magpie）
- 7B/14B sanity check 各 50 题
- 数据质量过滤（可选）

#### Phase 8: 文档同步

- 同步更新 me2AI / AI2AI / README

---

## 运行记录

> 每次实跑后追加。格式：`### YYYY-MM-DD HH:MM · 阶段名`

### 2026-04-28 01:00 · v3→v4 决策

- 用户提出 5 个核心问题：评测、成本、可执行性、目标定义、文档化
- 反思发现 v3 仍偏 in-distribution，缺数学推理真正标尺（MATH）
- 决定升级到 v4：三剑客主干 + MATH-500 + DPO loss 改进
- 目标：拿到 85-92 分（Excellent 区间）

### 2026-04-28 06:24 · v4 文档化

- 创建 me2AI.md（11 章节，含架构图 + 评分对照）
- 创建 AI2AI.md（本文件，含 v1-v4 完整迭代史）
- 接下来开始 v4 代码落地

### 2026-04-29 · GPU 服务器 + Colab 训练完成（v4 首次完整跑通）

**GPU 服务器（NVIDIA L20 47.4GB）— Group B（DoRA + 五段课程 + Standard DPO）**

| 阶段 | 步数 | 初始loss | 末段loss | 耗时 |
|------|------|---------|---------|------|
| Stage A gsm8k | 800 | 1.286 | 0.225 | ~2.5h |
| Stage B1 openr1 | 1000 | 0.952 | 0.641 | ~2.1h |
| Stage B2 orca | 1200 | 0.549 | 0.347 | ~1.1h |
| Stage B3 numina | 600 | 0.616 | 0.531 | ~0.8h |
| Stage C magpie | 300 | 0.623 | 0.517 | ~0.5h |
| DPO（distilabel，sigmoid，β=0.1）| 600 | 0.687 | 0.458 | ~1.9h |

**GPU Group B 评测结果（n=50，注：CI ±14pp，待扩充）**

| 指标 | SFT only | SFT+DPO | Δ |
|------|---------|---------|---|
| GSM8K | 62% | **66%** | +4pp |
| MATH-500 | 50% | **56%** | +6pp |
| BBH-27 macro | 38.52% | **41.11%** | +2.6pp |

**Colab T4（Teacher-DPO）— Group C 训练完成，评测待补**

- SFT 与 GPU 完全一致（seed=42，最大偏差 <0.02，hardware noise）
- DPO train_loss 0.2394（vs GPU 0.5611），teacher 数据质量更高导致更低 loss
- 评测文件为旧 run 残留（03-14），**v4 Colab 模型从未正式评测**

**关键发现**

1. DPO 无 KL 漂移：rewards/chosen 维持 0 附近，margin 主要来自压低 rejected
2. Stage A 过拟合风险：800 steps + 16.7 packing-epoch，loss 降到 0.22，后续消融可将 max_steps 减到 500 验证
3. n=50 评测太小：95% CI ±14pp，无法区分 4-6pp 的组间差异

**待完成事项（消融实验）**

- [ ] L1: 评测 Colab Group C 模型（需下载权重）
- [ ] L2: Group B 评测扩充到 n=200（`bash run_eval_expanded.sh`）
- [ ] L3: GSM8K badcase 错误分类（`bash run_local_pipeline.sh`）
- [ ] L4: 生成 Targeted DPO 数据（L3 完成后）
- [ ] L5: 统计显著性检验（所有 eval 完成后）
- [ ] L6: 可视化（`eval/visualize.py`）
- [ ] G1-G3: Group A LoRA baseline（Colab `colab_ablation.ipynb`）
- [ ] G4-G5: Group D Targeted DPO（创新核心，Colab `colab_ablation.ipynb`）
- [ ] G6-G7: Group E/F 可选

**评测样本量规范（v4 统一）**

- GSM8K: n=200（CI ±6.9pp）
- MATH-500: n=200（CI ±6.9pp）
- BBH-27: 30/task × 27 = 810（macro CI ±3.4pp）

### 2026-04-30 · Colab 消融流水线整合 + 质量检查

**整合 colab_eval_l1l2 → colab_ablation**

- 将 `colab_eval_l1l2.ipynb`（L1+L2 扩充评测）合并到 `colab_ablation.ipynb` 的 P1 Phase
- 删除 `colab_eval_l1l2.ipynb`，统一为单个 notebook 完成全部消融实验
- `colab_ablation.ipynb` 结构：P0（环境）→ P1（Group B 评测 n=200）→ P2（错误分类+Targeted DPO 数据）→ P3（Group A 训练+评测）→ P4（Group D 训练+评测）→ P5（汇总+同步）

**质量检查修复（3 个问题）**

1. Cell 6 f-string 嵌套花括号 `{'='*60}` → 改为 `SEP` 变量（Python 3.12+ 语法，Colab 3.10/3.11 不支持）
2. Cells 10/12/15 移除 `--resume_from_checkpoint` CLI 参数（`sft_train.py`/`dpo_train.py` 内部自动检测 checkpoint，无此 CLI 参数）
3. `eval/compare_table.py` 确认为模块级脚本，subprocess 调用无需修改

**eval/model_loader.py Mac MPS 兼容**

- 检测 bitsandbytes NF4 量化权重（uint8 dtype）时自动回退到 base model + adapter 加载
- 清除 config.json 中嵌入的 quantization_config
- 注意：模型权重本身仍为 NF4 量化格式，Mac MPS 无法直接加载——评测统一在 Colab A100 执行

**待完成（Colab 执行）**

- [ ] P1: Group B n=200 评测（GSM8K/MATH/BBH）
- [ ] P2: 错误分类 + Targeted DPO 数据生成
- [ ] P3: Group A 训练 + 评测
- [ ] P4: Group D 训练 + 评测
- [ ] P5: 汇总