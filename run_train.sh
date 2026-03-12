#!/usr/bin/env bash
# =============================================================================
# run_train.sh — Qwen2.5-1.5B Reasoning 两阶段微调一键训练脚本
#
# 用法：
#   bash run_train.sh                   # 完整流程（Baseline + 数据 + SFT + DPO + 合并 + 评测）
#   bash run_train.sh --quick           # 快速测试（本地 JSON，各 500 条，SFT=50 steps，DPO=30 steps）
#   bash run_train.sh --skip-data       # 跳过数据下载（已有缓存时使用）
#   bash run_train.sh --skip-sft        # 跳过 SFT（已训练时使用）
#   bash run_train.sh --skip-dpo        # 跳过 DPO
#   bash run_train.sh --skip-eval       # 跳过最终评测
#   bash run_train.sh --skip-baseline   # 跳过 Baseline 评测（训练前）
#   bash run_train.sh --only-sft        # 仅运行 SFT（跳过 DPO、合并、评测）
#   bash run_train.sh --force           # 强制重跑所有阶段（忽略已有产物）
#   bash run_train.sh --sft_n 20000 --dpo_n 20000   # 自定义采样量
#
# 数据集：
#   完整流程  SFT : AI-MO/NuminaMath-CoT + Magpie-Align/Magpie-Reasoning-150K（各 50k）
#   完整流程  DPO : microsoft/orca-math-word-problems-200k（50k）
#   --quick   SFT : data/processed/sft_train.json（本地 500 条）
#   --quick   DPO : data/processed/dpo_train.json（本地 500 条）
#
# 断点续跑（自动检测）：
#   若 outputs/sft/ 已含模型文件，SFT 阶段自动跳过；DPO 同理。
#   使用 --force 可强制重跑所有阶段。
# =============================================================================
set -euo pipefail

# ── 颜色输出 ─────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✅ $1${NC}"; }
info() { echo -e "${CYAN}[$(date '+%H:%M:%S')] ➤  $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  $1${NC}"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}" >&2; exit 1; }

# ── 默认参数 ──────────────────────────────────────────────────────────────────
QUICK=false
SKIP_DATA=false
SKIP_SFT=false
SKIP_DPO=false
SKIP_EVAL=false
SKIP_BASELINE=false
ONLY_SFT=false
FORCE=false
SFT_N=50000
DPO_N=50000

# ── 解析参数 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)      QUICK=true ;;
        --skip-data)  SKIP_DATA=true ;;
        --skip-sft)   SKIP_SFT=true ;;
        --skip-dpo)   SKIP_DPO=true ;;
        --skip-eval)      SKIP_EVAL=true ;;
        --skip-baseline)  SKIP_BASELINE=true ;;
        --only-sft)       ONLY_SFT=true ;;
        --force)      FORCE=true ;;
        --sft_n)      SFT_N="$2"; shift ;;
        --dpo_n)      DPO_N="$2"; shift ;;
        *) die "未知参数: $1" ;;
    esac
    shift
done

if $QUICK; then
    SFT_N=500; DPO_N=500
    warn "快速测试模式：每个数据集仅取 500 条，SFT max_steps=50，DPO max_steps=30"
fi

# ── 切换到项目根目录 ──────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
info "项目目录：$PROJECT_DIR"

# ── 激活虚拟环境 ─────────────────────────────────────────────────────────────
VENV_DIR="$PROJECT_DIR/.venv"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
    source "$VENV_DIR/bin/activate"
    info "虚拟环境已激活：$VENV_DIR"
else
    die "未找到虚拟环境 $VENV_DIR，请先运行：\n  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
fi

# ── 检查 Python ───────────────────────────────────────────────────────────────
PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PYTHON_VER"

# ── 读取并验证 HuggingFace Token ─────────────────────────────────────────────
HF_TOKEN=$(python - <<'EOF'
import yaml, sys
try:
    with open("config/sft_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    t = cfg.get("hf_token", "").strip()
    print(t)
except Exception as e:
    print("")
EOF
)

if [[ -z "$HF_TOKEN" ]]; then
    die "请先在 config/sft_config.yaml 中填写 hf_token 字段，再运行此脚本"
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
log "HuggingFace Token 已加载"

# ── 检查 GPU ──────────────────────────────────────────────────────────────────
python - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("⚠️  未检测到 GPU，将使用 CPU（速度极慢）")
    sys.exit(0)
name  = torch.cuda.get_device_name(0)
vram  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
print(f"GPU : {name}  显存: {vram} GB")
if vram < 12:
    print("⚠️  显存 < 12GB，可能 OOM，建议减小 batch_size 或 max_seq_length")
EOF

# ── 创建目录 ──────────────────────────────────────────────────────────────────
mkdir -p outputs/sft outputs/dpo outputs/merged data/processed logs
info "输出目录已准备"

# ── 自动检测已完成的阶段 ──────────────────────────────────────────────────────
# 若输出目录下存在 adapter_config.json，则认为该阶段已完成
stage_done() {
    local dir="$1"
    [[ -f "$dir/adapter_config.json" ]] && return 0
    return 1
}

if ! $FORCE; then
    if stage_done "outputs/sft" && ! $SKIP_SFT; then
        warn "检测到 outputs/sft/adapter_config.json，SFT 已完成，自动跳过（使用 --force 可强制重跑）"
        SKIP_SFT=true
    fi
    if stage_done "outputs/dpo" && ! $SKIP_DPO; then
        warn "检测到 outputs/dpo/adapter_config.json，DPO 已完成，自动跳过（使用 --force 可强制重跑）"
        SKIP_DPO=true
    fi
fi

# ── 快速测试：切换到本地 JSON，覆写 max_steps ────────────────────────────────
# 全量模式：用 HuggingFace 数据集（yaml 中的 datasets / dataset 字段）
# quick 模式：用本地预处理 JSON（yaml 中的 dataset_path 字段），并临时隐藏 HF 字段
if $QUICK; then
    python - <<'EOF'
import yaml

# SFT config: 隐藏 datasets 字段，只用 dataset_path，并覆写 steps
with open("config/sft_config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.pop("datasets", None)          # quick 模式不用 HF 数据集
cfg["train"]["max_steps"] = 50
with open("config/sft_config.yaml", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

# DPO config: 隐藏 dataset 字段，只用 dataset_path，并覆写 steps
with open("config/dpo_config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg.pop("dataset", None)           # quick 模式不用 HF 数据集
cfg["train"]["max_steps"] = 30
with open("config/dpo_config.yaml", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

print("⚡ quick 模式：使用本地 JSON，max_steps 已覆写为 SFT=50 / DPO=30")
EOF
else
    # 全量模式：确保 HF 数据集字段存在（防止上次 quick 运行后未还原）
    python - <<EOF
import yaml

with open("config/sft_config.yaml") as f:
    cfg = yaml.safe_load(f)
if "datasets" not in cfg:
    cfg["datasets"] = [
        {"name": "AI-MO/NuminaMath-CoT",             "split": "train", "max_samples": $SFT_N},
        {"name": "Magpie-Align/Magpie-Reasoning-150K","split": "train", "max_samples": $SFT_N},
    ]
else:
    for d in cfg["datasets"]:
        d["max_samples"] = $SFT_N
with open("config/sft_config.yaml", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

with open("config/dpo_config.yaml") as f:
    cfg = yaml.safe_load(f)
if "dataset" not in cfg:
    cfg["dataset"] = {"name": "microsoft/orca-math-word-problems-200k", "split": "train", "max_samples": $DPO_N}
else:
    cfg["dataset"]["max_samples"] = $DPO_N
with open("config/dpo_config.yaml", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f"📦 全量模式：SFT 数据集 NuminaMath-CoT + Magpie-Reasoning-150K（各 $SFT_N 条）")
print(f"📦 全量模式：DPO 数据集 orca-math-word-problems-200k（$DPO_N 条）")
EOF
fi

# =============================================================================
# Step 0：Baseline 评测（训练前，原始模型）
# =============================================================================
if $SKIP_BASELINE; then
    warn "已跳过 Baseline 评测（--skip-baseline）"
elif ! $FORCE && [[ -f "logs/gsm8k_baseline.json" ]] && [[ -f "logs/bbh_baseline.json" ]]; then
    warn "已检测到 baseline 评测结果，自动跳过（使用 --force 可强制重跑）"
else
    info "Step 0/5 — Baseline 评测（原始模型，训练前）"
    BASE_MODEL=$(python -c "import yaml; c=yaml.safe_load(open('config/sft_config.yaml')); print(c['model_name'])")
    BASELINE_N=200; $QUICK && BASELINE_N=50

    info "GSM8K baseline → $BASE_MODEL"
    python eval/gsm8k_eval.py \
        --model_path  "$BASE_MODEL" \
        --max_samples "$BASELINE_N" \
        --output      logs/gsm8k_baseline.json \
        2>&1 | tee logs/gsm8k_baseline.log

    info "BBH baseline → $BASE_MODEL"
    python eval/bbh_eval.py \
        --model_path  "$BASE_MODEL" \
        --max_samples "$BASELINE_N" \
        --output      logs/bbh_baseline.json \
        2>&1 | tee logs/bbh_baseline.log

    log "Baseline 评测完成"
fi

# =============================================================================
# Step 1：数据准备
# =============================================================================
if $SKIP_DATA; then
    warn "已跳过数据下载（--skip-data）"
else
    info "Step 1/5 — 数据集下载与预处理"
    python scripts/prepare_data.py \
        --sft_n "$SFT_N" \
        --dpo_n "$DPO_N" \
        2>&1 | tee logs/prepare_data.log
    log "数据准备完成"
fi

# =============================================================================
# Step 2：SFT 训练
# =============================================================================
if $SKIP_SFT; then
    warn "已跳过 SFT（--skip-sft 或已检测到产物）"
else
    info "Step 2/5 — SFT 监督微调"
    python scripts/sft_train.py \
        --config config/sft_config.yaml \
        2>&1 | tee logs/sft_train.log
    log "SFT 训练完成，产物保存至 outputs/sft/"
fi

if $ONLY_SFT; then
    log "已启用 --only-sft，流程结束"
    exit 0
fi

# =============================================================================
# Step 3：DPO 训练
# =============================================================================
if $SKIP_DPO; then
    warn "已跳过 DPO（--skip-dpo 或已检测到产物）"
else
    info "Step 3/5 — DPO 偏好优化"
    python scripts/dpo_train.py \
        --config config/dpo_config.yaml \
        2>&1 | tee logs/dpo_train.log
    log "DPO 训练完成，产物保存至 outputs/dpo/"
fi

# =============================================================================
# Step 4a：合并 SFT adapter → 评测 SFT 单独效果
# =============================================================================
if $SKIP_EVAL; then
    warn "已跳过 SFT 单独评测（--skip-eval）"
else
    EVAL_N=200; $QUICK && EVAL_N=50

    if ! $FORCE && [[ -f "logs/gsm8k_sft.json" ]] && [[ -f "logs/bbh_sft.json" ]]; then
        warn "已检测到 SFT 评测结果，自动跳过（使用 --force 可强制重跑）"
    else
        info "Step 4a — 合并 SFT LoRA adapter → outputs/sft_merged"
        mkdir -p outputs/sft_merged
        python scripts/merge_lora.py \
            --adapter_path outputs/sft \
            --output_path  outputs/sft_merged \
            --config       config/sft_config.yaml \
            2>&1 | tee logs/merge_sft.log
        log "SFT 权重合并完成 → outputs/sft_merged/"

        info "Step 4a — GSM8K 评测（SFT only，前 $EVAL_N 条）"
        python eval/gsm8k_eval.py \
            --model_path  outputs/sft_merged \
            --max_samples "$EVAL_N" \
            --output      logs/gsm8k_sft.json \
            2>&1 | tee logs/gsm8k_sft_eval.log

        info "Step 4a — BBH 评测（SFT only，前 $EVAL_N 条）"
        python eval/bbh_eval.py \
            --model_path  outputs/sft_merged \
            --max_samples "$EVAL_N" \
            --output      logs/bbh_sft.json \
            2>&1 | tee logs/bbh_sft_eval.log

        log "SFT 单独评测完成"
    fi
fi

# =============================================================================
# Step 4b：合并 DPO adapter → 评测 DPO 单独效果
# =============================================================================
if $SKIP_EVAL; then
    warn "已跳过 DPO 单独评测（--skip-eval）"
else
    if ! $FORCE && [[ -f "logs/gsm8k_dpo.json" ]] && [[ -f "logs/bbh_dpo.json" ]]; then
        warn "已检测到 DPO 评测结果，自动跳过（使用 --force 可强制重跑）"
    else
        info "Step 4b — 合并 DPO LoRA adapter → outputs/dpo_merged"
        mkdir -p outputs/dpo_merged
        python scripts/merge_lora.py \
            --adapter_path outputs/dpo \
            --output_path  outputs/dpo_merged \
            --config       config/dpo_config.yaml \
            2>&1 | tee logs/merge_dpo.log
        log "DPO 权重合并完成 → outputs/dpo_merged/"

        info "Step 4b — GSM8K 评测（DPO only，前 $EVAL_N 条）"
        python eval/gsm8k_eval.py \
            --model_path  outputs/dpo_merged \
            --max_samples "$EVAL_N" \
            --output      logs/gsm8k_dpo.json \
            2>&1 | tee logs/gsm8k_dpo_eval.log

        info "Step 4b — BBH 评测（DPO only，前 $EVAL_N 条）"
        python eval/bbh_eval.py \
            --model_path  outputs/dpo_merged \
            --max_samples "$EVAL_N" \
            --output      logs/bbh_dpo.json \
            2>&1 | tee logs/bbh_dpo_eval.log

        log "DPO 单独评测完成"
    fi
fi

# =============================================================================
# Step 4c：合并最终 SFT+DPO 权重（DPO adapter 已在 SFT 基础上训练，即完整两阶段）
# =============================================================================
if ! $FORCE && [[ -f "outputs/merged/config.json" ]]; then
    warn "已检测到 outputs/merged/config.json，SFT+DPO 合并已完成，自动跳过（使用 --force 可强制重跑）"
else
    info "Step 4c — 合并最终 SFT+DPO 权重 → outputs/merged"
    python scripts/merge_lora.py \
        --adapter_path outputs/dpo \
        --output_path  outputs/merged \
        2>&1 | tee logs/merge_lora.log
    log "最终权重合并完成 → outputs/merged/"
fi

# =============================================================================
# Step 5：最终评测（SFT+DPO 完整两阶段）+ 四行对比表
# =============================================================================
if $SKIP_EVAL; then
    warn "已跳过最终评测（--skip-eval）"
else
    EVAL_N=200; $QUICK && EVAL_N=50

    info "Step 5 — GSM8K 评测（SFT+DPO，前 $EVAL_N 条）"
    if ! $FORCE && [[ -f "logs/gsm8k_result.json" ]]; then
        warn "logs/gsm8k_result.json 已存在，跳过（使用 --force 可强制重跑）"
    else
        python eval/gsm8k_eval.py \
            --model_path  outputs/merged \
            --max_samples "$EVAL_N" \
            --output      logs/gsm8k_result.json \
            2>&1 | tee logs/gsm8k_eval.log
    fi

    info "Step 5 — BBH 评测（SFT+DPO，前 $EVAL_N 条）"
    if ! $FORCE && [[ -f "logs/bbh_result.json" ]]; then
        warn "logs/bbh_result.json 已存在，跳过（使用 --force 可强制重跑）"
    else
        python eval/bbh_eval.py \
            --model_path  outputs/merged \
            --max_samples "$EVAL_N" \
            --output      logs/bbh_result.json \
            2>&1 | tee logs/bbh_eval.log
    fi

    # ── 四行对比表 ──────────────────────────────────────────────────────────
    python - <<'EOF'
import json, os

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

base_gsm = load("logs/gsm8k_baseline.json")
sft_gsm  = load("logs/gsm8k_sft.json")
dpo_gsm  = load("logs/gsm8k_dpo.json")
ft_gsm   = load("logs/gsm8k_result.json")

base_bbh = load("logs/bbh_baseline.json")
sft_bbh  = load("logs/bbh_sft.json")
dpo_bbh  = load("logs/bbh_dpo.json")
ft_bbh   = load("logs/bbh_result.json")

def acc(r):
    return r["accuracy"] if r else None

def fmt(r):
    a = acc(r)
    return f"{a:>7.2%}" if a is not None else "   N/A "

def delta(r, base):
    a, b = acc(r), acc(base)
    if a is None or b is None:
        return "   N/A "
    d = a - b
    return f"{d:>+7.2%}"

W = 30
print()
print("=" * 66)
print(f"  {'模型':<{W}} {'GSM8K':>9}  {'BBH':>9}")
print("=" * 66)
print(f"  {'Baseline (原始模型)':<{W}} {fmt(base_gsm)}  {fmt(base_bbh)}")
print(f"  {'SFT only':<{W}} {fmt(sft_gsm)}  {fmt(sft_bbh)}")
print(f"  {'DPO only (基于SFT)':<{W}} {fmt(dpo_gsm)}  {fmt(dpo_bbh)}")
print(f"  {'SFT + DPO':<{W}} {fmt(ft_gsm)}  {fmt(ft_bbh)}")
print("-" * 66)
print(f"  {'Δ vs Baseline (SFT only)':<{W}} {delta(sft_gsm,base_gsm)}  {delta(sft_bbh,base_bbh)}")
print(f"  {'Δ vs Baseline (DPO only)':<{W}} {delta(dpo_gsm,base_gsm)}  {delta(dpo_bbh,base_bbh)}")
print(f"  {'Δ vs Baseline (SFT+DPO)':<{W}} {delta(ft_gsm, base_gsm)}  {delta(ft_bbh, base_bbh)}")
print("=" * 66)

# 更新 compare_metrics.json
metrics = {}
for k, v in [
    ("baseline_gsm8k", base_gsm), ("sft_gsm8k", sft_gsm),
    ("dpo_gsm8k", dpo_gsm),       ("finetuned_gsm8k", ft_gsm),
    ("baseline_bbh",  base_bbh),  ("sft_bbh",  sft_bbh),
    ("dpo_bbh",  dpo_bbh),        ("finetuned_bbh",  ft_bbh),
]:
    if v:
        metrics[k] = v["accuracy"]
with open("logs/compare_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
EOF
fi

# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  🎉 全部流程完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "  SFT adapter      : outputs/sft/"
echo "  SFT merged       : outputs/sft_merged/"
echo "  DPO adapter      : outputs/dpo/"
echo "  DPO merged       : outputs/dpo_merged/"
echo "  SFT+DPO merged   : outputs/merged/"
echo "  评测日志         : logs/"
echo ""
