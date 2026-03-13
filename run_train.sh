#!/usr/bin/env bash
# =============================================================================
# run_train.sh — Qwen2.5-1.5B Reasoning 两阶段微调一键训练脚本
#
# 用法：
#   bash run_train.sh                   # 完整流程（Baseline + 参考模型 + 数据 + SFT + DPO + 合并 + 评测）
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

# ── 加载 .env 文件（如果存在）─────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    set -a
    # shellcheck source=.env
    source .env
    set +a
fi

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

# ── 运行时优化环境变量（提升吞吐） ──────────────────────────────────────────
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="$(nproc)"

# ── 检查 Python ───────────────────────────────────────────────────────────────
PYTHON_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PYTHON_VER"

# ── 读取并验证 HuggingFace Token ─────────────────────────────────────────────
# 优先使用环境变量（来自 .env 或系统）；其次回退到 config/sft_config.yaml
if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN=$(python - <<'EOF'
import yaml, sys
try:
    with open("config/sft_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    t = cfg.get("hf_token", "").strip()
    print(t)
except Exception:
    print("")
EOF
)
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    die "请在 .env 中设置 HF_TOKEN=hf_xxx，或在 config/sft_config.yaml 中填写 hf_token"
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

# ── GPU 监控（每 60 秒记录一次，训练结束自动汇总）─────────────────────────────
GPU_MON_LOG="logs/gpu_usage.csv"
GPU_MON_PID=""

start_gpu_monitor() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits \
            -l 60 > "$GPU_MON_LOG" 2>/dev/null &
        GPU_MON_PID=$!
        info "GPU 监控已启动（PID=$GPU_MON_PID，日志: $GPU_MON_LOG）"
    fi
}

stop_gpu_monitor() {
    if [[ -n "${GPU_MON_PID:-}" ]] && kill -0 "$GPU_MON_PID" 2>/dev/null; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        wait "$GPU_MON_PID" 2>/dev/null || true
        info "GPU 监控已停止"
    fi
}

trap stop_gpu_monitor EXIT
start_gpu_monitor

# ── 自动检测已完成的阶段 ──────────────────────────────────────────────────────
# 若输出目录下存在 adapter_config.json，则认为该阶段已完成
stage_done() {
    local dir="$1"
    [[ -f "$dir/adapter_config.json" ]] && return 0
    return 1
}

run_eval_pair() {
    local label="$1"
    local model_path="$2"
    local tag="$3"
    local max_samples="$4"
    shift 4
    local extra_args=("$@")
    local gsm8k_json="logs/gsm8k_${tag}.json"
    local bbh_json="logs/bbh_${tag}.json"

    if ! $FORCE && [[ -f "$gsm8k_json" ]] && [[ -f "$bbh_json" ]]; then
        warn "已检测到 ${label} 评测结果，自动跳过（使用 --force 可强制重跑）"
        return 0
    fi

    info "GSM8K 评测（${label}） → ${model_path}"
    python eval/gsm8k_eval.py \
        --model_path  "$model_path" \
        --max_samples "$max_samples" \
        --sampling_mode "stratified" \
        --seed 42 \
        --output      "$gsm8k_json" \
        "${extra_args[@]}" \
        2>&1 | tee "logs/gsm8k_${tag}.log"

    info "BBH 评测（${label}） → ${model_path}"
    python eval/bbh_eval.py \
        --model_path  "$model_path" \
        --max_samples "$max_samples" \
        --sampling_mode "stratified" \
        --seed 42 \
        --output      "$bbh_json" \
        "${extra_args[@]}" \
        2>&1 | tee "logs/bbh_${tag}.log"

    log "${label} 评测完成"
}

run_eval_pair_api() {
    local label="$1"
    local api_model="$2"
    local tag="$3"
    local max_samples="$4"
    local api_base_url="$5"
    local api_key="$6"
    local timeout="$7"
    local max_retries="$8"
    local gsm8k_json="logs/gsm8k_${tag}.json"
    local bbh_json="logs/bbh_${tag}.json"

    if ! $FORCE && [[ -f "$gsm8k_json" ]] && [[ -f "$bbh_json" ]]; then
        warn "已检测到 ${label} 评测结果，自动跳过（使用 --force 可强制重跑）"
        return 0
    fi

    info "GSM8K API 评测（${label}） → ${api_model}"
    python eval/gsm8k_api_eval.py \
        --api_base_url "$api_base_url" \
        --api_key "$api_key" \
        --model "$api_model" \
        --max_samples "$max_samples" \
        --sampling_mode "stratified" \
        --seed 42 \
        --timeout "$timeout" \
        --max_retries "$max_retries" \
        --output "$gsm8k_json" \
        2>&1 | tee "logs/gsm8k_${tag}.log"

    info "BBH API 评测（${label}） → ${api_model}"
    python eval/bbh_api_eval.py \
        --api_base_url "$api_base_url" \
        --api_key "$api_key" \
        --model "$api_model" \
        --max_samples "$max_samples" \
        --sampling_mode "stratified" \
        --seed 42 \
        --timeout "$timeout" \
        --max_retries "$max_retries" \
        --output "$bbh_json" \
        2>&1 | tee "logs/bbh_${tag}.log"

    log "${label} API 评测完成"
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

BASELINE_N=50

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
else
    info "Step 0a/6 — Baseline 评测（原始模型，训练前）"
    BASE_MODEL=$(python -c "import yaml; c=yaml.safe_load(open('config/sft_config.yaml')); print(c['model_name'])")
    run_eval_pair "Baseline（原始模型）" "$BASE_MODEL" "baseline" "$BASELINE_N"
fi

# =============================================================================
# Step 0b：参考开源模型评测（7B / 14B）
# =============================================================================
if $SKIP_BASELINE; then
    warn "已跳过 7B/14B 参考模型评测（--skip-baseline）"
else
    info "Step 0b/6 — 参考开源模型 API 评测（7B / 14B）"
    API_EVAL_CFG="config/benchmark_models.yaml"
    [[ -f "$API_EVAL_CFG" ]] || die "未找到 $API_EVAL_CFG，请先创建并填写 API 配置"

    IFS=$'\t' read -r API_BASE_URL API_KEY API_MODEL_7B API_MODEL_14B API_TIMEOUT API_MAX_RETRIES <<< "$(python - <<'EOF'
import yaml
c = yaml.safe_load(open("config/benchmark_models.yaml", "r", encoding="utf-8")) or {}
api = c.get("api_evaluation", {}) or {}
refs = api.get("reference_models", {}) or {}
model_7b = (refs.get("model_7b", {}) or {}).get("model", "")
model_14b = (refs.get("model_14b", {}) or {}).get("model", "")
print("\t".join([
    str(api.get("api_base_url", "")).strip(),
    str(api.get("api_key", "")).strip(),
    str(model_7b).strip(),
    str(model_14b).strip(),
    str(api.get("timeout", 90)).strip(),
    str(api.get("max_retries", 4)).strip(),
]))
EOF
)"

    # 环境变量 DASHSCOPE_API_KEY 优先于 yaml 中的 api_key
    [[ -n "${DASHSCOPE_API_KEY:-}" ]] && API_KEY="$DASHSCOPE_API_KEY"

    [[ -n "$API_BASE_URL" ]] || die "config/benchmark_models.yaml 中 api_evaluation.api_base_url 不能为空"
    [[ -n "$API_KEY" ]] || die "请在 .env 中设置 DASHSCOPE_API_KEY=sk-xxx，或在 config/benchmark_models.yaml 中填写 api_key"
    [[ -n "$API_MODEL_7B" ]] || die "config/benchmark_models.yaml 中 api_evaluation.reference_models.model_7b.model 不能为空"
    [[ -n "$API_MODEL_14B" ]] || die "config/benchmark_models.yaml 中 api_evaluation.reference_models.model_14b.model 不能为空"

    run_eval_pair_api "7B 参考模型" "$API_MODEL_7B" "qwen25_7b" "$BASELINE_N" "$API_BASE_URL" "$API_KEY" "$API_TIMEOUT" "$API_MAX_RETRIES"
    run_eval_pair_api "14B 参考模型" "$API_MODEL_14B" "qwen25_14b" "$BASELINE_N" "$API_BASE_URL" "$API_KEY" "$API_TIMEOUT" "$API_MAX_RETRIES"
fi

# =============================================================================
# Step 1：数据准备
# =============================================================================
if $SKIP_DATA; then
    warn "已跳过数据下载（--skip-data）"
else
    info "Step 1/6 — 数据集下载与预处理"
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
    info "Step 2/6 — SFT 监督微调"
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
    info "Step 3/6 — DPO 偏好优化"
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
    EVAL_N=50

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
# Step 4b：DPO only 评测已移除
# =============================================================================
warn "已移除 DPO only 单独评测阶段"

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
# Step 5：最终评测（SFT+DPO 完整两阶段）+ 对比表
# =============================================================================
if $SKIP_EVAL; then
    warn "已跳过最终评测（--skip-eval）"
else
    EVAL_N=50

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

    # ── 对比表 ──────────────────────────────────────────────────────────────
    python eval/compare_table.py
fi

if [[ -f "$GPU_MON_LOG" ]]; then
python - <<'EOF'
import csv
from statistics import mean

path = "logs/gpu_usage.csv"
rows = []
with open(path, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    for r in reader:
        if len(r) < 7:
            continue
        try:
            rows.append({
                "gpu": float(r[1].strip()),
                "mem_util": float(r[2].strip()),
                "mem_used": float(r[3].strip()),
                "mem_total": float(r[4].strip()),
                "power": float(r[5].strip()),
                "temp": float(r[6].strip()),
            })
        except Exception:
            pass

if not rows:
    print("⚠️  GPU 监控日志为空，跳过汇总")
else:
    avg_gpu = mean(x["gpu"] for x in rows)
    p95_gpu = sorted(x["gpu"] for x in rows)[int(len(rows) * 0.95) - 1]
    avg_mem_util = mean(x["mem_util"] for x in rows)
    avg_mem_used = mean(x["mem_used"] for x in rows)
    max_mem_used = max(x["mem_used"] for x in rows)
    mem_total = rows[0]["mem_total"]
    avg_power = mean(x["power"] for x in rows)
    avg_temp = mean(x["temp"] for x in rows)

    print("\n================ GPU 资源利用汇总 ================")
    print(f"平均 GPU 利用率      : {avg_gpu:.1f}%")
    print(f"P95 GPU 利用率       : {p95_gpu:.1f}%")
    print(f"平均显存利用率       : {avg_mem_util:.1f}%")
    print(f"平均显存占用         : {avg_mem_used:.1f} MiB / {mem_total:.0f} MiB")
    print(f"峰值显存占用         : {max_mem_used:.1f} MiB / {mem_total:.0f} MiB")
    print(f"平均功耗             : {avg_power:.1f} W")
    print(f"平均温度             : {avg_temp:.1f} °C")
    print("GPU 监控日志          : logs/gpu_usage.csv")
    print("=================================================\n")
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
echo "  SFT+DPO merged   : outputs/merged/"
echo "  评测日志         : logs/"
echo ""
