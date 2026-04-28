#!/usr/bin/env bash
# =============================================================================
# run_gpu.sh — GPU 阶段训练脚本（Colab A100 / 服务器专用）
#
# 前提：本地已运行 run_local.sh，data/processed/ 和 logs/ 已同步到此服务器
#
# 用法：
#   bash run_gpu.sh                      # SFT(课程) → merge → eval → DPO → merge → eval
#   bash run_gpu.sh --quick              # 快速测试（各 50/30 步）
#   bash run_gpu.sh --skip-sft           # 跳过 SFT（已完成时）
#   bash run_gpu.sh --skip-dpo           # 跳过 DPO（仅评测 SFT）
#   bash run_gpu.sh --skip-eval          # 跳过本地模型评测
#   bash run_gpu.sh --iterative-dpo N    # 启用 N 轮 Iterative DPO（默认 0=关闭）
#   bash run_gpu.sh --use-filtered-data  # 使用本地过滤后的数据（sft_train_filtered.json）
#   bash run_gpu.sh --use-teacher-dpo    # 将 dpo_teacher_round_1.json 追加到 DPO 训练集
#   bash run_gpu.sh --force              # 强制重跑所有阶段
# =============================================================================
set -euo pipefail

# ── 加载 .env ─────────────────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    set -a; source .env; set +a
fi

# ── 颜色 ──────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✅ $1${NC}"; }
info() { echo -e "${CYAN}[$(date '+%H:%M:%S')] ➤  $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠️  $1${NC}"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ❌ $1${NC}" >&2; exit 1; }

# ── 默认参数 ──────────────────────────────────────────────────────────────────
QUICK=false
SKIP_SFT=false
SKIP_DPO=false
SKIP_EVAL=false
FORCE=false
ITERATIVE_ROUNDS=0
USE_FILTERED_DATA=false
USE_TEACHER_DPO=false
EVAL_N=200

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)             QUICK=true ;;
        --skip-sft)          SKIP_SFT=true ;;
        --skip-dpo)          SKIP_DPO=true ;;
        --skip-eval)         SKIP_EVAL=true ;;
        --force)             FORCE=true ;;
        --iterative-dpo)     ITERATIVE_ROUNDS="$2"; shift ;;
        --use-filtered-data) USE_FILTERED_DATA=true ;;
        --use-teacher-dpo)   USE_TEACHER_DPO=true ;;
        --eval_n)            EVAL_N="$2"; shift ;;
        *) die "未知参数: $1" ;;
    esac
    shift
done

$QUICK && warn "⚡ 快速测试：SFT=50 steps，DPO=30 steps，eval=50 samples"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
info "项目目录：$PROJECT_DIR"

# ── 虚拟环境 ──────────────────────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    info "虚拟环境已激活"
fi

# ── 确定 Python 解释器（venv 优先） ──────────────────────────────────────────
PYTHON="${PROJECT_DIR}/.venv/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(which python3 2>/dev/null || which python)"
fi

# ── 检查 GPU ──────────────────────────────────────────────────────────────────
"$PYTHON" - <<'EOF'
import torch, sys
if not torch.cuda.is_available():
    print("❌ 未检测到 GPU，run_gpu.sh 必须在 GPU 环境中运行", file=sys.stderr)
    sys.exit(1)
name  = torch.cuda.get_device_name(0)
vram  = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
print(f"GPU : {name}  显存: {vram} GB")
if vram < 12:
    print("⚠️  显存 < 12GB，建议减小 batch_size 或 max_seq_length")
EOF

# ── 环境变量优化 ──────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="$(nproc)"

HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ── 日志目录 ──────────────────────────────────────────────────────────────────
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="logs/runs/gpu_${RUN_ID}"
RESULTS_DIR="$RUN_LOG_DIR/results"
mkdir -p "$RUN_LOG_DIR" "$RESULTS_DIR" outputs/sft outputs/dpo outputs/merged logs

ln -sfn "runs/gpu_${RUN_ID}" logs/gpu_latest

exec > >(tee -a "$RUN_LOG_DIR/run_gpu.log") 2>&1
info "GPU 运行日志：$RUN_LOG_DIR/run_gpu.log"

# ── GPU 监控 ──────────────────────────────────────────────────────────────────
GPU_MON_LOG="$RUN_LOG_DIR/gpu_usage.csv"
GPU_MON_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits -l 60 > "$GPU_MON_LOG" 2>/dev/null &
    GPU_MON_PID=$!
fi
trap '[[ -n "${GPU_MON_PID:-}" ]] && kill "$GPU_MON_PID" 2>/dev/null || true' EXIT

# ── 数据路径决策 ──────────────────────────────────────────────────────────────
SFT_DATA="data/processed/sft_train.json"
DPO_DATA="data/processed/dpo_train.json"

if $USE_FILTERED_DATA; then
    [[ -f "data/processed/sft_train_filtered.json" ]] && SFT_DATA="data/processed/sft_train_filtered.json" \
        && info "使用质量过滤后 SFT 数据: $SFT_DATA"
    [[ -f "data/processed/dpo_train_filtered.json" ]] && DPO_DATA="data/processed/dpo_train_filtered.json" \
        && info "使用质量过滤后 DPO 数据: $DPO_DATA"
fi

if $USE_TEACHER_DPO && [[ -f "data/processed/dpo_teacher_round_1.json" ]]; then
    info "追加 Teacher-Guided DPO 数据到训练集"
    "$PYTHON" - "$DPO_DATA" <<'EOF'
import json, sys
base = json.load(open(sys.argv[1], "r", encoding="utf-8"))
teacher = json.load(open("data/processed/dpo_teacher_round_1.json", "r", encoding="utf-8"))
teacher_full = [x for x in teacher if x.get("rejected")]
combined = base + teacher_full
# 写回临时文件
out = sys.argv[1].replace(".json", "_with_teacher.json")
json.dump(combined, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"合并 DPO: {len(base)} (base) + {len(teacher_full)} (teacher) = {len(combined)} 条 → {out}")
EOF
    DPO_DATA="${DPO_DATA%.json}_with_teacher.json"
fi

# ── 配置写回：把数据路径注入 yaml（保留 stages 字段用于 SFT，但覆写 dataset_path 作为 fallback）───
if $QUICK; then
    "$PYTHON" - <<EOF
import yaml
for cfg_path, key, val, steps, max_field in [
    ("config/sft_config.yaml", "dataset_path", "$SFT_DATA", 50, "max_steps"),
    ("config/dpo_config.yaml", "dataset_path", "$DPO_DATA", 30, "max_steps"),
]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg[key] = val
    cfg.get("train", {})["max_steps"] = steps
    # quick 模式：禁用 stages（避免每段都要下载数据集），改用 dataset_path
    cfg.pop("stages", None)
    cfg.pop("datasets", None)
    cfg.pop("dataset", None)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
print("⚡ quick 模式：dataset_path 写入完毕，stages 已禁用")
EOF
else
    "$PYTHON" - <<EOF
import yaml
for cfg_path, key, val in [
    ("config/sft_config.yaml", "dataset_path", "$SFT_DATA"),
    ("config/dpo_config.yaml", "dataset_path", "$DPO_DATA"),
]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg[key] = val
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
print("📦 数据路径已写入 config（stages 保持两阶段课程）")
EOF
fi

# ── 自适应 fp16/bf16 ───────────────────────────────────────────────────────────
"$PYTHON" - <<'EOF'
import torch, yaml
supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
for path in ["config/sft_config.yaml", "config/dpo_config.yaml"]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["bf16"] = bool(supported)
    cfg["train"]["fp16"] = not bool(supported)
    # 同步到 stages
    for st in cfg.get("stages", []):
        t = st.get("train", {})
        if t:
            t["bf16"] = bool(supported)
            t["fp16"] = not bool(supported)
    with open(path, "w") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
print(f"bf16={'已启用' if supported else '不支持 → 切换 fp16'}")
EOF

stage_done() { [[ -f "$1/adapter_config.json" ]] && return 0 || return 1; }

# ── Watchdog 包装：长时命令"卡死 3 分钟即重启" ────────────────────────────────
# 用法：watchdog_run <log_path> <idle_seconds> <max_retries> -- <cmd...>
WATCHDOG_IDLE="${WATCHDOG_IDLE:-180}"     # 默认 180 秒无输出即判定卡死
WATCHDOG_RETRIES="${WATCHDOG_RETRIES:-3}" # 默认最多重试 3 次
watchdog_run() {
    local log_path="$1"; shift
    local idle="$1"; shift
    local retries="$1"; shift
    [[ "$1" == "--" ]] && shift
    python3 scripts/watchdog_run.py \
        --idle-timeout "$idle" \
        --max-retries "$retries" \
        --log "$log_path" \
        -- "$@"
}

# =============================================================================
# Step 1：SFT 两段式课程训练
# =============================================================================
if ! $FORCE && stage_done "outputs/sft" && ! $SKIP_SFT; then
    warn "outputs/sft 已有 adapter，SFT 自动跳过（--force 可强制重跑）"
    SKIP_SFT=true
fi

if $SKIP_SFT; then
    warn "已跳过 SFT"
else
    info "Step 1 — SFT 两段式课程（Stage1a MetaMath+GSM8K → Stage1b Magpie）| 卡死 ${WATCHDOG_IDLE}s 自动重启"
    watchdog_run "$RUN_LOG_DIR/sft_train.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 scripts/sft_train.py --config config/sft_config.yaml
    [[ -f "outputs/sft/trainer_log_history.jsonl" ]] && \
        cp -f "outputs/sft/trainer_log_history.jsonl" "$RUN_LOG_DIR/sft_trainer_log_history.jsonl"
    log "SFT 完成 → outputs/sft/"
fi

# =============================================================================
# Step 2：合并 SFT adapter + 评测
# =============================================================================
if $SKIP_EVAL; then
    warn "已跳过 SFT 评测（--skip-eval）"
else
    $QUICK && _eval_n=50 || _eval_n=$EVAL_N
    $QUICK && _math_n=50 || _math_n=500
    $QUICK && _bbh_per=20 || _bbh_per=100
    info "Step 2 — SFT 评测（GSM8K=${_eval_n}, MATH-500=${_math_n}, BBH 27×${_bbh_per}）"
    mkdir -p outputs/sft_merged
    watchdog_run "$RUN_LOG_DIR/merge_sft.log" 120 2 -- \
        python3 scripts/merge_lora.py \
            --adapter_path outputs/sft \
            --output_path  outputs/sft_merged

    watchdog_run "$RUN_LOG_DIR/gsm8k_sft_eval.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 eval/gsm8k_eval.py \
            --model_path  outputs/sft_merged \
            --max_samples "$_eval_n" \
            --badcase_output "$RESULTS_DIR/gsm8k_sft_badcases.jsonl" \
            --output      "$RESULTS_DIR/gsm8k_sft.json"

    watchdog_run "$RUN_LOG_DIR/math_sft_eval.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 eval/math_eval.py \
            --model_path  outputs/sft_merged \
            --max_samples "$_math_n" \
            --badcase_output "$RESULTS_DIR/math_sft_badcases.jsonl" \
            --output      "$RESULTS_DIR/math_sft.json"

    watchdog_run "$RUN_LOG_DIR/bbh_sft_eval.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 eval/bbh_full_eval.py \
            --mode local \
            --model_path  outputs/sft_merged \
            --max_samples "$_bbh_per" \
            --output_dir  "$RESULTS_DIR/bbh_sft"

    cp -f "$RESULTS_DIR/gsm8k_sft.json" "logs/gsm8k_sft.json"
    cp -f "$RESULTS_DIR/math_sft.json"  "logs/math_sft.json"
    [[ -f "$RESULTS_DIR/bbh_sft_summary.json" ]] && \
        cp -f "$RESULTS_DIR/bbh_sft_summary.json" "logs/bbh_sft.json"
    log "SFT 评测完成（GSM8K + MATH-500 + BBH 27 任务）"
fi

# =============================================================================
# Step 3：DPO 训练
# =============================================================================
if ! $FORCE && stage_done "outputs/dpo" && ! $SKIP_DPO; then
    warn "outputs/dpo 已有 adapter，DPO 自动跳过（--force 可强制重跑）"
    SKIP_DPO=true
fi

if $SKIP_DPO; then
    warn "已跳过 DPO"
else
    # 确保 SFT 已合并为完整模型，DPO base_adapter_path 必须指向 outputs/sft_merged
    if [[ ! -f outputs/sft_merged/config.json ]]; then
        info "Step 3 (pre) — outputs/sft_merged 不存在，先合并 SFT adapter"
        mkdir -p outputs/sft_merged
        python3 scripts/merge_lora.py \
            --adapter_path outputs/sft \
            --output_path  outputs/sft_merged \
            2>&1 | tee -a "$RUN_LOG_DIR/merge_sft.log"
    fi
    info "Step 3 — DPO 偏好优化（Teacher-Guided + argilla-math-preference）| 卡死 ${WATCHDOG_IDLE}s 自动重启"
    watchdog_run "$RUN_LOG_DIR/dpo_train.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 scripts/dpo_train.py --config config/dpo_config.yaml
    [[ -f "outputs/dpo/trainer_log_history.jsonl" ]] && \
        cp -f "outputs/dpo/trainer_log_history.jsonl" "$RUN_LOG_DIR/dpo_trainer_log_history.jsonl"
    log "DPO 完成 → outputs/dpo/"
fi

# =============================================================================
# Step 4：合并最终权重 + 评测
# =============================================================================
if ! $FORCE && [[ -f "outputs/merged/config.json" ]]; then
    warn "outputs/merged 已存在，跳过合并"
else
    info "Step 4 — 合并最终 SFT+DPO 权重"
    python3 scripts/merge_lora.py \
        --adapter_path outputs/dpo \
        --output_path  outputs/merged \
        2>&1 | tee "$RUN_LOG_DIR/merge_lora.log"
    log "合并完成 → outputs/merged/"
fi

if $SKIP_EVAL; then
    warn "已跳过最终评测（--skip-eval）"
else
    $QUICK && _eval_n=50 || _eval_n=$EVAL_N
    $QUICK && _math_n=50 || _math_n=500
    $QUICK && _bbh_per=20 || _bbh_per=100
    info "Step 4 — 最终评测（SFT+DPO，GSM8K=${_eval_n}, MATH-500=${_math_n}, BBH 27×${_bbh_per}）"

    watchdog_run "$RUN_LOG_DIR/gsm8k_eval.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 eval/gsm8k_eval.py \
            --model_path  outputs/merged \
            --max_samples "$_eval_n" \
            --output      "$RESULTS_DIR/gsm8k_result.json"

    watchdog_run "$RUN_LOG_DIR/math_eval.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 eval/math_eval.py \
            --model_path  outputs/merged \
            --max_samples "$_math_n" \
            --output      "$RESULTS_DIR/math_result.json"

    watchdog_run "$RUN_LOG_DIR/bbh_eval.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 eval/bbh_full_eval.py \
            --mode local \
            --model_path  outputs/merged \
            --max_samples "$_bbh_per" \
            --output_dir  "$RESULTS_DIR/bbh_result"

    cp -f "$RESULTS_DIR/gsm8k_result.json" "logs/gsm8k_result.json"
    cp -f "$RESULTS_DIR/math_result.json"  "logs/math_result.json"
    [[ -f "$RESULTS_DIR/bbh_result_summary.json" ]] && \
        cp -f "$RESULTS_DIR/bbh_result_summary.json" "logs/bbh_result.json"

    python3 eval/compare_table.py || warn "compare_table 跳过"
    log "最终评测完成"
fi

# =============================================================================
# Step 5（可选）：Iterative DPO
# =============================================================================
if [[ "$ITERATIVE_ROUNDS" -gt 0 ]]; then
    info "Step 5 — Iterative DPO（${ITERATIVE_ROUNDS} 轮，online 模式）| 卡死 ${WATCHDOG_IDLE}s 自动重启"
    watchdog_run "$RUN_LOG_DIR/iterative_dpo.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
        python3 scripts/iterative_dpo_loop.py \
            --rounds            "$ITERATIVE_ROUNDS" \
            --steps_per_round   200 \
            --base_adapter      outputs/sft \
            --base_dpo_config   config/dpo_config.yaml \
            --eval_subset       gsm8k \
            --eval_n            "$EVAL_N" \
            --mode              online
    log "Iterative DPO 完成"
else
    info "跳过 Iterative DPO（加 --iterative-dpo 2 可开启 2 轮）"
fi

# ── GPU 统计摘要 ───────────────────────────────────────────────────────────────
[[ -n "${GPU_MON_PID:-}" ]] && kill "$GPU_MON_PID" 2>/dev/null || true
if [[ -f "$GPU_MON_LOG" ]]; then
    "$PYTHON" - "$GPU_MON_LOG" <<'PY'
import csv, sys
from statistics import mean
rows = []
with open(sys.argv[1]) as f:
    for r in csv.reader(f):
        try:
            if len(r) >= 7:
                rows.append({"gpu": float(r[1]), "mem": float(r[3]), "total": float(r[4]), "pw": float(r[5]), "temp": float(r[6])})
        except Exception:
            pass
if rows:
    print(f"\nGPU 利用率均值={mean(x['gpu'] for x in rows):.1f}%  "
          f"峰值显存={max(x['mem'] for x in rows):.0f}/{rows[0]['total']:.0f}MiB  "
          f"均功耗={mean(x['pw'] for x in rows):.1f}W  均温={mean(x['temp'] for x in rows):.1f}°C")
PY
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  🎉 GPU 训练全部完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "  SFT adapter      : outputs/sft/"
echo "  SFT merged       : outputs/sft_merged/"
echo "  DPO adapter      : outputs/dpo/"
echo "  SFT+DPO merged   : outputs/merged/"
echo "  结果日志         : $RESULTS_DIR/"
echo ""
echo -e "${CYAN}  同步结果回本地：${NC}"
echo "    rsync -avz user@server:/path/to/project/logs/  ./logs/"
echo "    rsync -avz user@server:/path/to/project/outputs/merged/  ./outputs/merged/"
echo ""
