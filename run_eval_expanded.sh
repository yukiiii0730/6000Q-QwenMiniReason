#!/usr/bin/env bash
# =============================================================================
# run_eval_expanded.sh — 扩充样本量的完整本地评测脚本（L1 + L2）
#
# 样本量选择依据（95% CI）：
#   GSM8K  200 题  → ±6.9pp  (之前 50 题 ±14pp，现在 3× 更精确)
#   MATH   200 题  → ±6.9pp
#   BBH    30/task → 27×30=810 题，macro CI ±3.4pp
#
# 用法：
#   bash run_eval_expanded.sh                    # 评测 GPU 组（Group B）
#   bash run_eval_expanded.sh --group-c          # 同时评测 Colab Group C 模型
#   bash run_eval_expanded.sh --model outputs/my_model  # 指定任意模型路径
#   bash run_eval_expanded.sh --gsm8k-only       # 只跑 GSM8K（快速验证）
#   bash run_eval_expanded.sh --skip-sft         # 只评最终 merged 模型
#
# 前提：
#   - outputs/sft_merged/ 和 outputs/merged/ 存在（GPU Group B）
#   - Group C eval 需要先把 colab 模型下载到 colab/outputs/merged/
#     下载指令见脚本末尾注释
# =============================================================================
set -euo pipefail

if [[ -f ".env" ]]; then set -a; source .env; set +a; fi

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[eval] $*${NC}"; }
ok()   { echo -e "${GREEN}[eval] ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}[eval] ⚠ $*${NC}"; }
err()  { echo -e "${RED}[eval] ✗ $*${NC}" >&2; }

# ── 默认参数 ──────────────────────────────────────────────────────────────────
GSM8K_N=200        # ±6.9pp CI
MATH_N=200         # ±6.9pp CI
BBH_PER=30         # 30×27=810 题，macro CI ±3.4pp
SKIP_SFT=false
SKIP_FINAL=false
GSM8K_ONLY=false
GROUP_C=false
CUSTOM_MODEL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-sft)         SKIP_SFT=true ;;
        --skip-final)       SKIP_FINAL=true ;;
        --gsm8k-only)       GSM8K_ONLY=true ;;
        --group-c)          GROUP_C=true ;;
        --model)            CUSTOM_MODEL="$2"; shift ;;
        --gsm8k-n)          GSM8K_N="$2"; shift ;;
        --math-n)           MATH_N="$2"; shift ;;
        --bbh-per)          BBH_PER="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
export TOKENIZERS_PARALLELISM=false

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RESULTS_DIR="logs/runs/eval_expanded_${RUN_ID}/results"
mkdir -p "$RESULTS_DIR" logs

# ── 评测函数 ──────────────────────────────────────────────────────────────────
eval_model() {
    local label="$1"     # 显示名称，如 "GroupB_sft"
    local model_path="$2"
    local out_prefix="$3" # logs/ 里的前缀，如 "gsm8k_sft"

    if [[ ! -f "$model_path/config.json" ]]; then
        err "模型不存在：$model_path  — 跳过 $label"
        return 1
    fi

    info "─── 评测 [$label] → $model_path ───"

    # GSM8K
    info "[$label] GSM8K n=$GSM8K_N ..."
    "$PYTHON" eval/gsm8k_eval.py \
        --model_path "$model_path" \
        --max_samples "$GSM8K_N" \
        --sampling_mode stratified \
        --output "$RESULTS_DIR/${out_prefix}_gsm8k.json" \
        --badcase_output "$RESULTS_DIR/${out_prefix}_gsm8k_badcases.jsonl"
    cp -f "$RESULTS_DIR/${out_prefix}_gsm8k.json" "logs/${out_prefix}.json" 2>/dev/null || true
    ok "[$label] GSM8K 完成"

    if ! $GSM8K_ONLY; then
        # MATH-500
        info "[$label] MATH-500 n=$MATH_N ..."
        "$PYTHON" eval/math_eval.py \
            --model_path "$model_path" \
            --max_samples "$MATH_N" \
            --sampling_mode stratified \
            --output "$RESULTS_DIR/${out_prefix}_math.json" \
            --badcase_output "$RESULTS_DIR/${out_prefix}_math_badcases.jsonl"
        cp -f "$RESULTS_DIR/${out_prefix}_math.json" "logs/math_${out_prefix##*_}.json" 2>/dev/null || true
        ok "[$label] MATH-500 完成"

        # BBH-27
        info "[$label] BBH-27 ($BBH_PER/task × 27) ..."
        "$PYTHON" eval/bbh_full_eval.py \
            --mode local \
            --model_path "$model_path" \
            --max_samples "$BBH_PER" \
            --output_dir "$RESULTS_DIR/${out_prefix}_bbh"
        local bbh_summary="$RESULTS_DIR/${out_prefix}_bbh/${out_prefix}_bbh_summary.json"
        [[ -f "$bbh_summary" ]] || bbh_summary="$(find "$RESULTS_DIR/${out_prefix}_bbh" -name '*summary*' | head -1)"
        [[ -f "$bbh_summary" ]] && cp -f "$bbh_summary" "logs/bbh_${out_prefix##*_}.json" || true
        ok "[$label] BBH-27 完成"
    fi
}

# ── Group B：GPU 模型（已有权重）──────────────────────────────────────────────
if [[ -n "$CUSTOM_MODEL" ]]; then
    eval_model "custom" "$CUSTOM_MODEL" "custom_sft"
else
    if ! $SKIP_SFT; then
        eval_model "GroupB_SFT" "outputs/sft_merged" "gsm8k_sft"
    fi
    if ! $SKIP_FINAL; then
        eval_model "GroupB_DPO" "outputs/merged" "gsm8k_result"
    fi
fi

# ── Group C：Colab Teacher-DPO 模型（需提前下载权重）────────────────────────
if $GROUP_C; then
    info "─── Group C（Colab Teacher-DPO）eval ───"
    COLAB_MERGED="colab/outputs/merged"
    if [[ ! -f "$COLAB_MERGED/config.json" ]] || [[ ! -f "$COLAB_MERGED/model.safetensors" ]]; then
        warn "Colab 模型权重未找到：$COLAB_MERGED"
        warn "请先从 Google Drive 下载 Colab 训练产物："
        warn "  1. 打开 Drive，找到 Qwen-Reasoning/outputs/merged/"
        warn "  2. 下载 model.safetensors + config.json 等到本地 colab/outputs/merged/"
        warn "  或用 rclone：rclone copy gdrive:Qwen-Reasoning/outputs/merged/ colab/outputs/merged/"
        warn "  或用 gdown：gdown --folder <Drive_folder_id> -O colab/outputs/merged/"
    else
        mkdir -p logs
        eval_model "GroupC_TeacherDPO" "$COLAB_MERGED" "groupc_result"
        ok "Group C 评测结果写入 logs/groupc_*.json"
    fi
fi

# ── 更新对比表 ─────────────────────────────────────────────────────────────────
info "更新对比表 ..."
"$PYTHON" eval/compare_table.py || warn "compare_table 非致命失败"

ok "全部完成！结果见 $RESULTS_DIR 和 logs/"
echo ""
echo "预计耗时参考（Apple M 系列 Mac，MPS 半精度）："
echo "  GSM8K 200题 ≈ 40-60 min"
echo "  MATH  200题 ≈ 50-70 min"
echo "  BBH   810题 ≈ 90-120 min"
echo "  合计约 3-4 小时（可 tmux 后台运行）"

# =============================================================================
# Group C 模型下载说明（如果已有 rclone 配置 gdrive）：
#
#   mkdir -p colab/outputs/merged colab/outputs/sft_merged
#   rclone copy gdrive:Qwen-Reasoning/outputs/merged/ colab/outputs/merged/ --progress
#   rclone copy gdrive:Qwen-Reasoning/outputs/sft_merged/ colab/outputs/sft_merged/ --progress
#
# 如无 rclone，可在 Colab 里打包下载：
#   !zip -r /content/drive/MyDrive/colab_merged.zip /content/6000Q-QwenMiniReason/outputs/merged/
#   然后本地解压到 colab/outputs/merged/
# =============================================================================
