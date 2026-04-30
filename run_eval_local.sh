#!/usr/bin/env bash
# =============================================================================
# run_eval_local.sh — 在 Mac / 本机 CPU·MPS·NVIDIA 上跑与 run_gpu.sh 一致的本地评测
#
# 典型流程（与 Colab/ECS 解耦）：
#   1) 云端：bash run_gpu.sh --skip-eval    # 只训练 + 合并，不上全量评测
#   2) 将 outputs/（含 sft_merged、merged）同步到本机项目根目录
#   3) 本机：bash run_eval_local.sh
#
# 先自检：python3 scripts/check_eval_env.py
#
# 用法：
#   bash run_eval_local.sh                  # 与 run_gpu 默认相同：评 SFT 合并 + 评最终合并
#   bash run_eval_local.sh --quick          # 小样本（同 run_gpu --quick）
#   bash run_eval_local.sh --load-in-4bit   # 仅 NVIDIA+CUDA；Mac 请去掉此参数，用半精度+MPS
#   bash run_eval_local.sh --skip-sft-eval  # 只评 outputs/merged
#   bash run_eval_local.sh --skip-final-eval  # 只评 outputs/sft_merged
#   bash run_eval_local.sh --eval_n 100
# =============================================================================
set -euo pipefail

if [[ -f ".env" ]]; then
    set -a; source .env; set +a
fi

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${CYAN}[run_eval_local] $*${NC}"; }
ok()   { echo -e "${GREEN}[run_eval_local] $*${NC}"; }
warn() { echo -e "${YELLOW}[run_eval_local] $*${NC}"; }

QUICK=false
EVAL_N=200
SKIP_SFT_EVAL=false
SKIP_FINAL_EVAL=false
LOAD_4BIT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)            QUICK=true ;;
        --eval_n)           EVAL_N="$2"; shift ;;
        --skip-sft-eval)    SKIP_SFT_EVAL=true ;;
        --skip-final-eval)  SKIP_FINAL_EVAL=true ;;
        --load-in-4bit)     LOAD_4BIT=true ;;
        -h|--help)
            sed -n '1,30p' "$0"
            exit 0
            ;;
        *) echo "未知参数: $1（--help 查看）" >&2; exit 1 ;;
    esac
    shift
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

export TOKENIZERS_PARALLELISM=true
HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$HF_TOKEN" ]] && export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

if $QUICK; then
    _eval_n=50
    _math_n=50
    _bbh_per=20
    warn "快速模式：GSM8K=${_eval_n}, MATH=${_math_n}, BBH 27×${_bbh_per}"
else
    _eval_n=$EVAL_N
    _math_n=500
    _bbh_per=100
fi

BIT_ARGS=()
$LOAD_4BIT && BIT_ARGS+=(--load_in_4bit)

if $SKIP_SFT_EVAL && $SKIP_FINAL_EVAL; then
    echo "不能同时 --skip-sft-eval 与 --skip-final-eval" >&2
    exit 1
fi

RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RESULTS_DIR="logs/runs/eval_local_${RUN_ID}/results"
mkdir -p "$RESULTS_DIR" logs

if ! $SKIP_SFT_EVAL; then
    [[ -f "outputs/sft_merged/config.json" ]] || {
        echo "缺少 outputs/sft_merged/（请从训练机同步或先 merge）" >&2
        exit 1
    }
    info "SFT 阶段评测（GSM8K=${_eval_n}, MATH=${_math_n}, BBH 27×${_bbh_per}）"
    "$PYTHON" eval/gsm8k_eval.py \
        --model_path outputs/sft_merged \
        --max_samples "$_eval_n" \
        --badcase_output "$RESULTS_DIR/gsm8k_sft_badcases.jsonl" \
        --output "$RESULTS_DIR/gsm8k_sft.json" \
        "${BIT_ARGS[@]}"

    "$PYTHON" eval/math_eval.py \
        --model_path outputs/sft_merged \
        --max_samples "$_math_n" \
        --badcase_output "$RESULTS_DIR/math_sft_badcases.jsonl" \
        --output "$RESULTS_DIR/math_sft.json" \
        "${BIT_ARGS[@]}"

    "$PYTHON" eval/bbh_full_eval.py \
        --mode local \
        --model_path outputs/sft_merged \
        --max_samples "$_bbh_per" \
        --output_dir "$RESULTS_DIR/bbh_sft" \
        "${BIT_ARGS[@]}"

    cp -f "$RESULTS_DIR/gsm8k_sft.json" "logs/gsm8k_sft.json"
    cp -f "$RESULTS_DIR/math_sft.json"  "logs/math_sft.json"
    [[ -f "$RESULTS_DIR/bbh_sft_summary.json" ]] && \
        cp -f "$RESULTS_DIR/bbh_sft_summary.json" "logs/bbh_sft.json"
    ok "SFT 合并模型评测已写入 logs/"
fi

if ! $SKIP_FINAL_EVAL; then
    [[ -f "outputs/merged/config.json" ]] || {
        echo "缺少 outputs/merged/（请从训练机同步或先 merge SFT+DPO）" >&2
        exit 1
    }
    info "最终模型评测（GSM8K=${_eval_n}, MATH=${_math_n}, BBH 27×${_bbh_per}）"
    "$PYTHON" eval/gsm8k_eval.py \
        --model_path outputs/merged \
        --max_samples "$_eval_n" \
        --output "$RESULTS_DIR/gsm8k_result.json" \
        "${BIT_ARGS[@]}"

    "$PYTHON" eval/math_eval.py \
        --model_path outputs/merged \
        --max_samples "$_math_n" \
        --output "$RESULTS_DIR/math_result.json" \
        "${BIT_ARGS[@]}"

    "$PYTHON" eval/bbh_full_eval.py \
        --mode local \
        --model_path outputs/merged \
        --max_samples "$_bbh_per" \
        --output_dir "$RESULTS_DIR/bbh_result" \
        "${BIT_ARGS[@]}"

    cp -f "$RESULTS_DIR/gsm8k_result.json" "logs/gsm8k_result.json"
    cp -f "$RESULTS_DIR/math_result.json"  "logs/math_result.json"
    [[ -f "$RESULTS_DIR/bbh_result_summary.json" ]] && \
        cp -f "$RESULTS_DIR/bbh_result_summary.json" "logs/bbh_result.json"

    "$PYTHON" eval/compare_table.py || warn "compare_table 非致命失败（可检查 logs/*.json 是否完整）"
    ok "最终评测与对比表已更新"
fi

ok "详细 JSON 见: $RESULTS_DIR 与 logs/"
