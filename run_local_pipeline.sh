#!/usr/bin/env bash
# =============================================================================
# run_local_pipeline.sh — 本地 L3-L6 全流程一键脚本
#
# L3: 错误分类（classify_errors.py，调 qwen-flash API，~¥0.5）
# L4: 生成 Error-Type-Targeted DPO 数据（build_targeted_dpo.py，调 qwen2.5-72b，~¥5-10）
# L5: 统计显著性检验（stats_significance.py）
# L6: 可视化 + 对比表更新
#
# 用法：
#   bash run_local_pipeline.sh              # 跑 L3+L4+L5+L6
#   bash run_local_pipeline.sh --skip-l3    # 已有 results/errors/sft，跳 L3
#   bash run_local_pipeline.sh --skip-l4    # 已有 targeted DPO 数据，跳 L4
#   bash run_local_pipeline.sh --l5-only    # 只跑统计显著性
#   bash run_local_pipeline.sh --l6-only    # 只跑可视化
#   bash run_local_pipeline.sh --l4-per-type-n 300  # 每类生成 300 条（默认 200）
#
# 前提：
#   - .env 含 DASHSCOPE_API_KEY
#   - gpu/logs/runs/gpu_*/results/gsm8k_sft_badcases.jsonl 存在（L3 输入）
#   - logs/*.json 评测结果齐全（L5/L6 输入）
# =============================================================================
set -euo pipefail

if [[ -f ".env" ]]; then set -a; source .env; set +a; fi

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[pipeline] $*${NC}"; }
ok()   { echo -e "${GREEN}[pipeline] ✓ $*${NC}"; }
warn() { echo -e "${YELLOW}[pipeline] ⚠ $*${NC}"; }
err()  { echo -e "${RED}[pipeline] ✗ $*${NC}" >&2; }

SKIP_L3=false
SKIP_L4=false
L5_ONLY=false
L6_ONLY=false
PER_TYPE_N=200

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-l3)          SKIP_L3=true ;;
        --skip-l4)          SKIP_L4=true ;;
        --l5-only)          L5_ONLY=true; SKIP_L3=true; SKIP_L4=true ;;
        --l6-only)          L6_ONLY=true; SKIP_L3=true; SKIP_L4=true ;;
        --l4-per-type-n)    PER_TYPE_N="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PROJECT_DIR}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

# ── 检查 API KEY ───────────────────────────────────────────────────────────────
if ! $SKIP_L3 && ! $SKIP_L4; then
    if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
        err "DASHSCOPE_API_KEY 未设置，L3/L4 需要调用 API"
        err "请在 .env 中添加：DASHSCOPE_API_KEY=sk-xxx"
        exit 1
    fi
fi

# ── L3：错误分类 ──────────────────────────────────────────────────────────────
if ! $SKIP_L3 && ! $L5_ONLY && ! $L6_ONLY; then
    info "=== L3：GSM8K badcase 错误分类（qwen-flash）==="

    # 找最新的 SFT badcase 文件
    BADCASE_FILE=""
    # 优先用 GPU run 的 badcase
    for f in gpu/logs/runs/*/results/gsm8k_sft_badcases.jsonl; do
        BADCASE_FILE="$f"
    done
    # fallback 到普通 logs
    if [[ -z "$BADCASE_FILE" ]] || [[ ! -f "$BADCASE_FILE" ]]; then
        for f in logs/runs/*/results/gsm8k_sft_badcases.jsonl; do
            BADCASE_FILE="$f"
        done
    fi

    if [[ -z "$BADCASE_FILE" ]] || [[ ! -f "$BADCASE_FILE" ]]; then
        warn "找不到 gsm8k_sft_badcases.jsonl"
        warn "请先运行：bash run_eval_expanded.sh --skip-final"
        warn "（会评测 outputs/sft_merged 并生成 badcase 文件）"
        SKIP_L3=true
    else
        info "使用 badcase 文件：$BADCASE_FILE"
        BADCASE_COUNT=$(wc -l < "$BADCASE_FILE")
        info "共 $BADCASE_COUNT 条 badcase"

        mkdir -p results/errors/sft
        "$PYTHON" scripts/classify_errors.py \
            --badcase_jsonl "$BADCASE_FILE" \
            --output_dir results/errors/sft \
            --workers 4
        ok "L3 完成 → results/errors/sft/"
    fi
fi

# ── L4：生成 Targeted DPO 数据 ────────────────────────────────────────────────
if ! $SKIP_L4 && ! $L5_ONLY && ! $L6_ONLY; then
    info "=== L4：生成 Error-Type-Targeted DPO 数据（qwen2.5-72b）==="

    if [[ ! -d "results/errors/sft/by_type" ]]; then
        if $SKIP_L3; then
            err "results/errors/sft/by_type 不存在，请先运行 L3（去掉 --skip-l3）"
            exit 1
        else
            warn "L3 似乎未完成，跳过 L4"
        fi
    else
        mkdir -p data/processed
        "$PYTHON" scripts/build_targeted_dpo.py \
            --by_type_dir results/errors/sft/by_type \
            --per_type_n "$PER_TYPE_N" \
            --tag round1 \
            --output_dir data/processed \
            --workers 4
        ok "L4 完成 → data/processed/dpo_targeted_round1.json"
        echo "      各类型数据量："
        "$PYTHON" -c "
import json, glob
for f in sorted(glob.glob('data/processed/dpo_targeted_by_type/*.json')):
    data = json.load(open(f))
    print(f'      {f.split(\"/\")[-1]}: {len(data)} 条')
total = json.load(open('data/processed/dpo_targeted_round1.json'))
print(f'      合计: {len(total)} 条')
" 2>/dev/null || true
    fi
fi

# ── L5：统计显著性检验 ────────────────────────────────────────────────────────
if ! $L6_ONLY; then
    info "=== L5：统计显著性检验（McNemar + Bootstrap CI）==="

    # 检查评测结果是否齐全
    NEEDED_FILES=(
        "logs/gsm8k_sft.json"
        "logs/gsm8k_result.json"
    )
    MISSING=false
    for f in "${NEEDED_FILES[@]}"; do
        if [[ ! -f "$f" ]]; then
            warn "缺少 $f，L5 可能不完整"
            MISSING=true
        fi
    done

    if $MISSING; then
        warn "部分评测结果缺失，L5 将跳过（先跑 run_eval_expanded.sh）"
    else
        mkdir -p results/stats
        "$PYTHON" scripts/stats_significance.py \
            --results_dir logs \
            --output_dir results/stats \
            --alpha 0.05 || warn "stats_significance 非致命失败，继续"
        ok "L5 完成 → results/stats/"
    fi
fi

# ── L6：可视化 + 对比表 ───────────────────────────────────────────────────────
info "=== L6：可视化 + 对比表更新 ==="

# 对比表
"$PYTHON" eval/compare_table.py || warn "compare_table 非致命失败"

# 可视化（雷达图 + 错误分布 + ablation 条形图）
mkdir -p eval/figures
"$PYTHON" eval/visualize.py \
    --metrics_json logs/compare_metrics.json \
    --out_dir eval/figures \
    --ablation_json results/ablation/summary_table.json 2>/dev/null || \
"$PYTHON" eval/visualize.py \
    --metrics_json logs/compare_metrics.json \
    --out_dir eval/figures || warn "visualize 非致命失败（可能缺少部分数据）"

ok "L6 完成 → eval/figures/"

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo ""
ok "=== 本地 Pipeline 完成 ==="
echo "  L3 结果：results/errors/sft/"
echo "  L4 结果：data/processed/dpo_targeted_round1.json"
echo "  L5 结果：results/stats/"
echo "  L6 结果：eval/figures/  +  logs/compare_table.md"
echo ""
echo "下一步："
echo "  1. 把 data/processed/dpo_targeted_round1.json 上传到 Google Drive"
echo "     cp data/processed/dpo_targeted_round1.json ~/Google\ Drive/Qwen-Reasoning/data/processed/"
echo "  2. 在 Colab 运行 notebooks/colab_ablation.ipynb（G4-G5 Group D）"
