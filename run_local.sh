#!/usr/bin/env bash
# =============================================================================
# run_local.sh — 本地（无 GPU）可完成的所有步骤
#
# 功能：
#   1. 数据准备    数据集下载 + 预处理 → data/processed/
#   2. API 评测    Baseline(1.5B) / 7B / 14B / 235B-Thinking 全部走 DashScope API
#                  ⬆ 不需要 GPU，纯 Python + HTTP 请求即可
#   3. 数据质量过滤  llm_quality_filter.py（qwen-flash 评分，纯 API）
#   4. Teacher CoT  build_teacher_dpo.py（qwen3-235b-thinking 生成 chosen，纯 API）
#
# GPU-only 步骤（SFT / DPO / 本地模型评测）请在 Colab/服务器上运行 run_gpu.sh
#
# 用法：
#   bash run_local.sh                 # 完整本地流程
#   bash run_local.sh --quick         # 每个数据集 500 条，快速测试
#   bash run_local.sh --force         # 忽略缓存，重新生成所有 API 评测结果
#   bash run_local.sh --skip-data     # 跳过数据下载（已有缓存）
#   bash run_local.sh --skip-api      # 跳过 API 评测（已有结果）
#   bash run_local.sh --skip-filter   # 跳过质量过滤
#   bash run_local.sh --skip-teacher  # 跳过 Teacher DPO 数据生成
#   bash run_local.sh --teacher-source <jsonl>  # 指定 student badcase 文件路径
#   bash run_local.sh --eval-235b     # 额外跑 235B-Thinking 评测（费用更高，默认跳过）
# =============================================================================
# 不使用 set -u：macOS 默认 bash 3.2 在 -u 下对空数组 "${arr[@]}" 会报 unbound variable
set -eo pipefail

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
FORCE=false
SKIP_DATA=false
SKIP_API=false
SKIP_FILTER=false
SKIP_TEACHER=false
EVAL_235B=false
TEACHER_SOURCE=""   # 已有 student badcase 文件；空=从 GSM8K seed 题
# v4 五段课程数据
GSM8K_N=7500            # Stage A: GSM8K-train（in-distribution）
OPENR1_N=10000          # Stage B1: OpenR1-Math（R1 蒸馏）
ORCA_N=15000            # Stage B2: Orca-Math-200k（GPT-4 蒸馏）
NUMINAMATH_N=8000       # Stage B3: NuminaMath-CoT（多源专家）
MAGPIE_N=3000           # Stage C: Magpie（通用推理）
DPO_N=5000
# 评测样本数
EVAL_N=200              # GSM8K 评测样本
MATH_N=500              # MATH-500 全量
BBH_N=100               # BBH 每子任务样本（27×100 = 2700）
SANITY_N=50             # 7B/14B sanity check 样本数（用公开值时只跑 50 题验证 pipeline）
SANITY_BBH_N=5          # sanity check 时 BBH 每子任务题数（27×5=135 次，够验证 pipeline）
USE_PUBLISHED_BASELINE=true  # 默认用 Qwen 官方公开值，自跑 50 题 sanity check
FILTER_KEEP_TOP=0.7     # 保留质量 top 70%
TEACHER_MAX=1500        # Teacher DPO 最大样本数

# Watchdog 配置（卡死自动重启）
WATCHDOG_IDLE="${WATCHDOG_IDLE:-180}"     # 默认 180 秒无输出即判定卡死
WATCHDOG_RETRIES="${WATCHDOG_RETRIES:-3}"
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

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)          QUICK=true ;;
        --force)          FORCE=true ;;
        --skip-data)      SKIP_DATA=true ;;
        --skip-api)       SKIP_API=true ;;
        --skip-filter)    SKIP_FILTER=true ;;
        --skip-teacher)   SKIP_TEACHER=true ;;
        --eval-235b)      EVAL_235B=true ;;
        --full-baseline)  USE_PUBLISHED_BASELINE=false ;;  # 不用公开值，自己跑全量 baseline
        --teacher-source) TEACHER_SOURCE="$2"; shift ;;
        --gsm8k_n)        GSM8K_N="$2"; shift ;;
        --openr1_n)       OPENR1_N="$2"; shift ;;
        --orca_n)         ORCA_N="$2"; shift ;;
        --numinamath_n)   NUMINAMATH_N="$2"; shift ;;
        --magpie_n)       MAGPIE_N="$2"; shift ;;
        --dpo_n)          DPO_N="$2"; shift ;;
        --eval_n)         EVAL_N="$2"; shift ;;
        --math_n)         MATH_N="$2"; shift ;;
        --bbh_n)          BBH_N="$2"; shift ;;
        --sanity_n)       SANITY_N="$2"; shift ;;
        --sanity_bbh_n)   SANITY_BBH_N="$2"; shift ;;
        *) die "未知参数: $1" ;;
    esac
    shift
done

if $QUICK; then
    GSM8K_N=500; OPENR1_N=500; ORCA_N=500; NUMINAMATH_N=500; MAGPIE_N=500
    DPO_N=500; EVAL_N=50; MATH_N=50; BBH_N=20; SANITY_N=20; SANITY_BBH_N=5; TEACHER_MAX=100
    warn "⚡ 快速测试：GSM8K=$GSM8K_N / OpenR1=$OPENR1_N / Orca=$ORCA_N / NuminaMath=$NUMINAMATH_N / Magpie=$MAGPIE_N | EVAL=$EVAL_N MATH=$MATH_N BBH=$BBH_N"
fi

# ── 切换到项目根目录 ──────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
info "项目目录：$PROJECT_DIR"

# ── 虚拟环境（本地可选；若用 conda 或系统 Python 直接注释掉此段）────────────
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    info "虚拟环境已激活"
fi

# ── TLS：macOS + 中文路径 + 坏 certifi 时 HF 常报 cacert.pem invalid path ─────
# Darwin 上只要系统根证书存在就强制使用（与 scripts/hf_ssl_env.py 策略一致）
if [[ "$(uname -s)" == "Darwin" ]] && [[ -f /etc/ssl/cert.pem ]]; then
    export SSL_CERT_FILE="/etc/ssl/cert.pem"
    export REQUESTS_CA_BUNDLE="/etc/ssl/cert.pem"
    export CURL_CA_BUNDLE="/etc/ssl/cert.pem"
    info "macOS：已 export SSL_CERT_FILE=/etc/ssl/cert.pem（HF/HTTPS 子进程）"
elif [[ -x ".venv/bin/python3" ]]; then
    if ! .venv/bin/python3 -c "import os,certifi; p=certifi.where(); raise SystemExit(0 if (p and os.path.isfile(p) and os.path.getsize(p)>4096) else 1)" 2>/dev/null; then
        if [[ -f /etc/ssl/cert.pem ]]; then
            export SSL_CERT_FILE="/etc/ssl/cert.pem"
            export REQUESTS_CA_BUNDLE="/etc/ssl/cert.pem"
            export CURL_CA_BUNDLE="/etc/ssl/cert.pem"
            warn "certifi CA 异常，已回退 /etc/ssl/cert.pem"
        fi
    fi
fi

# ── 检查 API Key ───────────────────────────────────────────────────────────────
DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"
[[ -n "$DASHSCOPE_API_KEY" ]] || die "请在 .env 中设置 DASHSCOPE_API_KEY=sk-xxx"

HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]]; then
    warn "HF_TOKEN 未设置（某些受限数据集可能无法下载，已开放数据集不影响）"
else
    export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# ── 确定 Python 解释器（优先 venv，保证有依赖；纯 bash 回退不依赖 PyYAML）────
PYTHON="${PROJECT_DIR}/.venv/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON="$(which python3 2>/dev/null || which python || echo '')"
fi

# ── 读取 benchmark_models.yaml（纯 awk，无需 PyYAML）────────────────────────
# 用法: yaml_get_model <section_key> <default>
# section_key 例: model_7b / model_14b / model_235b
yaml_get_model() {
    local section="$1" default="$2"
    local result
    result=$(awk -v sec="${section}:" '
        /^  reference_models:/ { in_ref=1 }
        in_ref && index($0, sec) { in_sec=1 }
        in_sec && /^      model:/ {
            sub(/^[^:]*: */, ""); gsub(/[[:space:]]*$/, ""); print; exit
        }
        in_sec && /^    [a-z]/ && !index($0, sec) { in_sec=0 }
    ' config/benchmark_models.yaml 2>/dev/null)
    echo "${result:-$default}"
}

MODEL_7B=$(yaml_get_model  "model_7b"  "qwen2.5-7b-instruct")
MODEL_14B=$(yaml_get_model "model_14b" "qwen2.5-14b-instruct")
MODEL_235B=$(yaml_get_model "model_235b" "qwen3-235b-a22b-thinking-2507")

# ── 日志目录 ──────────────────────────────────────────────────────────────────
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_LOG_DIR="logs/runs/local_${RUN_ID}"
RESULTS_DIR="$RUN_LOG_DIR/results"
mkdir -p "$RUN_LOG_DIR" "$RESULTS_DIR" logs data/processed

exec > >(tee -a "$RUN_LOG_DIR/run_local.log") 2>&1
info "本次本地运行日志：$RUN_LOG_DIR/run_local.log"

# ── --force：清除 API 评测缓存，强制重新生成 ──────────────────────────────────
if $FORCE; then
    warn "--force：清除 API 评测缓存 + data/processed 处理结果（HF 原始数据集缓存保留，无需重新下载）"
    deleted=0
    for f in logs/gsm8k_qwen*.json logs/math_qwen*.json logs/bbh_qwen*.json \
              data/processed/sft_train.json data/processed/dpo_train.json \
              data/processed/sft_train_filtered.json data/processed/dpo_train_filtered.json; do
        [[ -f "$f" ]] || continue
        rm -f "$f" && warn "  已删除：$f" && (( deleted++ )) || true
    done
    [[ $deleted -eq 0 ]] && warn "  （没有找到可清除的缓存文件）" || warn "  共清除 ${deleted} 个缓存文件"
fi

# ── 辅助函数 ──────────────────────────────────────────────────────────────────
result_exists() {
    local tag="$1"
    [[ -f "logs/gsm8k_${tag}.json" ]] && [[ -f "logs/bbh_${tag}.json" ]]
}

run_api_eval() {
    # $1 label, $2 model, $3 tag, $4 GSM8K n, $5 thinking?, $6 MATH n, $7 BBH n
    local label="$1" model="$2" tag="$3"
    local gsm8k_n="${4:-$EVAL_N}"
    local thinking_kind="${5:-}"
    local math_n="${6:-$MATH_N}"
    local bbh_n="${7:-$BBH_N}"
    local timeout=120
    local thinking_args=()
    if [[ "$thinking_kind" == "thinking" ]]; then
        thinking_args=(--enable_thinking --thinking_budget 4096)
        timeout=300
    fi

    info "API 评测：${label} model=${model} | GSM8K=${gsm8k_n} MATH=${math_n} BBH=${bbh_n}/task"

    if [[ ! -f "logs/gsm8k_${tag}.json" ]]; then
        watchdog_run "$RUN_LOG_DIR/gsm8k_${tag}.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
            python3 eval/gsm8k_api_eval.py \
                --api_base_url "$API_BASE_URL" \
                --api_key "$DASHSCOPE_API_KEY" \
                --model "$model" \
                --max_samples "$gsm8k_n" \
                --sampling_mode stratified \
                --seed 42 \
                --timeout "$timeout" \
                --max_retries 4 \
                "${thinking_args[@]}" \
                --output "$RESULTS_DIR/gsm8k_${tag}.json"
        cp -f "$RESULTS_DIR/gsm8k_${tag}.json" "logs/gsm8k_${tag}.json"
    fi

    if [[ ! -f "logs/math_${tag}.json" ]]; then
        watchdog_run "$RUN_LOG_DIR/math_${tag}.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
            python3 eval/math_api_eval.py \
                --api_base_url "$API_BASE_URL" \
                --api_key "$DASHSCOPE_API_KEY" \
                --model "$model" \
                --max_samples "$math_n" \
                --sampling_mode stratified \
                --seed 42 \
                --timeout "$timeout" \
                --max_retries 4 \
                --workers 4 \
                "${thinking_args[@]}" \
                --output "$RESULTS_DIR/math_${tag}.json"
        cp -f "$RESULTS_DIR/math_${tag}.json" "logs/math_${tag}.json"
    fi

    if [[ ! -f "logs/bbh_${tag}.json" ]]; then
        watchdog_run "$RUN_LOG_DIR/bbh_${tag}.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
            python3 eval/bbh_full_eval.py \
                --mode api \
                --api_base_url "$API_BASE_URL" \
                --api_key "$DASHSCOPE_API_KEY" \
                --model "$model" \
                --max_samples "$bbh_n" \
                --sampling_mode stratified \
                --seed 42 \
                --output_dir "$RESULTS_DIR/bbh_${tag}" \
                "${thinking_args[@]}"
        [[ -f "$RESULTS_DIR/bbh_${tag}_summary.json" ]] && \
            cp -f "$RESULTS_DIR/bbh_${tag}_summary.json" "logs/bbh_${tag}.json"
    fi

    log "${label} API 评测完成"
}

# =============================================================================
# Step 1：数据准备
# =============================================================================
if $SKIP_DATA; then
    warn "已跳过数据下载（--skip-data）"
else
    info "Step 1 — 数据集下载与预处理（v4 三剑客 + 锚点课程）"
    info "  Stage A  GSM8K-train         (${GSM8K_N} 条，in-distribution)"
    info "  Stage B1 OpenR1-Math         (${OPENR1_N} 条，R1 蒸馏)"
    info "  Stage B2 Orca-Math-200k      (${ORCA_N} 条，GPT-4 蒸馏)"
    info "  Stage B3 NuminaMath-CoT      (${NUMINAMATH_N} 条，多源专家)"
    info "  Stage C  Magpie-Reasoning    (${MAGPIE_N} 条，通用推理)"
    info "  DPO 兜底  distilabel-math-preference (${DPO_N} 条)"

    watchdog_run "$RUN_LOG_DIR/prepare_data.log" 600 2 -- \
        python3 scripts/prepare_data.py \
            --gsm8k_n      "$GSM8K_N" \
            --openr1_n     "$OPENR1_N" \
            --orca_n       "$ORCA_N" \
            --numinamath_n "$NUMINAMATH_N" \
            --magpie_n     "$MAGPIE_N" \
            --dpo_n        "$DPO_N"
    log "数据准备完成"
fi

# =============================================================================
# Step 2：API 评测（无需 GPU）
# =============================================================================
if $SKIP_API; then
    warn "已跳过 API 评测（--skip-api）"
else
    if $USE_PUBLISHED_BASELINE; then
        info "Step 2 — Baseline 策略：使用 Qwen 官方公开值（eval/published_baselines.json）"
        info "  对 7B/14B 只跑 ${SANITY_N} 题 sanity check 验证 pipeline 没崩"
        # sanity check：GSM8K/MATH 各 SANITY_N 题，BBH 每子任务 SANITY_BBH_N 题（27×5=135 次）
        run_api_eval "Qwen2.5-7B (sanity)"   "$MODEL_7B"  "qwen25_7b_sanity"  "$SANITY_N" "" "$SANITY_N" "$SANITY_BBH_N"
        run_api_eval "Qwen2.5-14B (sanity)"  "$MODEL_14B" "qwen25_14b_sanity" "$SANITY_N" "" "$SANITY_N" "$SANITY_BBH_N"
        info "✅ sanity check 完成；正式 baseline 表见 eval/published_baselines.json"
    else
        info "Step 2 — Baseline 策略：自跑全量（--full-baseline）"
        run_api_eval "Qwen2.5-7B-Instruct"   "$MODEL_7B"   "qwen25_7b"  "$EVAL_N" "" "$MATH_N" "$BBH_N"
        run_api_eval "Qwen2.5-14B-Instruct"  "$MODEL_14B"  "qwen25_14b" "$EVAL_N" "" "$MATH_N" "$BBH_N"
    fi

    if $EVAL_235B; then
        run_api_eval "Qwen3-235B-Thinking" "$MODEL_235B" "qwen3_235b" "$EVAL_N" "thinking" "$MATH_N" "$BBH_N"
    else
        warn "跳过 235B-Thinking 评测（贵且非主对比对象；加 --eval-235b 可开启）"
    fi

    log "API 评测全部完成"
fi

# =============================================================================
# Step 3：LLM 数据质量过滤（SFT + DPO，纯 API）
# =============================================================================
if $SKIP_FILTER; then
    warn "已跳过数据质量过滤（--skip-filter）"
else
    info "Step 3 — LLM 数据质量过滤（qwen-flash 评分，本地 CPU 即可）"

    SFT_IN="data/processed/sft_train.json"
    SFT_FILTERED="data/processed/sft_train_filtered.json"
    DPO_IN="data/processed/dpo_train.json"
    DPO_FILTERED="data/processed/dpo_train_filtered.json"

    if [[ -f "$SFT_IN" ]]; then
        watchdog_run "$RUN_LOG_DIR/sft_filter.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
            python3 scripts/llm_quality_filter.py \
                --input   "$SFT_IN" \
                --output  "$SFT_FILTERED" \
                --mode    sft \
                --keep_top_ratio "$FILTER_KEEP_TOP" \
                --workers 8
        log "SFT 质量过滤完成 → $SFT_FILTERED"
    else
        warn "SFT 数据不存在，跳过过滤（请先运行 Step 1）"
    fi

    if [[ -f "$DPO_IN" ]]; then
        watchdog_run "$RUN_LOG_DIR/dpo_filter.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
            python3 scripts/llm_quality_filter.py \
                --input   "$DPO_IN" \
                --output  "$DPO_FILTERED" \
                --mode    dpo \
                --keep_top_ratio "$FILTER_KEEP_TOP" \
                --workers 8
        log "DPO 质量过滤完成 → $DPO_FILTERED"
    else
        warn "DPO 数据不存在，跳过过滤"
    fi
fi

# =============================================================================
# Step 4：Teacher-Guided DPO 数据生成（235B-Thinking 生成 chosen，纯 API）
# =============================================================================
if $SKIP_TEACHER; then
    warn "已跳过 Teacher DPO 数据生成（--skip-teacher）"
else
    info "Step 4 — Teacher-Guided DPO 数据生成"
    info "  Teacher: qwen3-235b-a22b-thinking-2507  chosen 生成，本地 CPU 即可"

    TEACHER_DPO_OUT="data/processed/dpo_teacher_round_1.json"

    if [[ -n "$TEACHER_SOURCE" ]]; then
        info "  使用已有 student badcase 文件: $TEACHER_SOURCE"
        watchdog_run "$RUN_LOG_DIR/teacher_dpo.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
            python3 scripts/build_teacher_dpo.py \
                --rejected_jsonl "$TEACHER_SOURCE" \
                --rejected_kind  badcase \
                --output         "$TEACHER_DPO_OUT" \
                --max_samples    "$TEACHER_MAX" \
                --workers        4
    else
        info "  无 student badcase，从 GSM8K-train seed 题生成（仅 chosen，无 rejected）"
        watchdog_run "$RUN_LOG_DIR/teacher_dpo.log" "$WATCHDOG_IDLE" "$WATCHDOG_RETRIES" -- \
            python3 scripts/build_teacher_dpo.py \
                --source_dataset gsm8k \
                --source_config  main \
                --source_split   train \
                --output         "$TEACHER_DPO_OUT" \
                --max_samples    "$TEACHER_MAX" \
                --workers        4
        warn "  ⚠️  seed 模式产出没有 rejected，需要 GPU 上跑 student 后补全才能用于 DPO"
    fi

    [[ -f "$TEACHER_DPO_OUT" ]] && log "Teacher DPO 数据已生成 → $TEACHER_DPO_OUT"
fi

# =============================================================================
# 汇总
# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  ✅ 本地步骤全部完成！${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "  数据目录     : data/processed/"
echo "  API 评测结果 : logs/gsm8k_qwen25_7b.json | logs/bbh_qwen25_7b.json"
echo "                 logs/gsm8k_qwen25_14b.json | logs/bbh_qwen25_14b.json"
echo "  Teacher DPO  : data/processed/dpo_teacher_round_1.json"
echo ""
echo -e "${CYAN}  下一步 → 把 data/ 和 logs/ 同步到 GPU 服务器/Colab，运行 run_gpu.sh${NC}"
echo ""
echo "  同步命令示例（rsync 到服务器）："
echo "    rsync -avz --progress data/ user@server:/path/to/project/data/"
echo "    rsync -avz --progress logs/ user@server:/path/to/project/logs/"
echo ""
