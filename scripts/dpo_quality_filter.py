"""Step 2a: DPO 数据质量清洗 — 用大模型 API 检查现有 DPO 数据质量。

读取 DPO 数据（prompt/chosen/rejected），调用大模型 API 对每条数据打分：
  - chosen 是否正确？
  - rejected 是否确实错误？
  - prompt 是否清晰？
过滤低质量样本，输出高质量子集。

用法：
    # 用 OpenAI-compatible API（如 Qwen API / vLLM / 本地部署）
    python scripts/dpo_quality_filter.py \
        --input data/processed/dpo_train.json \
        --output data/processed/dpo_filtered.json \
        --api_base_url https://api.openai.com/v1 \
        --api_key sk-xxx \
        --model gpt-4o-mini \
        --max_samples 500

    # 不用 API，仅用规则过滤（快速版）
    python scripts/dpo_quality_filter.py \
        --input data/processed/dpo_train.json \
        --output data/processed/dpo_filtered.json \
        --rule_only
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


# ── 规则过滤（不需要 API）──────────────────────────────────────

def rule_filter(row: dict) -> dict:
    """基于规则的快速质量检查，返回 {pass: bool, reasons: []}"""
    prompt = str(row.get("prompt", "")).strip()
    chosen = str(row.get("chosen", "")).strip()
    rejected = str(row.get("rejected", "")).strip()
    reasons = []

    # 长度检查
    if len(prompt) < 20:
        reasons.append("prompt 太短")
    if len(chosen) < 40:
        reasons.append("chosen 太短")
    if len(rejected) < 20:
        reasons.append("rejected 太短")

    # chosen 和 rejected 不能一样
    if chosen == rejected:
        reasons.append("chosen == rejected")

    # chosen 应该比 rejected 更长（通常正确答案有更详细的推理）
    if len(chosen) < len(rejected) * 0.3:
        reasons.append("chosen 比 rejected 短太多")

    # 检查是否包含推理痕迹
    reasoning_patterns = r"(因此|所以|先|再|最后|therefore|hence|step|answer|=)"
    if not re.search(reasoning_patterns, chosen, re.IGNORECASE):
        reasons.append("chosen 缺少推理痕迹")

    # 检查是否包含数字（数学题应该有数字）
    if not re.search(r"\d", chosen):
        reasons.append("chosen 没有数字")

    # 明显低质量
    bad_patterns = ["as an ai", "i can't", "i cannot", "抱歉", "无法"]
    for bad in bad_patterns:
        if bad in chosen.lower():
            reasons.append(f"chosen 含拒绝回复: {bad}")
            break

    return {"pass": len(reasons) == 0, "reasons": reasons}


# ── API 质量检查 ───────────────────────────────────────────────

QUALITY_CHECK_PROMPT = """请评估以下数学题的 DPO 训练数据质量。

【题目】
{prompt}

【正确答案 (chosen)】
{chosen}

【错误答案 (rejected)】
{rejected}

请用 JSON 格式回答：
{{
  "chosen_correct": true/false,     // chosen 的答案是否正确
  "rejected_wrong": true/false,     // rejected 的答案是否确实错误
  "prompt_clear": true/false,       // 题目是否清晰完整
  "quality_score": 1-5,             // 整体质量 1=很差 5=很好
  "reason": "简短说明"
}}
只输出 JSON，不要其他内容。"""


def call_api(base_url: str, api_key: str, model: str, prompt: str,
             timeout: int = 60, max_retries: int = 3) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 256,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    for i in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, Exception) as e:
            if i < max_retries - 1:
                time.sleep(1.5 * (2 ** i))
            else:
                return ""


def api_filter(row: dict, base_url: str, api_key: str, model: str) -> dict:
    """调用大模型 API 检查数据质量"""
    prompt_text = QUALITY_CHECK_PROMPT.format(
        prompt=row["prompt"][:500],
        chosen=row["chosen"][:800],
        rejected=row["rejected"][:800],
    )
    resp = call_api(base_url, api_key, model, prompt_text)

    try:
        # 提取 JSON
        match = re.search(r"\{.*\}", resp, re.DOTALL)
        if match:
            result = json.loads(match.group())
            score = result.get("quality_score", 0)
            return {
                "pass": score >= 3 and result.get("chosen_correct", False),
                "score": score,
                "detail": result,
            }
    except (json.JSONDecodeError, Exception):
        pass

    return {"pass": True, "score": 0, "detail": {"reason": "API 解析失败，保留"}}


# ── 主流程 ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 2a: DPO 数据质量清洗")
    parser.add_argument("--input", required=True, help="输入 DPO JSON 文件")
    parser.add_argument("--output", required=True, help="输出过滤后的 JSON")
    parser.add_argument("--rejected_output", default="", help="被过滤掉的数据（可选）")
    parser.add_argument("--rule_only", action="store_true", help="仅用规则过滤，不调 API")
    parser.add_argument("--api_base_url", default="")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max_samples", type=int, default=0, help="最多检查多少条（0=全部）")
    parser.add_argument("--min_score", type=int, default=3, help="API 模式下的最低分")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        rows = json.load(f)
    print(f"📥 加载 {len(rows)} 条 DPO 数据")

    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    kept, dropped = [], []
    for i, row in enumerate(rows):
        # 先过规则
        rule_result = rule_filter(row)
        if not rule_result["pass"]:
            row["_filter_reason"] = rule_result["reasons"]
            dropped.append(row)
            continue

        # 再过 API（如果启用）
        if not args.rule_only and args.api_base_url and args.api_key:
            api_result = api_filter(row, args.api_base_url, args.api_key, args.model)
            if not api_result["pass"]:
                row["_filter_reason"] = api_result.get("detail", {}).get("reason", "低分")
                dropped.append(row)
                continue

        kept.append(row)

        if (i + 1) % 100 == 0:
            print(f"  进度 {i+1}/{len(rows)}，保留 {len(kept)}，过滤 {len(dropped)}")

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    if args.rejected_output:
        with open(args.rejected_output, "w", encoding="utf-8") as f:
            json.dump(dropped, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 清洗完成:")
    print(f"   原始: {len(rows)} 条")
    print(f"   保留: {len(kept)} 条 → {args.output}")
    print(f"   过滤: {len(dropped)} 条")


if __name__ == "__main__":
    main()
