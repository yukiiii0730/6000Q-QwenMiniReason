"""Step 4: Long-CoT 数据合成。

用大模型（Qwen2.5-72B 等）对 NuminaMath 中的难题生成详尽的 Step-by-step 推理链。
确保 1.5B 模型学到的是"思维链条"而非"答案模板"。

不需要 GPU，纯 API 调用。

用法：
    python scripts/synthesize_long_cot.py \
        --input data/processed/sft_train.json \
        --output data/processed/sft_long_cot.json \
        --api_base_url https://api.openai.com/v1 \
        --api_key sk-xxx \
        --model qwen2.5-72b-instruct \
        --max_samples 3000 \
        --min_input_len 100
"""

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


LONG_COT_PROMPT = """你是一个数学教师，擅长详尽的逐步推理。请对以下数学题进行非常详细的解答。

要求：
1. 将解题过程分解为清晰的小步骤，每步只做一件事
2. 每步都要解释"为什么这样做"
3. 使用"Step 1:", "Step 2:" 等编号
4. 在最后明确给出最终答案
5. 推理过程要尽量详尽，至少 5 个步骤

题目：{question}

详细解答："""


def call_api(base_url: str, api_key: str, model: str, prompt: str,
             timeout: int = 120, max_retries: int = 3) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,   # 稍微允许一些多样性
        "max_tokens": 2048,   # Long-CoT 需要更多 token
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    for i in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            if i < max_retries - 1:
                time.sleep(2 * (2 ** i))
            else:
                print(f"  ⚠️ API 失败: {e}")
                return ""


def is_good_cot(response: str) -> bool:
    """检查生成的 CoT 质量"""
    if len(response) < 200:
        return False
    # 至少有 3 个步骤标记
    steps = len(re.findall(r"(?:Step\s*\d|步骤\s*\d|第\s*\d)", response, re.IGNORECASE))
    if steps < 3:
        return False
    # 包含数字（数学题应该有）
    if not re.search(r"\d", response):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Step 4: Long-CoT 数据合成")
    parser.add_argument("--input", required=True, help="SFT 数据 JSON（取其中的数学题）")
    parser.add_argument("--output", required=True, help="输出的 Long-CoT 数据")
    parser.add_argument("--api_base_url", required=True)
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="qwen2.5-72b-instruct")
    parser.add_argument("--max_samples", type=int, default=3000, help="最多合成多少条")
    parser.add_argument("--min_input_len", type=int, default=80,
                        help="题目最短长度（过滤太简单的题）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    # 筛选有 input（数学题）且足够复杂的题目
    candidates = [
        row for row in all_data
        if row.get("input", "").strip() and len(row["input"].strip()) >= args.min_input_len
    ]
    print(f"📥 共 {len(all_data)} 条，其中 {len(candidates)} 条为复杂数学题")

    # 随机采样
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[:args.max_samples]

    # 合成 Long-CoT
    results = []
    for i, row in enumerate(candidates):
        question = row["input"].strip()
        prompt = LONG_COT_PROMPT.format(question=question)

        response = call_api(args.api_base_url, args.api_key, args.model, prompt)

        if response and is_good_cot(response):
            results.append({
                "instruction": "请对以下数学题进行详尽的逐步推理，展示完整的思考过程，最后给出答案。",
                "input": question,
                "output": response.strip(),
            })

        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(candidates)}，已合成 {len(results)} 条")

        # 控制 API 调用频率
        time.sleep(0.5)

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Long-CoT 合成完成:")
    print(f"   候选题目: {len(candidates)}")
    print(f"   有效合成: {len(results)} 条 → {args.output}")
    print(f"   平均长度: {sum(len(r['output']) for r in results) / max(len(results),1):.0f} 字符")


if __name__ == "__main__":
    main()
