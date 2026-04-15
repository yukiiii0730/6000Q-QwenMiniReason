"""Step 2b: Teacher-Guided DPO 数据构造。

流程：
  1. 用 SFT 模型对训练集题目推理，收集输出
  2. 用大模型（Teacher）对同样的题目生成正确答案
  3. SFT 模型的错误输出 → rejected，Teacher 的正确输出 → chosen
  4. 输出高质量 DPO pairs

用法：
    # 用本地 SFT 模型 + API Teacher
    python scripts/teacher_guided_dpo.py \
        --questions data/processed/sft_train.json \
        --student_model /path/to/sft_model \
        --teacher_api_url https://api.openai.com/v1 \
        --teacher_api_key sk-xxx \
        --teacher_model gpt-4o-mini \
        --output data/processed/dpo_teacher_guided.json \
        --max_samples 2000
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def call_api(base_url: str, api_key: str, model: str, prompt: str,
             timeout: int = 90, max_retries: int = 3) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 1024,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    for i in range(max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
        except Exception:
            if i < max_retries - 1:
                time.sleep(1.5 * (2 ** i))
    return ""


def extract_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def generate_with_local_model(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """用本地模型推理"""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Step 2b: Teacher-Guided DPO 构造")
    parser.add_argument("--questions", required=True, help="SFT 数据（取 instruction/input 作为题目）")
    parser.add_argument("--student_model", default="", help="本地 SFT 模型路径（用于生成 rejected）")
    parser.add_argument("--teacher_api_url", required=True, help="Teacher 大模型 API URL")
    parser.add_argument("--teacher_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--teacher_model", default="gpt-4o-mini")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 加载题目
    with open(args.questions, "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # 只取有 input 的（数学题），随机采样
    import random
    rng = random.Random(args.seed)
    questions = [q for q in all_questions if q.get("input", "").strip()]
    rng.shuffle(questions)
    questions = questions[:args.max_samples]
    print(f"📥 从 {len(all_questions)} 条中选取 {len(questions)} 道题")

    # 加载本地 Student 模型（如果指定）
    student_model, student_tokenizer = None, None
    if args.student_model:
        print(f"📥 加载 Student 模型: {args.student_model}")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        )

    # 构造 DPO pairs
    dpo_rows = []
    for i, q in enumerate(questions):
        question_text = q["input"].strip()
        math_prompt = f"请解答以下数学题，逐步推理后给出最终答案。\n题目：{question_text}\n解答："

        # Student 生成 rejected
        if student_model:
            student_answer = generate_with_local_model(student_model, student_tokenizer, math_prompt)
        else:
            # 没有本地模型，用 API 调一个小模型当 student
            student_answer = call_api(args.teacher_api_url, args.teacher_api_key,
                                       "qwen2.5-1.5b-instruct", math_prompt)

        # Teacher 生成 chosen
        teacher_answer = call_api(args.teacher_api_url, args.teacher_api_key,
                                   args.teacher_model, math_prompt)

        if not teacher_answer or not student_answer:
            continue

        # 简单检查：如果 student 和 teacher 答案不同，才作为 DPO pair
        student_num = extract_number(student_answer)
        teacher_num = extract_number(teacher_answer)

        if student_num != teacher_num and teacher_num:
            dpo_rows.append({
                "prompt": question_text,
                "chosen": teacher_answer.strip(),
                "rejected": student_answer.strip(),
            })

        if (i + 1) % 50 == 0:
            print(f"  进度 {i+1}/{len(questions)}，已构造 {len(dpo_rows)} 对")

    # 保存
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dpo_rows, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Teacher-Guided DPO 构造完成:")
    print(f"   题目数: {len(questions)}")
    print(f"   有效 pair: {len(dpo_rows)} 条 → {args.output}")


if __name__ == "__main__":
    main()
