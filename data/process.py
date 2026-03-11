import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


DEFAULT_INSTRUCTION = "请先进行清晰的逐步推理，再给出最终答案。"


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])


def split_gsm8k_answer(answer: str) -> Dict[str, str]:
    # GSM8K 常见格式：... #### 42
    parts = answer.split("####")
    if len(parts) == 1:
        final_num = re.findall(r"-?\d+(?:\.\d+)?", answer)
        final = final_num[-1] if final_num else answer.strip()
        return {"cot": answer.strip(), "final": final}
    cot = parts[0].strip()
    final = parts[-1].strip()
    return {"cot": cot, "final": final}


def to_alpaca(records: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for x in records:
        # NuminaMath-CoT 格式: problem + solution
        if "problem" in x and "solution" in x:
            out.append(
                {
                    "instruction": DEFAULT_INSTRUCTION,
                    "input": str(x["problem"]).strip(),
                    "output": str(x["solution"]).strip(),
                }
            )
        # Magpie-Reasoning 格式: instruction + response
        elif "instruction" in x and "response" in x:
            out.append(
                {
                    "instruction": str(x["instruction"]).strip(),
                    "input": "",
                    "output": str(x["response"]).strip(),
                }
            )
        # GSM8K 格式: question + answer (含 ####)
        elif "question" in x and "answer" in x:
            parsed = split_gsm8k_answer(str(x["answer"]))
            out.append(
                {
                    "instruction": DEFAULT_INSTRUCTION,
                    "input": str(x["question"]).strip(),
                    "output": f"推理过程：\n{parsed['cot']}\n\n最终答案：{parsed['final']}",
                }
            )
        # 通用 Alpaca 格式: instruction + output
        elif "instruction" in x and "output" in x:
            out.append(
                {
                    "instruction": str(x.get("instruction", "")).strip(),
                    "input": str(x.get("input", "")).strip(),
                    "output": str(x["output"]).strip(),
                }
            )
        elif "prompt" in x and "response" in x:
            out.append(
                {
                    "instruction": DEFAULT_INSTRUCTION,
                    "input": str(x["prompt"]).strip(),
                    "output": str(x["response"]).strip(),
                }
            )
    return out


def to_sharegpt_pairs(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将偏好数据转为 DPO 常见字段：prompt/chosen/rejected。"""
    out = []
    for x in records:
        prompt = x.get("prompt") or x.get("question") or x.get("input")
        chosen = x.get("chosen") or x.get("preferred") or x.get("response_a")
        rejected = x.get("rejected") or x.get("dispreferred") or x.get("response_b")
        if prompt and chosen and rejected:
            out.append(
                {
                    "prompt": str(prompt).strip(),
                    "chosen": str(chosen).strip(),
                    "rejected": str(rejected).strip(),
                }
            )
    return out


def save_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="数据格式转换脚本")
    parser.add_argument("--input", required=True, help="输入数据文件（json/jsonl）")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument(
        "--format",
        choices=["alpaca", "dpo"],
        required=True,
        help="输出格式：alpaca（SFT）或 dpo（偏好对）",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    records = load_records(in_path)
    converted = to_alpaca(records) if args.format == "alpaca" else to_sharegpt_pairs(records)
    save_json(out_path, converted)
    print(f"转换完成：{len(converted)} 条 -> {out_path}")


if __name__ == "__main__":
    main()
