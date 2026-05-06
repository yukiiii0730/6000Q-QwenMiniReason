"""从已有评测 JSON 中重新抽取答案，修正尾部句点等提取 bug。

用法:
  python3 scripts/recalc_eval.py                    # 仅预览差异
  python3 scripts/recalc_eval.py --write            # 覆写 JSON
  python3 scripts/recalc_eval.py --dir "logs 3"     # 指定目录
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 复用评测脚本中的提取函数
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from gsm8k_eval import extract_number as gsm8k_extract, _normalize_num
from math_eval import is_equiv as math_is_equiv


# ── 主逻辑 ──────────────────────────────────────────────────────

def is_gsm8k_file(details: list) -> bool:
    """通过 gt_raw 是否含 #### 判断是否为 GSM8K。"""
    for d in details[:5]:
        if "####" in d.get("gt_raw", ""):
            return True
    return False


def recalc_gsm8k(details: list) -> tuple[list, int]:
    """重新提取 GSM8K 答案，返回 (新 details, correct 数)。"""
    correct = 0
    for d in details:
        pred_raw = d.get("pred_raw", "")
        gt_raw = d.get("gt_raw", "")
        new_pred = _normalize_num(gsm8k_extract(pred_raw))
        new_gt = _normalize_num(gsm8k_extract(gt_raw))
        ok = new_pred == new_gt and new_pred != ""
        d["pred"] = new_pred
        d["gt"] = new_gt
        d["correct"] = ok
        correct += int(ok)
    return details, correct


def recalc_math(details: list) -> tuple[list, int]:
    """用原始 is_equiv 重新比较已有 pred/gt 字段（MATH 的 _strip_string 已含 rstrip('.')）。
    不重新从 raw 提取，避免简化版函数与原版不一致。"""
    correct = 0
    for d in details:
        pred = d.get("pred", "")
        gt = d.get("gt", "")
        ok = math_is_equiv(pred, gt)
        d["correct"] = ok
        correct += int(ok)
    return details, correct


def process_file(path: str, write: bool) -> dict | None:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details", [])
    if not details or "pred_raw" not in details[0]:
        return None

    old_acc = data.get("accuracy", 0)
    old_correct = data.get("correct", sum(1 for d in details if d.get("correct")))
    total = len(details)

    if is_gsm8k_file(details):
        new_details, new_correct = recalc_gsm8k(details)
        task = "GSM8K"
    else:
        new_details, new_correct = recalc_math(details)
        task = "MATH"

    new_acc = new_correct / max(total, 1)
    diff = new_correct - old_correct

    info = {
        "path": path,
        "task": task,
        "total": total,
        "old_acc": old_acc,
        "new_acc": new_acc,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "diff": diff,
    }

    if diff != 0 and write:
        data["accuracy"] = new_acc
        data["correct"] = new_correct
        data["details"] = new_details
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return info


def main():
    parser = argparse.ArgumentParser(description="重新提取评测答案（修正尾部句点 bug）")
    parser.add_argument("--write", action="store_true", help="覆写 JSON（默认仅预览）")
    parser.add_argument("--dir", nargs="*", default=["logs", "logs 2", "logs 3"],
                        help="要扫描的日志目录")
    args = parser.parse_args()

    results = []
    for d in args.dir:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(d, fname)
            try:
                info = process_file(path, write=args.write)
                if info:
                    results.append(info)
            except Exception as e:
                print(f"  ⚠️ {path}: {e}")

    # 打印汇总
    print(f"\n{'='*70}")
    print(f"{'文件':<45} {'任务':<6} {'旧acc':>7} {'新acc':>7} {'差值':>5}")
    print(f"{'-'*70}")
    changed = 0
    for r in results:
        marker = " ✅" if r["diff"] > 0 else (" ❌" if r["diff"] < 0 else "")
        print(f"{r['path']:<45} {r['task']:<6} {r['old_acc']:>7.1%} {r['new_acc']:>7.1%} {r['diff']:>+5d}{marker}")
        if r["diff"] != 0:
            changed += 1
    print(f"{'='*70}")
    print(f"共 {len(results)} 个文件，{changed} 个有变化")
    if not args.write and changed > 0:
        print("\n⚠️  仅预览模式，加 --write 覆写 JSON")


if __name__ == "__main__":
    main()
