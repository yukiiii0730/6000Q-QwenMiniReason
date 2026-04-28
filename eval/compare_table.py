"""结果汇总表（v4）

【v4 改进】
1. 加入 MATH-500 列（与 GSM8K 同级主指标）
2. 大模型 baseline（7B/14B）优先用 published_baselines.json 的官方公开值
   （Qwen2.5 Tech Report），自跑结果作为 sanity check 标注
3. 输出 Markdown 表格（便于贴 LaTeX 报告）

数据源优先级：
  Self-tested  (logs/{gsm8k,math,bbh}_<tag>.json)
  Published    (eval/published_baselines.json)
  N/A
"""
from __future__ import annotations

import json
import os
from pathlib import Path

PUBLISHED_BASELINES_PATH = Path("eval/published_baselines.json")


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def acc(r):
    if r is None:
        return None
    return r.get("accuracy")


def macro(r):
    if r is None:
        return None
    return r.get("macro_avg_accuracy") or r.get("accuracy")


def fmt_pct(v, width=7):
    if v is None:
        return f"{'N/A':>{width}}"
    return f"{v*100:>{width-1}.2f}%"


def fmt_delta(v):
    if v is None:
        return "  —    "
    sign = "+" if v >= 0 else "−"
    return f" {sign}{abs(v)*100:5.2f}pp"


def load_published_for(model_id: str) -> dict:
    """从 published_baselines.json 读官方公开值。"""
    if not PUBLISHED_BASELINES_PATH.exists():
        return {}
    try:
        with PUBLISHED_BASELINES_PATH.open() as f:
            data = json.load(f)
        return data.get(model_id, {}) or {}
    except Exception:
        return {}


def published_acc(model_id: str, key: str):
    return load_published_for(model_id).get(key)


# === 收集自跑结果 ===
selfgsm = {
    "baseline_1.5b": load_json("logs/gsm8k_baseline.json"),
    "qwen25_7b":     load_json("logs/gsm8k_qwen25_7b.json") or load_json("logs/gsm8k_qwen25_7b_sanity.json"),
    "qwen25_14b":    load_json("logs/gsm8k_qwen25_14b.json") or load_json("logs/gsm8k_qwen25_14b_sanity.json"),
    "sft":           load_json("logs/gsm8k_sft.json"),
    "sft_dpo":       load_json("logs/gsm8k_result.json"),
}
selfmath = {
    "baseline_1.5b": load_json("logs/math_baseline.json"),
    "qwen25_7b":     load_json("logs/math_qwen25_7b.json") or load_json("logs/math_qwen25_7b_sanity.json"),
    "qwen25_14b":    load_json("logs/math_qwen25_14b.json") or load_json("logs/math_qwen25_14b_sanity.json"),
    "sft":           load_json("logs/math_sft.json"),
    "sft_dpo":       load_json("logs/math_result.json"),
}
selfbbh = {
    "baseline_1.5b": load_json("logs/bbh_baseline.json"),
    "qwen25_7b":     load_json("logs/bbh_qwen25_7b.json") or load_json("logs/bbh_qwen25_7b_sanity.json"),
    "qwen25_14b":    load_json("logs/bbh_qwen25_14b.json") or load_json("logs/bbh_qwen25_14b_sanity.json"),
    "sft":           load_json("logs/bbh_sft.json"),
    "sft_dpo":       load_json("logs/bbh_result.json"),
}


def best(self_val, model_id: str, pub_key: str) -> tuple[float | None, str]:
    """返回 (value, source)：自跑结果优先，否则用官方公开。"""
    if self_val is not None:
        return self_val, "self"
    pub = published_acc(model_id, pub_key)
    if pub is not None:
        return pub, "published"
    return None, "na"


# === 行：(label, model_id, gsm_self, math_self, bbh_self) ===
ROWS = [
    ("Qwen2.5-1.5B-Instruct (Baseline)", "Qwen/Qwen2.5-1.5B-Instruct",
     acc(selfgsm["baseline_1.5b"]), acc(selfmath["baseline_1.5b"]), macro(selfbbh["baseline_1.5b"])),
    ("Qwen2.5-7B-Instruct (官方/sanity)", "Qwen/Qwen2.5-7B-Instruct",
     acc(selfgsm["qwen25_7b"]), acc(selfmath["qwen25_7b"]), macro(selfbbh["qwen25_7b"])),
    ("Qwen2.5-14B-Instruct (官方/sanity)", "Qwen/Qwen2.5-14B-Instruct",
     acc(selfgsm["qwen25_14b"]), acc(selfmath["qwen25_14b"]), macro(selfbbh["qwen25_14b"])),
    ("我们 SFT only", "ours_sft",
     acc(selfgsm["sft"]), acc(selfmath["sft"]), macro(selfbbh["sft"])),
    ("我们 SFT + DPO", "ours_sft_dpo",
     acc(selfgsm["sft_dpo"]), acc(selfmath["sft_dpo"]), macro(selfbbh["sft_dpo"])),
]

print()
print("=" * 90)
print(f"  {'Model':<38} {'GSM8K':>9} {'MATH-500':>11} {'BBH':>9}  {'Source':>10}")
print("=" * 90)

base_gsm = base_math = base_bbh = None
table_rows = []
for label, mid, gs, ma, bb in ROWS:
    g_val, g_src = best(gs, mid, "GSM8K")
    m_val, m_src = best(ma, mid, "MATH" if mid != "ours_sft" else "MATH-500")
    b_val, b_src = best(bb, mid, "BBH-macro")
    src_tag = "/".join(set([g_src, m_src, b_src]) - {"na"}) or "—"
    print(f"  {label:<38} {fmt_pct(g_val, 9)} {fmt_pct(m_val, 11)} {fmt_pct(b_val, 9)}   [{src_tag:>8}]")
    table_rows.append({"label": label, "model_id": mid,
                       "gsm8k": g_val, "math500": m_val, "bbh_macro": b_val,
                       "source": {"gsm8k": g_src, "math500": m_src, "bbh": b_src}})
    if mid == "Qwen/Qwen2.5-1.5B-Instruct":
        base_gsm, base_math, base_bbh = g_val, m_val, b_val

print("-" * 90)
sft = next((r for r in table_rows if r["model_id"] == "ours_sft"), None)
ftd = next((r for r in table_rows if r["model_id"] == "ours_sft_dpo"), None)
if sft and base_gsm is not None:
    dgs = (sft["gsm8k"] - base_gsm) if sft["gsm8k"] is not None else None
    dma = (sft["math500"] - base_math) if (sft["math500"] is not None and base_math is not None) else None
    dbb = (sft["bbh_macro"] - base_bbh) if (sft["bbh_macro"] is not None and base_bbh is not None) else None
    print(f"  {'Δ SFT vs Baseline':<38} {fmt_delta(dgs):>9} {fmt_delta(dma):>11} {fmt_delta(dbb):>9}")
if ftd and base_gsm is not None:
    dgs = (ftd["gsm8k"] - base_gsm) if ftd["gsm8k"] is not None else None
    dma = (ftd["math500"] - base_math) if (ftd["math500"] is not None and base_math is not None) else None
    dbb = (ftd["bbh_macro"] - base_bbh) if (ftd["bbh_macro"] is not None and base_bbh is not None) else None
    print(f"  {'Δ SFT+DPO vs Baseline':<38} {fmt_delta(dgs):>9} {fmt_delta(dma):>11} {fmt_delta(dbb):>9}")
print("=" * 90)
print("[Source]: self = 自跑全量；published = Qwen2.5 Tech Report (arXiv:2412.15115)")
print("        sanity check 文件名 *_sanity 表示只跑 50 题验证 pipeline 没崩")

# === 写出 Markdown 表 ===
md = ["# Comparison Table (auto-generated)\n",
      "| Model | GSM8K | MATH-500 | BBH-macro | Source |",
      "|---|---|---|---|---|"]
for r in table_rows:
    src = "/".join(sorted({v for v in r["source"].values()} - {"na"})) or "—"
    md.append(f"| {r['label']} | {fmt_pct(r['gsm8k'], 6)} | {fmt_pct(r['math500'], 6)} | "
              f"{fmt_pct(r['bbh_macro'], 6)} | {src} |")

Path("logs").mkdir(exist_ok=True)
Path("logs/compare_table.md").write_text("\n".join(md), encoding="utf-8")
with Path("logs/compare_metrics.json").open("w", encoding="utf-8") as f:
    json.dump({"rows": table_rows}, f, ensure_ascii=False, indent=2)
print(f"\n📝 已写出 logs/compare_table.md 和 logs/compare_metrics.json")
