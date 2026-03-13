import json
import os


def load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


base_gsm = load("logs/gsm8k_baseline.json")
ref7_gsm = load("logs/gsm8k_qwen25_7b.json")
ref14_gsm = load("logs/gsm8k_qwen25_14b.json")
sft_gsm = load("logs/gsm8k_sft.json")
ft_gsm = load("logs/gsm8k_result.json")

base_bbh = load("logs/bbh_baseline.json")
ref7_bbh = load("logs/bbh_qwen25_7b.json")
ref14_bbh = load("logs/bbh_qwen25_14b.json")
sft_bbh = load("logs/bbh_sft.json")
ft_bbh = load("logs/bbh_result.json")


def acc(r):
    return r["accuracy"] if r else None


def fmt(r):
    a = acc(r)
    return f"{a:>7.2%}" if a is not None else "   N/A "


def delta(r, base):
    a, b = acc(r), acc(base)
    if a is None or b is None:
        return "   N/A "
    d = a - b
    return f"{d:>+7.2%}"


W = 30
print()
print("=" * 66)
print(f"  {'模型':<{W}} {'GSM8K':>9}  {'BBH':>9}")
print("=" * 66)
print(f"  {'Baseline (原始模型)':<{W}} {fmt(base_gsm)}  {fmt(base_bbh)}")
print(f"  {'Qwen2.5-7B-Instruct':<{W}} {fmt(ref7_gsm)}  {fmt(ref7_bbh)}")
print(f"  {'Qwen2.5-14B-Instruct':<{W}} {fmt(ref14_gsm)}  {fmt(ref14_bbh)}")
print(f"  {'SFT only':<{W}} {fmt(sft_gsm)}  {fmt(sft_bbh)}")
print(f"  {'SFT + DPO':<{W}} {fmt(ft_gsm)}  {fmt(ft_bbh)}")
print("-" * 66)
print(f"  {'Δ vs Baseline (SFT only)':<{W}} {delta(sft_gsm, base_gsm)}  {delta(sft_bbh, base_bbh)}")
print(f"  {'Δ vs Baseline (SFT+DPO)':<{W}} {delta(ft_gsm, base_gsm)}  {delta(ft_bbh, base_bbh)}")
print("=" * 66)

metrics = {}
for k, v in [
    ("baseline_gsm8k", base_gsm),
    ("sft_gsm8k", sft_gsm),
    ("qwen25_7b_gsm8k", ref7_gsm),
    ("qwen25_14b_gsm8k", ref14_gsm),
    ("finetuned_gsm8k", ft_gsm),
    ("baseline_bbh", base_bbh),
    ("sft_bbh", sft_bbh),
    ("qwen25_7b_bbh", ref7_bbh),
    ("qwen25_14b_bbh", ref14_bbh),
    ("finetuned_bbh", ft_bbh),
]:
    if v:
        metrics[k] = v["accuracy"]

with open("logs/compare_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
