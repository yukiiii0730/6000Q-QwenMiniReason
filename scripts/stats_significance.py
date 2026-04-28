#!/usr/bin/env python3
"""统计显著性检验（v4）—— 让结果对比"可信"

【为什么必加】
v3 只报告 SFT 50% / SFT+DPO 40% 这种点估计是不严谨的。
论文级评测要求：
  1. McNemar 配对检验 —— 同一批题上两个模型的差异是否显著
  2. Bootstrap 95% CI —— 单个模型 accuracy 的置信区间
  3. Δ accuracy 的 paired-bootstrap CI —— 改进量是否显著为正

【输入】
两个 eval JSON 文件（必须是同一批题，sample_indices 对得上），每个文件含
details 列表，每条带 correct: bool 字段。

用法：
    # 单模型 95% CI
    python scripts/stats_significance.py --single results/gsm8k_sft.json

    # 两模型对比（同一批题，必须 sample_indices 一致）
    python scripts/stats_significance.py \
        --a results/gsm8k_sft.json --a_name "SFT" \
        --b results/gsm8k_result.json --b_name "SFT+DPO"
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def wilson_ci(p: float, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(corrects: list[bool], n_boot: int = 5000, seed: int = 42, alpha: float = 0.05):
    rng = random.Random(seed)
    n = len(corrects)
    if n == 0:
        return (0.0, 0.0)
    accs = []
    for _ in range(n_boot):
        sampled = [corrects[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(sampled) / n)
    accs.sort()
    lo = accs[int(alpha / 2 * n_boot)]
    hi = accs[int((1 - alpha / 2) * n_boot)]
    return (lo, hi)


def paired_bootstrap_delta(a: list[bool], b: list[bool],
                           n_boot: int = 5000, seed: int = 42, alpha: float = 0.05):
    """配对 bootstrap：返回 Δ = acc(b) - acc(a) 的 95% CI 和 p-value。"""
    if len(a) != len(b):
        raise ValueError("配对 bootstrap 要求 a 和 b 长度相同（同一批题）")
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        ai = sum(a[i] for i in idxs) / n
        bi = sum(b[i] for i in idxs) / n
        deltas.append(bi - ai)
    deltas.sort()
    lo = deltas[int(alpha / 2 * n_boot)]
    hi = deltas[int((1 - alpha / 2) * n_boot)]
    # p-value：双侧检验，H0: delta=0
    point = sum(b) / n - sum(a) / n
    if point > 0:
        p = sum(1 for d in deltas if d <= 0) * 2 / n_boot
    else:
        p = sum(1 for d in deltas if d >= 0) * 2 / n_boot
    return point, (lo, hi), min(1.0, p)


def mcnemar_test(a: list[bool], b: list[bool]) -> dict:
    """McNemar 配对检验。
    b01: a 错 b 对（b 多对的题）
    b10: a 对 b 错（b 多错的题）
    H0: b01 == b10
    """
    if len(a) != len(b):
        raise ValueError("McNemar 配对要求长度相同")
    b01 = sum(1 for i in range(len(a)) if (not a[i]) and b[i])
    b10 = sum(1 for i in range(len(a)) if a[i] and (not b[i]))
    n_disc = b01 + b10
    # 连续性修正的 chi-square (Edwards 1948)
    if n_disc == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / n_disc
        # df=1 卡方分布 → 用近似 p = exp(-chi2/2)
        p = math.exp(-chi2 / 2)  # 上尾近似
    return {"b_better_only": b01, "a_better_only": b10,
            "discordant_total": n_disc,
            "chi2": round(chi2, 4),
            "p_value_approx": round(p, 5),
            "significant_005": bool(p < 0.05)}


def load_eval(path: str) -> tuple[list[bool], list]:
    """读 eval JSON，返回 (correct list, sample_indices)。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    details = data.get("details") or []
    corrects = [bool(d.get("correct", False)) for d in details]
    indices = data.get("sample_indices") or list(range(len(corrects)))
    return corrects, indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", help="单模型 95% CI")
    ap.add_argument("--a", help="模型 A 的 eval JSON")
    ap.add_argument("--a_name", default="A")
    ap.add_argument("--b", help="模型 B 的 eval JSON")
    ap.add_argument("--b_name", default="B")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.single:
        corrects, _ = load_eval(args.single)
        n = len(corrects)
        acc = sum(corrects) / max(n, 1)
        wlo, whi = wilson_ci(acc, n)
        blo, bhi = bootstrap_ci(corrects, n_boot=args.n_boot, seed=args.seed)
        print(f"\n📊 单模型: {args.single}")
        print(f"   N = {n}")
        print(f"   Accuracy = {acc*100:.2f}%")
        print(f"   Wilson 95% CI    = [{wlo*100:.2f}, {whi*100:.2f}]")
        print(f"   Bootstrap 95% CI = [{blo*100:.2f}, {bhi*100:.2f}]")
        return

    if args.a and args.b:
        a, ia = load_eval(args.a)
        b, ib = load_eval(args.b)
        if len(a) != len(b) or ia != ib:
            print(f"⚠️  两份评测的样本不完全配对 (n_a={len(a)}, n_b={len(b)})")
            print("    这会让 McNemar 和 paired bootstrap 不严格可比")
            print("    请确保两个评测用相同的 --seed 和 --sampling_mode 跑")
            # 截断到共同长度
            n = min(len(a), len(b))
            a = a[:n]; b = b[:n]
        else:
            n = len(a)

        acc_a = sum(a) / n
        acc_b = sum(b) / n
        wlo_a, whi_a = wilson_ci(acc_a, n)
        wlo_b, whi_b = wilson_ci(acc_b, n)
        delta, (dlo, dhi), pval = paired_bootstrap_delta(a, b, n_boot=args.n_boot, seed=args.seed)
        mc = mcnemar_test(a, b)

        print(f"\n📊 配对对比：{args.a_name} vs {args.b_name}  (N={n})")
        print(f"   {args.a_name:>15s}: {acc_a*100:.2f}%  CI=[{wlo_a*100:.2f}, {whi_a*100:.2f}]")
        print(f"   {args.b_name:>15s}: {acc_b*100:.2f}%  CI=[{wlo_b*100:.2f}, {whi_b*100:.2f}]")
        print(f"\n   Δ ({args.b_name} − {args.a_name}) = {delta*100:+.2f} pp")
        print(f"   Paired bootstrap 95% CI: [{dlo*100:+.2f}, {dhi*100:+.2f}] pp")
        print(f"   Paired bootstrap p-value (two-sided): {pval:.4f}")
        print(f"\n   McNemar test:")
        print(f"   - {args.a_name} 错 / {args.b_name} 对   (b 修复了 a 的错): {mc['b_better_only']}")
        print(f"   - {args.a_name} 对 / {args.b_name} 错   (b 引入了新错):   {mc['a_better_only']}")
        print(f"   - chi2 = {mc['chi2']}, p ≈ {mc['p_value_approx']}, "
              f"significant (α=0.05): {mc['significant_005']}")

        result = {
            "n": n,
            "acc_a": acc_a, "acc_b": acc_b,
            "ci_a": [wlo_a, whi_a], "ci_b": [wlo_b, whi_b],
            "delta": delta, "delta_ci": [dlo, dhi], "delta_p": pval,
            "mcnemar": mc,
        }
        out = Path("logs/stats") / f"{args.a_name}_vs_{args.b_name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n📝 已保存 → {out}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
