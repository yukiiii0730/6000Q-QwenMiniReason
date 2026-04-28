"""结果可视化（v4 升级）

支持的图表：
  --radar          模型在 [GSM8K, MATH-500, BBH] 上的雷达图（多模型对比）
  --error_pie      错误类型分布饼图（来自 classify_errors.py 的 summary.json）
  --ablation_bar   ablation 实验条形图（来自 run_ablation.py 的 summary.json）
  --loss_curve     训练 loss 曲线（trainer_log_history.csv）
  --improve_bar    错误类型修复率条形图（before/after 对比）

用法：
    # 雷达图（自动从 logs/compare_metrics.json 读）
    python eval/visualize.py --radar

    # 错误分布饼图
    python eval/visualize.py --error_pie results/errors/sft/summary.json

    # ablation 实验条形图
    python eval/visualize.py --ablation_bar results/ablation/summary.json

    # 错误修复率（before vs after Targeted DPO）
    python eval/visualize.py --improve_bar \
        --before results/errors/sft/summary.json \
        --after  results/errors/dpo/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 后端无关
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 用 DejaVu / Liberation 兜底，避免中文字体缺失（图表用英文，也可在 PPT 里再翻译）
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 雷达图：模型在多个 benchmark 上对比
# =============================================================================
def plot_radar_multi(rows: list[dict], output_path: Path) -> None:
    """rows: [{"label": "...", "gsm8k": 0.5, "math500": 0.3, "bbh_macro": 0.6}, ...]"""
    metrics_keys = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("bbh_macro", "BBH")]
    labels = [k[1] for k in metrics_keys]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    cmap = plt.cm.get_cmap("tab10")

    for i, row in enumerate(rows):
        vals = [row.get(k[0]) for k in metrics_keys]
        if any(v is None for v in vals):
            continue  # 缺数据的模型跳过
        vals = list(vals) + [vals[0]]
        color = cmap(i % 10)
        ax.plot(angles, vals, linewidth=2, color=color, label=row.get("label", f"row{i}"))
        ax.fill(angles, vals, alpha=0.10, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["20%", "40%", "60%", "80%"], fontsize=9)
    ax.set_title("Reasoning Benchmark Comparison", fontsize=13, pad=20)
    ax.legend(bbox_to_anchor=(1.25, 1.05), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 已生成 {output_path}")


def cmd_radar(args):
    metrics_json = Path(args.metrics_json or "logs/compare_metrics.json")
    if not metrics_json.exists():
        raise SystemExit(f"找不到 {metrics_json}，请先运行 eval/compare_table.py")
    data = json.loads(metrics_json.read_text())
    rows = data.get("rows") or []
    plot_radar_multi(rows, Path(args.out_dir) / "radar.png")


# =============================================================================
# 错误分布饼图（来自 classify_errors.py）
# =============================================================================
def plot_error_pie(summary_path: Path, output_path: Path) -> None:
    data = json.loads(summary_path.read_text())
    dist = data.get("distribution") or data.get("distribution_pct") or {}
    if not dist:
        raise SystemExit(f"summary.json 中没有 distribution 字段: {summary_path}")

    labels, counts = [], []
    for k, v in dist.items():
        if v <= 0:
            continue
        labels.append(k)
        counts.append(v)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(p * sum(counts) / 100)})",
        startangle=90, textprops={"fontsize": 10},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(f"Error Type Distribution (n={int(sum(counts))})", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 已生成 {output_path}")


def cmd_error_pie(args):
    plot_error_pie(Path(args.error_pie), Path(args.out_dir) / "error_pie.png")


# =============================================================================
# Ablation 条形图
# =============================================================================
def plot_ablation_bar(summary_path: Path, output_path: Path) -> None:
    data = json.loads(summary_path.read_text())
    groups = data.get("groups") or {}
    if not groups:
        raise SystemExit(f"summary.json 中没有 groups: {summary_path}")

    keys = list(groups.keys())
    metrics = [("gsm8k", "GSM8K"), ("math500", "MATH-500"), ("bbh_macro", "BBH")]

    n_groups = len(keys)
    n_metrics = len(metrics)
    width = 0.25
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.5), 5))
    cmap = plt.cm.get_cmap("Set2")

    for i, (key, label) in enumerate(metrics):
        # 用 DPO 阶段的指标（DPO 是最终）
        vals = []
        for g in keys:
            dpo = (groups[g].get("dpo") or {}).get(key)
            sft = (groups[g].get("sft") or {}).get(key)
            vals.append((dpo if dpo is not None else sft) or 0)
        ax.bar(x + (i - n_metrics / 2 + 0.5) * width, vals, width,
               label=label, color=cmap(i))
        for xi, v in zip(x + (i - n_metrics / 2 + 0.5) * width, vals):
            if v > 0:
                ax.text(xi, v + 0.01, f"{v*100:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation Comparison (post-DPO)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 已生成 {output_path}")


def cmd_ablation_bar(args):
    plot_ablation_bar(Path(args.ablation_bar), Path(args.out_dir) / "ablation_bar.png")


# =============================================================================
# 错误修复率条形图（before vs after Targeted DPO）
# =============================================================================
def plot_improvement_bar(before_path: Path, after_path: Path, output_path: Path) -> None:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    bd = before.get("distribution") or {}
    ad = after.get("distribution") or {}

    types = sorted(set(bd.keys()) | set(ad.keys()))
    types = [t for t in types if t != "unknown"]
    bvals = [bd.get(t, 0) for t in types]
    avals = [ad.get(t, 0) for t in types]

    x = np.arange(len(types))
    width = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(types) * 1.2), 5))
    ax.bar(x - width / 2, bvals, width, label="Before DPO", color="#cc6677")
    ax.bar(x + width / 2, avals, width, label="After Targeted DPO", color="#117733")
    for xi, v in zip(x - width / 2, bvals):
        ax.text(xi, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + width / 2, avals):
        ax.text(xi, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, fontsize=10)
    ax.set_ylabel("Badcase Count")
    ax.set_title("Error-Type Repair Rate (Before vs After Targeted DPO)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 已生成 {output_path}")


def cmd_improve_bar(args):
    plot_improvement_bar(Path(args.before), Path(args.after),
                         Path(args.out_dir) / "improve_bar.png")


# =============================================================================
# 训练 loss 曲线
# =============================================================================
def plot_loss_curve(log_csv: Path, output_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(log_csv)
    if "step" not in df.columns or "loss" not in df.columns:
        raise SystemExit("CSV 需包含 step + loss 两列")
    fig = plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["loss"], linewidth=1.5)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"📊 已生成 {output_path}")


def cmd_loss(args):
    plot_loss_curve(Path(args.loss_curve), Path(args.out_dir) / "loss_curve.png")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="结果可视化（v4：雷达图+饼图+ablation+修复率+loss）")
    ap.add_argument("--out_dir", default="eval/figures")
    ap.add_argument("--radar", action="store_true",
                    help="生成多模型雷达图（默认从 logs/compare_metrics.json 读）")
    ap.add_argument("--metrics_json", help="自定义 metrics JSON 路径（含 rows: [...]）")
    ap.add_argument("--error_pie", help="错误分类 summary.json 路径")
    ap.add_argument("--ablation_bar", help="ablation summary.json 路径")
    ap.add_argument("--improve_bar", action="store_true",
                    help="错误修复率条形图（需配合 --before --after）")
    ap.add_argument("--before", help="DPO 之前的错误分类 summary.json")
    ap.add_argument("--after",  help="DPO 之后的错误分类 summary.json")
    ap.add_argument("--loss_curve", help="trainer_log_history.csv 路径")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    did = False
    if args.radar:
        cmd_radar(args); did = True
    if args.error_pie:
        cmd_error_pie(args); did = True
    if args.ablation_bar:
        cmd_ablation_bar(args); did = True
    if args.improve_bar:
        if not (args.before and args.after):
            raise SystemExit("--improve_bar 需要 --before 和 --after")
        cmd_improve_bar(args); did = True
    if args.loss_curve:
        cmd_loss(args); did = True

    if not did:
        ap.print_help()


if __name__ == "__main__":
    main()
