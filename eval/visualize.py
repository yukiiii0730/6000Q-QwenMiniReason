import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_radar(metrics: dict, output_path: Path) -> None:
    labels = list(metrics.keys())
    values = list(metrics.values())
    values += values[:1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Reasoning Benchmark Radar")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_loss_curve(log_csv: Path, output_path: Path) -> None:
    df = pd.read_csv(log_csv)
    required = {"step", "loss"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 需包含列: {required}")

    fig = plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["loss"], linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="结果可视化（雷达图 + Loss 曲线）")
    parser.add_argument("--metrics_json", help="形如 {'gsm8k':0.63,'bbh':0.48} 的 JSON 文件")
    parser.add_argument("--loss_csv", help="包含 step/loss 两列的训练日志 CSV")
    parser.add_argument("--out_dir", default="eval/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metrics_json:
        with open(args.metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        plot_radar(metrics, out_dir / "radar.png")
        print(f"已生成: {out_dir / 'radar.png'}")

    if args.loss_csv:
        plot_loss_curve(Path(args.loss_csv), out_dir / "loss_curve.png")
        print(f"已生成: {out_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
