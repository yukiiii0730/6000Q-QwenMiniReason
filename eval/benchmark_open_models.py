import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("accuracy")


def run_command(cmd: list[str], cwd: Path):
    print("➤", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description="批量评测开源模型在 GSM8K 与 BBH 上的表现")
    parser.add_argument("--config", default="config/benchmark_models.yaml")
    parser.add_argument("--skip_existing", action="store_true", help="已有结果时跳过对应模型")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(project_root / args.config)
    eval_cfg = cfg["evaluation"]
    models = cfg["models"]

    output_dir = project_root / eval_cfg.get("output_dir", "logs/open_model_benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_script = project_root / "eval" / "gsm8k_eval.py"
    bbh_script = project_root / "eval" / "bbh_eval.py"

    rows = []
    for model_cfg in models:
        label = model_cfg["label"]
        model_path = model_cfg["model_path"]
        slug = slugify(label)

        gsm8k_output = output_dir / f"{slug}_gsm8k.json"
        bbh_output = output_dir / f"{slug}_bbh.json"

        if args.skip_existing and gsm8k_output.exists() and bbh_output.exists():
            print(f"ℹ️ 跳过已有结果: {label}")
        else:
            gsm8k_cmd = [
                sys.executable,
                str(gsm8k_script),
                "--model_path",
                model_path,
                "--max_samples",
                str(eval_cfg.get("gsm8k_max_samples", 200)),
                "--output",
                str(gsm8k_output),
            ]
            bbh_cmd = [
                sys.executable,
                str(bbh_script),
                "--model_path",
                model_path,
                "--subset",
                str(eval_cfg.get("bbh_subset", "boolean_expressions")),
                "--max_samples",
                str(eval_cfg.get("bbh_max_samples", 200)),
                "--output",
                str(bbh_output),
            ]

            if eval_cfg.get("load_in_4bit", False):
                gsm8k_cmd.append("--load_in_4bit")
                bbh_cmd.append("--load_in_4bit")

            print(f"\n===== 评测模型: {label} =====")
            run_command(gsm8k_cmd, cwd=project_root)
            run_command(bbh_cmd, cwd=project_root)

        rows.append(
            {
                "label": label,
                "model_path": model_path,
                "gsm8k_accuracy": load_accuracy(gsm8k_output),
                "bbh_accuracy": load_accuracy(bbh_output),
                "note": model_cfg.get("note", ""),
            }
        )

    summary = {
        "evaluation": eval_cfg,
        "results": rows,
    }
    summary_path = output_dir / "benchmark_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"  {'模型':<28} {'GSM8K':>10} {'BBH':>10}")
    print("=" * 78)
    for row in rows:
        gsm = row["gsm8k_accuracy"]
        bbh = row["bbh_accuracy"]
        gsm_text = f"{gsm:.2%}" if gsm is not None else "N/A"
        bbh_text = f"{bbh:.2%}" if bbh is not None else "N/A"
        print(f"  {row['label']:<28} {gsm_text:>10} {bbh_text:>10}")
    print("=" * 78)
    print(f"结果汇总已保存至: {summary_path}")


if __name__ == "__main__":
    main()