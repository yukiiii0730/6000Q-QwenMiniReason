#!/usr/bin/env python3
"""
回填 badcase jsonl：从评测结果 json 中提取 correct=false 的样本，生成 *_badcases.jsonl。

用法示例：
  python scripts/backfill_badcases.py --results_dir logs/runs/20260314_091217/results
  python scripts/backfill_badcases.py --results_dir logs/runs/20260314_091217/results --overwrite
  python scripts/backfill_badcases.py --scan_runs logs/runs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


EXCLUDE_SUFFIX = "_badcases.jsonl"


def iter_result_json_files(results_dir: Path) -> Iterable[Path]:
    if not results_dir.exists() or not results_dir.is_dir():
        return []
    files = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name.endswith(EXCLUDE_SUFFIX):
            continue
        files.append(p)
    return files


def write_badcases_for_json(json_path: Path, overwrite: bool = False) -> tuple[int, int, Path] | None:
    out_path = json_path.with_name(f"{json_path.stem}_badcases.jsonl")
    if out_path.exists() and not overwrite:
        return None

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 跳过（JSON 解析失败）: {json_path} | {e}")
        return None

    if not isinstance(data, dict) or not isinstance(data.get("details"), list):
        print(f"[WARN] 跳过（缺少 details 列表）: {json_path}")
        return None

    details: list[dict[str, Any]] = data["details"]
    badcases = [x for x in details if isinstance(x, dict) and not x.get("correct", False)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in badcases:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(details), len(badcases), out_path


def process_one_results_dir(results_dir: Path, overwrite: bool) -> tuple[int, int, int]:
    result_files = list(iter_result_json_files(results_dir))
    if not result_files:
        print(f"[INFO] 未找到结果文件: {results_dir}")
        return 0, 0, 0

    wrote = 0
    skipped_existing = 0
    total_badcases = 0

    for jf in result_files:
        ret = write_badcases_for_json(jf, overwrite=overwrite)
        if ret is None:
            skipped_existing += 1
            continue
        total_n, bad_n, out = ret
        wrote += 1
        total_badcases += bad_n
        print(f"[OK] {jf.name}: {bad_n}/{total_n} -> {out.name}")

    return len(result_files), wrote, skipped_existing


def scan_runs(scan_root: Path, overwrite: bool) -> None:
    if not scan_root.exists() or not scan_root.is_dir():
        raise FileNotFoundError(f"scan_runs 路径不存在: {scan_root}")

    run_dirs = sorted([p for p in scan_root.iterdir() if p.is_dir()])
    if not run_dirs:
        print(f"[INFO] 未发现 run 目录: {scan_root}")
        return

    all_files = 0
    all_wrote = 0
    all_skipped = 0

    for rd in run_dirs:
        results_dir = rd / "results"
        files_n, wrote_n, skipped_n = process_one_results_dir(results_dir, overwrite=overwrite)
        all_files += files_n
        all_wrote += wrote_n
        all_skipped += skipped_n

    print("\n===== 汇总 =====")
    print(f"结果 JSON 文件数 : {all_files}")
    print(f"新生成 badcase   : {all_wrote}")
    print(f"已存在跳过       : {all_skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="从评测 JSON 回填 *_badcases.jsonl")
    parser.add_argument("--results_dir", default="", help="单个结果目录，例如 logs/runs/<run_id>/results")
    parser.add_argument("--scan_runs", default="", help="扫描多个运行目录，例如 logs/runs")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在 badcase 文件")
    args = parser.parse_args()

    if not args.results_dir and not args.scan_runs:
        parser.error("请至少提供 --results_dir 或 --scan_runs")

    if args.results_dir:
        results_dir = Path(args.results_dir)
        files_n, wrote_n, skipped_n = process_one_results_dir(results_dir, overwrite=args.overwrite)
        print("\n===== 汇总 =====")
        print(f"结果 JSON 文件数 : {files_n}")
        print(f"新生成 badcase   : {wrote_n}")
        print(f"已存在跳过       : {skipped_n}")

    if args.scan_runs:
        scan_runs(Path(args.scan_runs), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
