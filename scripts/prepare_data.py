#!/usr/bin/env python3
"""
数据集下载与预处理脚本（v4：三剑客主干 + in-distribution 锚点）

【为什么是 v4】
v3 用 MetaMathQA + GSM8K-train + Magpie，太偏 in-distribution，缺数学推理深度，
难以逼近 7B。v4 升级到学界共识的"三剑客"主干 + GSM8K-train 锚点：

  Stage A  GSM8K-train               7.5k    in-distribution（应用题底色）
  Stage B1 OpenR1-Math (verified)   10k     R1 蒸馏（高质量长 CoT）
  Stage B2 Orca-Math-200k           15k     GPT-4 蒸馏（应用题广度，1.5B 友好）
  Stage B3 NuminaMath-CoT (非 gsm8k) 8k      多源专家（奥赛/AMC/AOPS 题型多样）
  Stage C  Magpie-Reasoning          3k     通用推理防退化（占比降到 7%）

合计 ~43.5k，跨集去重 + 长度过滤后实际 ~38k。

【格式统一】
所有数据集 normalize 为：
   {"instruction": "...", "input": "<问题>", "output": "<解答>", "source": "<tag>"}

【长度过滤】
应用题档（GSM8K/Orca）：≤1024 tokens（按字符 ÷ 3 估算 token 数）
长 CoT 档（OpenR1/NuminaMath/Magpie）：≤2048 tokens

【跨数据集去重】
problem 文本 normalize（去标点+小写+压缩空白）后用 SHA-1 去重，避免：
  - GSM8K-train ⊆ NuminaMath-CoT 的 source=gsm8k 子集（强制去重）
  - 同集内重复

用法：
    python scripts/prepare_data.py                              # 默认 v4 五段配比
    python scripts/prepare_data.py --quick                      # 每集 500 条
    python scripts/prepare_data.py --gsm8k_n 7500 --openr1_n 10000 --orca_n 15000 \
        --numinamath_n 8000 --magpie_n 3000 --dpo_n 5000
"""
from __future__ import annotations

import os
import sys

# 必须在 import yaml / datasets 等任何可能走 HTTPS 的库之前修复 CA（含中文路径场景）
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from hf_ssl_env import apply_hf_ssl_fix

apply_hf_ssl_fix()

import argparse
import hashlib
import json
import re
from collections import Counter

import yaml
from datasets import load_dataset


DEFAULT_INSTRUCTION = "请先进行清晰的逐步推理，再给出最终答案。"

# ── 长度阈值（字符数；token 估算：char/3） ─────────────────────────────────────
# 应用题档：1024 tokens ~ 3072 字符
SHORT_MAX_CHARS = 3072
# 长 CoT 档：2048 tokens ~ 6144 字符
LONG_MAX_CHARS = 6144


# =============================================================================
# 工具函数
# =============================================================================
def load_hf_token(config_path: str) -> str:
    if not os.path.exists(config_path):
        return os.environ.get("HF_TOKEN", "").strip()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    token = (cfg or {}).get("hf_token", "").strip() or os.environ.get("HF_TOKEN", "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("✅ HuggingFace Token 已加载")
    else:
        print("⚠️  未配置 HF_TOKEN，如访问受限数据集请先在 .env 中设置")
    return token


def _select_first(ds, n: int):
    return ds.select(range(min(n, len(ds))))


_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_problem(text: str) -> str:
    """用于跨集去重的文本规范化：去标点+小写+压缩空白。"""
    if not text:
        return ""
    s = str(text).strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def problem_hash(text: str) -> str:
    return hashlib.sha1(normalize_problem(text).encode("utf-8")).hexdigest()


class Deduper:
    """跨数据集去重：保留首次出现，统计每个 source 重复了多少。"""
    def __init__(self):
        self._seen: set[str] = set()
        self.dropped = Counter()

    def accept(self, problem: str, source: str) -> bool:
        h = problem_hash(problem)
        if h in self._seen:
            self.dropped[source] += 1
            return False
        self._seen.add(h)
        return True


def length_ok(problem: str, output: str, max_chars: int) -> bool:
    return len(problem or "") + len(output or "") <= max_chars


# =============================================================================
# 字段适配器：每个数据集一个，输出统一 schema
# =============================================================================
def _format_gsm8k_solution(answer_field: str) -> str:
    """GSM8K answer 格式: <reasoning>\n#### <number> → <reasoning>\n\n答案：<number>"""
    s = (answer_field or "").strip()
    if "####" in s:
        cot, ans = s.split("####", 1)
        return f"{cot.strip()}\n\n答案：{ans.strip()}"
    return s


def _pick_openr1_verified_generation(example: dict) -> str:
    """OpenR1-Math: generations 列表 + correctness_math_verify 标记，挑 verified=True 的那条。"""
    gens = example.get("generations") or []
    correctness = example.get("correctness_math_verify") or []
    for g, ok in zip(gens, correctness):
        if ok and isinstance(g, str) and g.strip():
            return g.strip()
    sol = example.get("solution") or ""
    return str(sol).strip()


def adapt_gsm8k(ex: dict) -> dict | None:
    q = (ex.get("question") or "").strip()
    a = _format_gsm8k_solution(ex.get("answer", ""))
    if not q or not a:
        return None
    return {"instruction": DEFAULT_INSTRUCTION, "input": q, "output": a, "source": "gsm8k_train"}


def adapt_openr1(ex: dict) -> dict | None:
    q = (ex.get("problem") or "").strip()
    a = _pick_openr1_verified_generation(ex)
    if not q or not a:
        return None
    return {"instruction": DEFAULT_INSTRUCTION, "input": q, "output": a, "source": "openr1_math"}


def adapt_orca(ex: dict) -> dict | None:
    q = (ex.get("question") or "").strip()
    a = (ex.get("answer") or "").strip()
    if not q or not a:
        return None
    return {"instruction": DEFAULT_INSTRUCTION, "input": q, "output": a, "source": "orca_math"}


def adapt_numinamath(ex: dict) -> dict | None:
    q = (ex.get("problem") or "").strip()
    a = (ex.get("solution") or "").strip()
    src = (ex.get("source") or "").strip().lower()
    if not q or not a:
        return None
    # 排除 source=gsm8k 子集（与 GSM8K-train 重复）
    if src == "gsm8k":
        return None
    return {"instruction": DEFAULT_INSTRUCTION, "input": q, "output": a,
            "source": f"numinamath_{src or 'misc'}"}


def adapt_magpie(ex: dict) -> dict | None:
    instr = (ex.get("instruction") or "").strip()
    resp = (ex.get("response") or "").strip()
    if not instr or not resp:
        return None
    return {"instruction": instr, "input": "", "output": resp, "source": "magpie_reasoning"}


# =============================================================================
# 单数据集加载（在线 → 适配 → 长度过滤 → 去重）
# =============================================================================
def stream_dataset(name: str, config: str | None, split: str, n: int,
                   adapter, max_chars: int, deduper: Deduper, label: str) -> list[dict]:
    print(f"📥 {label}: 加载 {name}" + (f"({config})" if config else "") + f", split={split}, 取 {n}...")
    try:
        if config:
            ds = load_dataset(name, config, split=split)
        else:
            ds = load_dataset(name, split=split)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  {label} 下载失败：{e}")
        return []

    rows: list[dict] = []
    raw_total = 0
    skipped_format = 0
    skipped_length = 0
    skipped_dedup = 0

    # 不裁剪原数据集长度，因为可能因长度过滤损失大量样本，需要 oversample
    target_n = n
    for ex in ds:
        if len(rows) >= target_n:
            break
        raw_total += 1
        item = adapter(ex)
        if item is None:
            skipped_format += 1
            continue
        problem_text = item["input"] or item["instruction"]
        if not length_ok(problem_text, item["output"], max_chars):
            skipped_length += 1
            continue
        if not deduper.accept(problem_text, item["source"]):
            skipped_dedup += 1
            continue
        rows.append(item)

    print(f"   ✓ {label}: 接受 {len(rows)} 条 | "
          f"扫描 {raw_total} | 格式跳过 {skipped_format} | "
          f"超长跳过 {skipped_length} | 去重跳过 {skipped_dedup}")
    return rows


# =============================================================================
# 五段课程数据准备（v4 主流程）
# =============================================================================
def prepare_sft(output_path: str,
                gsm8k_n: int,
                openr1_n: int,
                orca_n: int,
                numinamath_n: int,
                magpie_n: int) -> None:
    target_n = gsm8k_n + openr1_n + orca_n + numinamath_n + magpie_n
    # 允许 20% 的去重+长度过滤损耗；若现有样本数不足目标的 80%，则重新生成
    min_acceptable = int(target_n * 0.80)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            with open(output_path, "r", encoding="utf-8") as _f:
                existing_count = len(json.load(_f))
            if existing_count >= min_acceptable:
                print(f"⏭️  SFT 缓存已存在（{existing_count} 条 ≥ 目标 {target_n}×80%={min_acceptable}），跳过: {output_path}")
                return
            print(f"⚠️  SFT 缓存样本数不足（{existing_count} 条 < {min_acceptable}），重新生成（HF 原始数据集无需重新下载）...")
        except Exception:
            print(f"⚠️  SFT 缓存读取失败，重新生成: {output_path}")

    deduper = Deduper()
    sft_rows: list[dict] = []

    # ── Stage A: GSM8K-train (in-distribution) ────────────────────────────
    if gsm8k_n > 0:
        rows = stream_dataset("gsm8k", "main", "train", gsm8k_n,
                              adapt_gsm8k, SHORT_MAX_CHARS, deduper,
                              "Stage A · GSM8K-train")
        sft_rows.extend(rows)

    # ── Stage B1: OpenR1-Math (R1 蒸馏，长 CoT) ────────────────────────────
    if openr1_n > 0:
        rows = stream_dataset("open-r1/OpenR1-Math-220k", "default", "train", openr1_n,
                              adapt_openr1, LONG_MAX_CHARS, deduper,
                              "Stage B1 · OpenR1-Math")
        sft_rows.extend(rows)

    # ── Stage B2: Orca-Math (GPT-4 蒸馏，应用题广度) ───────────────────────
    if orca_n > 0:
        rows = stream_dataset("microsoft/orca-math-word-problems-200k", None, "train", orca_n,
                              adapt_orca, SHORT_MAX_CHARS, deduper,
                              "Stage B2 · Orca-Math-200k")
        sft_rows.extend(rows)

    # ── Stage B3: NuminaMath-CoT (题型多样，去 gsm8k 子集) ─────────────────
    if numinamath_n > 0:
        rows = stream_dataset("AI-MO/NuminaMath-CoT", None, "train", numinamath_n,
                              adapt_numinamath, LONG_MAX_CHARS, deduper,
                              "Stage B3 · NuminaMath-CoT")
        sft_rows.extend(rows)

    # ── Stage C: Magpie (通用推理防退化) ───────────────────────────────────
    if magpie_n > 0:
        rows = stream_dataset("Magpie-Align/Magpie-Reasoning-150K", None, "train", magpie_n,
                              adapt_magpie, LONG_MAX_CHARS, deduper,
                              "Stage C · Magpie-Reasoning")
        sft_rows.extend(rows)

    # ── 写出 + 摘要 ────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_rows, f, ensure_ascii=False, indent=2)

    src_counter = Counter(r["source"] for r in sft_rows)
    print(f"\n✅ SFT 数据已保存：共 {len(sft_rows)} 条 → {output_path}")
    print("   分源统计:")
    for src, n in sorted(src_counter.items(), key=lambda x: -x[1]):
        print(f"      {src:30s}  {n:6d}")
    if deduper.dropped:
        print("   跨集去重丢弃:")
        for src, n in deduper.dropped.most_common():
            print(f"      {src:30s}  {n:6d}")


# =============================================================================
# DPO 数据（仅兜底；主力创新点 B 在 GPU 阶段产出 dpo_targeted_*.json）
# =============================================================================
def prepare_dpo(output_path: str, dpo_n: int) -> None:
    min_acceptable = int(dpo_n * 0.80)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            with open(output_path, "r", encoding="utf-8") as _f:
                existing_count = len(json.load(_f))
            if existing_count >= min_acceptable:
                print(f"⏭️  DPO 缓存已存在（{existing_count} 条 ≥ 目标 {dpo_n}×80%={min_acceptable}），跳过下载: {output_path}")
                return
            print(f"⚠️  DPO 缓存样本数不足（{existing_count} 条 < {min_acceptable}），重新生成...")
        except Exception:
            print(f"⚠️  DPO 缓存读取失败，重新生成: {output_path}")

    print(f"📥 下载 argilla/distilabel-math-preference-dpo（取 {dpo_n} 条）...")
    try:
        ds = load_dataset("argilla/distilabel-math-preference-dpo", split="train")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  下载 distilabel-math-preference-dpo 失败：{e}")
        print("    回退到 argilla/ultrafeedback-binarized-preferences-cleaned")
        try:
            ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned", split="train")
        except Exception as e2:  # noqa: BLE001
            print(f"❌ 所有 DPO 数据集都下载失败：{e2}")
            return

    ds = _select_first(ds, dpo_n)
    dpo_rows = []
    for x in ds:
        prompt = (x.get("instruction") or x.get("prompt") or x.get("question") or "").strip()
        chosen = x.get("chosen_response") or x.get("chosen") or ""
        rejected = x.get("rejected_response") or x.get("rejected") or ""
        if isinstance(chosen, list):
            chosen = "\n".join(c.get("content", "") for c in chosen if isinstance(c, dict))
        if isinstance(rejected, list):
            rejected = "\n".join(c.get("content", "") for c in rejected if isinstance(c, dict))
        chosen = str(chosen).strip()
        rejected = str(rejected).strip()
        if not prompt or not chosen or not rejected:
            continue
        dpo_rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_rows, f, ensure_ascii=False, indent=2)
    print(f"✅ DPO 兜底数据已保存：共 {len(dpo_rows)} 条 → {output_path}")


def update_configs(sft_out: str, dpo_out: str) -> None:
    """把 dataset_path 写入 config（quick / 兜底模式专用），不动 stages。"""
    for cfg_path, key, val in [
        ("config/sft_config.yaml", "dataset_path", sft_out),
        ("config/dpo_config.yaml", "dataset_path", dpo_out),
    ]:
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg[key] = val
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print("✅ config/*.yaml 中 dataset_path 已更新")


def main():
    parser = argparse.ArgumentParser(description="数据集下载与预处理（v4 三剑客 + 锚点）")
    parser.add_argument("--config",       default="config/sft_config.yaml")
    parser.add_argument("--output_dir",   default="data/processed")
    # v4 五段课程参数
    parser.add_argument("--gsm8k_n",      type=int, default=7500,
                        help="Stage A: GSM8K-train（in-distribution）")
    parser.add_argument("--openr1_n",     type=int, default=10000,
                        help="Stage B1: OpenR1-Math（R1 蒸馏长 CoT）")
    parser.add_argument("--orca_n",       type=int, default=15000,
                        help="Stage B2: Orca-Math（GPT-4 蒸馏应用题）")
    parser.add_argument("--numinamath_n", type=int, default=8000,
                        help="Stage B3: NuminaMath-CoT（去 gsm8k 子集）")
    parser.add_argument("--magpie_n",     type=int, default=3000,
                        help="Stage C: Magpie（通用推理防退化）")
    parser.add_argument("--dpo_n",        type=int, default=5000)
    parser.add_argument("--quick",        action="store_true",
                        help="快速测试：每集 500 条")
    # 向后兼容（v3 参数名仍能工作）
    parser.add_argument("--sft_n",        type=int, default=None,
                        help="(v3 兼容) 等价 --orca_n 为主力")
    parser.add_argument("--metamath_n",   type=int, default=None,
                        help="(v3 兼容) 已弃用，忽略")
    args = parser.parse_args()

    if args.quick:
        args.gsm8k_n = 500
        args.openr1_n = 500
        args.orca_n = 500
        args.numinamath_n = 500
        args.magpie_n = 500
        args.dpo_n = 500
        print("⚡ 快速测试模式：每集 500 条")

    if args.sft_n is not None:
        args.orca_n = args.sft_n  # 把 v3 的 --sft_n 当 orca 主力

    os.makedirs(args.output_dir, exist_ok=True)
    load_hf_token(args.config)

    sft_out = os.path.join(args.output_dir, "sft_train.json")
    dpo_out = os.path.join(args.output_dir, "dpo_train.json")

    print("=" * 70)
    print("📦 v4 数据准备计划")
    print("=" * 70)
    print(f"   Stage A  GSM8K-train         {args.gsm8k_n:6d} (in-distribution)")
    print(f"   Stage B1 OpenR1-Math         {args.openr1_n:6d} (R1 蒸馏)")
    print(f"   Stage B2 Orca-Math-200k      {args.orca_n:6d} (GPT-4 蒸馏)")
    print(f"   Stage B3 NuminaMath-CoT      {args.numinamath_n:6d} (多源专家，去 gsm8k 重叠)")
    print(f"   Stage C  Magpie-Reasoning    {args.magpie_n:6d} (通用推理)")
    print(f"   ─────────────────────────────")
    total = args.gsm8k_n + args.openr1_n + args.orca_n + args.numinamath_n + args.magpie_n
    print(f"   合计目标: {total} 条（去重+长度过滤后实际略少）")
    print(f"   DPO 兜底: argilla/distilabel-math-preference-dpo {args.dpo_n} 条")
    print("=" * 70)

    prepare_sft(sft_out,
                gsm8k_n=args.gsm8k_n,
                openr1_n=args.openr1_n,
                orca_n=args.orca_n,
                numinamath_n=args.numinamath_n,
                magpie_n=args.magpie_n)
    prepare_dpo(dpo_out, dpo_n=args.dpo_n)
    update_configs(sft_out, dpo_out)

    print("\n📦 数据集摘要：")
    print(f"   SFT : {sft_out}  (5-stage curriculum)")
    print(f"   DPO : {dpo_out}  (fallback)")


if __name__ == "__main__":
    main()
