#!/usr/bin/env python3
"""Bootstrap robustness analysis for the SHROOM model-size study.

This script uses only:
  1) the labeled SHROOM test JSON, and
  2) saved per-example prediction JSONs for each checkpoint.

It does NOT rerun inference or training.

Main analysis:
- Recompute point accuracy and Spearman's rho for every controlled checkpoint.
- Paired non-parametric bootstrap over test examples.
- 95% percentile confidence intervals for each checkpoint.
- Smallest-to-largest checkpoint differences within each family.
- LoRA-minus-OOTB differences for matched checkpoints.
- Family-average LoRA gains.
- Qwen-3B adjacent-checkpoint diagnostics.

The same bootstrap sample is used for all checkpoints in each replicate,
which makes model comparisons paired at the example level.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

DEFAULT_BASE = Path("test_set_scores_all - A100 GPU")
DEFAULT_GOLD = DEFAULT_BASE / "data" / "SHROOM_test-labeled" / "test.model-agnostic.json"
DEFAULT_PRED_DIR = DEFAULT_BASE / "outputs" / "predictions" / "archive"
DEFAULT_OUT_DIR = DEFAULT_BASE / "outputs" / "bootstrap_robustness"

# The 15 controlled checkpoints used in the main paper, each with OOTB + LoRA.
# LoRA files include timestamps, so glob patterns end with __*.json.
CHECKPOINTS = [
    # DeBERTa CE
    ("DeBERTa CE", "xsmall", 0, "OOTB", "test.model-agnostic__cross-encoder__nli-deberta-v3-xsmall.json"),
    ("DeBERTa CE", "xsmall", 0, "LoRA", "test.model-agnostic__finetuned__lora__soft__cross-encoder__nli-deberta-v3-xsmall__*.json"),
    ("DeBERTa CE", "small", 1, "OOTB", "test.model-agnostic__cross-encoder__nli-deberta-v3-small.json"),
    ("DeBERTa CE", "small", 1, "LoRA", "test.model-agnostic__finetuned__lora__soft__cross-encoder__nli-deberta-v3-small__*.json"),
    ("DeBERTa CE", "base", 2, "OOTB", "test.model-agnostic__cross-encoder__nli-deberta-v3-base.json"),
    ("DeBERTa CE", "base", 2, "LoRA", "test.model-agnostic__finetuned__lora__soft__cross-encoder__nli-deberta-v3-base__*.json"),
    ("DeBERTa CE", "large", 3, "OOTB", "test.model-agnostic__cross-encoder__nli-deberta-v3-large.json"),
    ("DeBERTa CE", "large", 3, "LoRA", "test.model-agnostic__finetuned__lora__soft__cross-encoder__nli-deberta-v3-large__*.json"),

    # FLAN-T5
    ("FLAN-T5", "small", 0, "OOTB", "test.model-agnostic__google__flan-t5-small.json"),
    ("FLAN-T5", "small", 0, "LoRA", "test.model-agnostic__finetuned__lora__soft__google__flan-t5-small__*.json"),
    ("FLAN-T5", "base", 1, "OOTB", "test.model-agnostic__google__flan-t5-base.json"),
    ("FLAN-T5", "base", 1, "LoRA", "test.model-agnostic__finetuned__lora__soft__google__flan-t5-base__*.json"),
    ("FLAN-T5", "large", 2, "OOTB", "test.model-agnostic__google__flan-t5-large.json"),
    ("FLAN-T5", "large", 2, "LoRA", "test.model-agnostic__finetuned__lora__soft__google__flan-t5-large__*.json"),
    ("FLAN-T5", "xl", 3, "OOTB", "test.model-agnostic__google__flan-t5-xl.json"),
    ("FLAN-T5", "xl", 3, "LoRA", "test.model-agnostic__finetuned__lora__soft__google__flan-t5-xl__*.json"),

    # Gemma
    ("Gemma", "270m", 0, "OOTB", "test.model-agnostic__google__gemma-3-270m-it.json"),
    ("Gemma", "270m", 0, "LoRA", "test.model-agnostic__finetuned__lora__soft__google__gemma-3-270m-it__*.json"),
    ("Gemma", "1b", 1, "OOTB", "test.model-agnostic__google__gemma-3-1b-it.json"),
    ("Gemma", "1b", 1, "LoRA", "test.model-agnostic__finetuned__lora__soft__google__gemma-3-1b-it__*.json"),
    ("Gemma", "4b", 2, "OOTB", "test.model-agnostic__google__gemma-3-4b-it.json"),
    ("Gemma", "4b", 2, "LoRA", "test.model-agnostic__finetuned__lora__soft__google__gemma-3-4b-it__*.json"),

    # Qwen
    ("Qwen", "0.5B", 0, "OOTB", "test.model-agnostic__Qwen__Qwen2.5-0.5B-Instruct.json"),
    ("Qwen", "0.5B", 0, "LoRA", "test.model-agnostic__finetuned__lora__soft__Qwen__Qwen2.5-0.5B-Instruct__*.json"),
    ("Qwen", "1.5B", 1, "OOTB", "test.model-agnostic__Qwen__Qwen2.5-1.5B-Instruct.json"),
    ("Qwen", "1.5B", 1, "LoRA", "test.model-agnostic__finetuned__lora__soft__Qwen__Qwen2.5-1.5B-Instruct__*.json"),
    ("Qwen", "3B", 2, "OOTB", "test.model-agnostic__Qwen__Qwen2.5-3B-Instruct.json"),
    ("Qwen", "3B", 2, "LoRA", "test.model-agnostic__finetuned__lora__soft__Qwen__Qwen2.5-3B-Instruct__*.json"),
    ("Qwen", "7B", 3, "OOTB", "test.model-agnostic__Qwen__Qwen2.5-7B-Instruct.json"),
    ("Qwen", "7B", 3, "LoRA", "test.model-agnostic__finetuned__lora__soft__Qwen__Qwen2.5-7B-Instruct__*.json"),
]

FAMILY_ORDER = ["DeBERTa CE", "FLAN-T5", "Gemma", "Qwen"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired bootstrap robustness analysis for SHROOM predictions")
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD, help="Path to labeled test.model-agnostic.json")
    p.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR, help="Directory containing checkpoint prediction JSONs")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for CSV outputs")
    p.add_argument("--n-boot", type=int, default=10_000, help="Number of bootstrap replicates (default: 10000)")
    p.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed (default: 42)")
    p.add_argument("--batch-size", type=int, default=25, help="Bootstrap batch size; lower if memory is tight")
    p.add_argument(
        "--stratify-by-task",
        action="store_true",
        help="Resample within DM/MT/PG separately, preserving the original task counts",
    )
    return p.parse_args()

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def resolve_prediction_file(pred_dir: Path, pattern: str) -> Path:
    matches = sorted(pred_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one match for pattern:\n  {pattern}\n"
            f"Found {len(matches)} matches:\n  " + "\n  ".join(str(x) for x in matches)
        )
    return matches[0]

def load_data(gold_path: Path, pred_dir: Path):
    gold_rows = load_json(gold_path)
    gold_by_id = {int(r["id"]): r for r in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("Duplicate IDs found in gold file")

    ids = np.array(sorted(gold_by_id), dtype=int)
    y_soft = np.array([float(gold_by_id[i]["p(Hallucination)"]) for i in ids], dtype=float)
    y_bin = np.array(
        [1 if gold_by_id[i]["label"] == "Hallucination" else 0 for i in ids],
        dtype=np.uint8,
    )
    tasks = np.array([str(gold_by_id[i]["task"]) for i in ids])

    metadata: List[Dict] = []
    score_columns = []

    for family, size, size_order, condition, pattern in CHECKPOINTS:
        path = resolve_prediction_file(pred_dir, pattern)
        rows = load_json(path)
        pred_by_id = {int(r["id"]): float(r["p(Hallucination)"]) for r in rows}

        if len(pred_by_id) != len(rows):
            raise ValueError(f"Duplicate IDs in prediction file: {path.name}")
        if set(pred_by_id) != set(gold_by_id):
            missing = sorted(set(gold_by_id) - set(pred_by_id))
            extra = sorted(set(pred_by_id) - set(gold_by_id))
            raise ValueError(
                f"ID mismatch in {path.name}: missing={len(missing)}, extra={len(extra)}"
            )

        scores = np.array([pred_by_id[i] for i in ids], dtype=float)
        score_columns.append(scores)
        metadata.append(
            {
                "family": family,
                "size": size,
                "size_order": size_order,
                "condition": condition,
                "file": path.name,
            }
        )

    score_matrix = np.column_stack(score_columns)  # shape: n_examples x n_models
    meta = pd.DataFrame(metadata)
    meta["model_index"] = np.arange(len(meta), dtype=int)
    return ids, y_soft, y_bin, tasks, score_matrix, meta

def point_metrics(scores: np.ndarray, y_soft: np.ndarray, y_bin: np.ndarray):
    acc = ((scores >= 0.5) == y_bin[:, None]).mean(axis=0)
    rho = np.array(
        [spearmanr(scores[:, j], y_soft).statistic for j in range(scores.shape[1])],
        dtype=float,
    )
    return acc, rho

def make_bootstrap_indices(
    rng: np.random.Generator,
    n_examples: int,
    batch_size: int,
    tasks: np.ndarray | None,
) -> np.ndarray:
    if tasks is None:
        return rng.integers(0, n_examples, size=(batch_size, n_examples))

    # Stratified bootstrap: preserve the exact DM/MT/PG counts in every replicate.
    parts = []
    for task in np.unique(tasks):
        group = np.flatnonzero(tasks == task)
        within = rng.integers(0, len(group), size=(batch_size, len(group)))
        parts.append(group[within])
    return np.concatenate(parts, axis=1)

def bootstrap_metrics(
    scores: np.ndarray,
    y_soft: np.ndarray,
    y_bin: np.ndarray,
    n_boot: int,
    seed: int,
    batch_size: int,
    tasks: np.ndarray | None = None,
):
    """Return bootstrap accuracy/rho matrices of shape n_boot x n_models.

    Spearman's rho is computed exactly as Pearson correlation of average ranks.
    This correctly handles ties, including ties introduced by duplicated examples
    in bootstrap samples.
    """
    rng = np.random.default_rng(seed)
    n_examples, n_models = scores.shape

    acc_boot = np.empty((n_boot, n_models), dtype=np.float32)
    rho_boot = np.empty((n_boot, n_models), dtype=np.float32)

    done = 0
    while done < n_boot:
        b = min(batch_size, n_boot - done)
        idx = make_bootstrap_indices(rng, n_examples, b, tasks)

        y_soft_b = y_soft[idx]       # b x n_examples
        y_bin_b = y_bin[idx]         # b x n_examples
        scores_b = scores[idx, :]    # b x n_examples x n_models

        # Accuracy for all models in all bootstrap replicates in this batch.
        acc_boot[done : done + b] = ((scores_b >= 0.5) == y_bin_b[:, :, None]).mean(axis=1)

        # Spearman rho = Pearson correlation of ranks.
        gold_ranks = rankdata(y_soft_b, axis=1, method="average")
        model_ranks = rankdata(scores_b, axis=1, method="average")

        gold_centered = gold_ranks - gold_ranks.mean(axis=1, keepdims=True)
        model_centered = model_ranks - model_ranks.mean(axis=1, keepdims=True)

        numerator = (gold_centered[:, :, None] * model_centered).sum(axis=1)
        denominator = np.sqrt(
            (gold_centered ** 2).sum(axis=1)[:, None]
            * (model_centered ** 2).sum(axis=1)
        )
        rho_boot[done : done + b] = numerator / denominator

        done += b
        print(f"Bootstrap progress: {done:,}/{n_boot:,}", end="\r", flush=True)

    print()
    return acc_boot, rho_boot

def ci95(x: np.ndarray):
    lo, hi = np.quantile(x, [0.025, 0.975])
    return float(lo), float(hi)

def build_lookup(meta: pd.DataFrame):
    return {
        (row["family"], row["size"], row["condition"]): int(row["model_index"])
        for _, row in meta.iterrows()
    }

def checkpoint_table(meta, point_acc, point_rho, acc_boot, rho_boot):
    rows = []
    for _, m in meta.iterrows():
        j = int(m["model_index"])
        alo, ahi = ci95(acc_boot[:, j])
        rlo, rhi = ci95(rho_boot[:, j])
        rows.append(
            {
                **m.to_dict(),
                "accuracy": point_acc[j],
                "accuracy_ci_low": alo,
                "accuracy_ci_high": ahi,
                "rho": point_rho[j],
                "rho_ci_low": rlo,
                "rho_ci_high": rhi,
            }
        )
    return pd.DataFrame(rows)

def smallest_to_largest_table(meta, point_acc, point_rho, acc_boot, rho_boot):
    rows = []
    for family in FAMILY_ORDER:
        for condition in ["OOTB", "LoRA"]:
            sub = meta[(meta["family"] == family) & (meta["condition"] == condition)].sort_values("size_order")
            small = sub.iloc[0]
            large = sub.iloc[-1]
            i_small = int(small["model_index"])
            i_large = int(large["model_index"])

            d_acc_boot = acc_boot[:, i_large] - acc_boot[:, i_small]
            d_rho_boot = rho_boot[:, i_large] - rho_boot[:, i_small]
            alo, ahi = ci95(d_acc_boot)
            rlo, rhi = ci95(d_rho_boot)

            rows.append(
                {
                    "family": family,
                    "condition": condition,
                    "smallest_checkpoint": small["size"],
                    "largest_checkpoint": large["size"],
                    "delta_accuracy": point_acc[i_large] - point_acc[i_small],
                    "delta_accuracy_ci_low": alo,
                    "delta_accuracy_ci_high": ahi,
                    "delta_rho": point_rho[i_large] - point_rho[i_small],
                    "delta_rho_ci_low": rlo,
                    "delta_rho_ci_high": rhi,
                }
            )
    return pd.DataFrame(rows)

def lora_vs_ootb_table(meta, lookup, point_acc, point_rho, acc_boot, rho_boot):
    rows = []
    for family in FAMILY_ORDER:
        sizes = (
            meta[meta["family"] == family]
            .sort_values("size_order")["size"]
            .drop_duplicates()
            .tolist()
        )
        for size in sizes:
            io = lookup[(family, size, "OOTB")]
            il = lookup[(family, size, "LoRA")]
            d_acc_boot = acc_boot[:, il] - acc_boot[:, io]
            d_rho_boot = rho_boot[:, il] - rho_boot[:, io]
            alo, ahi = ci95(d_acc_boot)
            rlo, rhi = ci95(d_rho_boot)
            rows.append(
                {
                    "family": family,
                    "size": size,
                    "delta_accuracy_lora_minus_ootb": point_acc[il] - point_acc[io],
                    "delta_accuracy_ci_low": alo,
                    "delta_accuracy_ci_high": ahi,
                    "delta_rho_lora_minus_ootb": point_rho[il] - point_rho[io],
                    "delta_rho_ci_low": rlo,
                    "delta_rho_ci_high": rhi,
                }
            )
    return pd.DataFrame(rows)

def family_mean_lora_table(meta, lookup, point_acc, point_rho, acc_boot, rho_boot):
    rows = []
    for family in FAMILY_ORDER:
        sizes = (
            meta[meta["family"] == family]
            .sort_values("size_order")["size"]
            .drop_duplicates()
            .tolist()
        )

        acc_deltas_boot = []
        rho_deltas_boot = []
        acc_deltas_point = []
        rho_deltas_point = []

        for size in sizes:
            io = lookup[(family, size, "OOTB")]
            il = lookup[(family, size, "LoRA")]
            acc_deltas_boot.append(acc_boot[:, il] - acc_boot[:, io])
            rho_deltas_boot.append(rho_boot[:, il] - rho_boot[:, io])
            acc_deltas_point.append(point_acc[il] - point_acc[io])
            rho_deltas_point.append(point_rho[il] - point_rho[io])

        mean_acc_boot = np.stack(acc_deltas_boot, axis=1).mean(axis=1)
        mean_rho_boot = np.stack(rho_deltas_boot, axis=1).mean(axis=1)
        alo, ahi = ci95(mean_acc_boot)
        rlo, rhi = ci95(mean_rho_boot)

        rows.append(
            {
                "family": family,
                "mean_delta_accuracy": float(np.mean(acc_deltas_point)),
                "mean_delta_accuracy_ci_low": alo,
                "mean_delta_accuracy_ci_high": ahi,
                "mean_delta_rho": float(np.mean(rho_deltas_point)),
                "mean_delta_rho_ci_low": rlo,
                "mean_delta_rho_ci_high": rhi,
            }
        )
    return pd.DataFrame(rows)

def qwen_diagnostics_table(lookup, point_acc, point_rho, acc_boot, rho_boot):
    rows = []
    for condition in ["OOTB", "LoRA"]:
        for size_a, size_b in [("1.5B", "3B"), ("3B", "7B")]:
            ia = lookup[("Qwen", size_a, condition)]
            ib = lookup[("Qwen", size_b, condition)]

            d_acc_boot = acc_boot[:, ib] - acc_boot[:, ia]
            d_rho_boot = rho_boot[:, ib] - rho_boot[:, ia]
            alo, ahi = ci95(d_acc_boot)
            rlo, rhi = ci95(d_rho_boot)

            rows.append(
                {
                    "condition": condition,
                    "from_checkpoint": size_a,
                    "to_checkpoint": size_b,
                    "delta_accuracy": point_acc[ib] - point_acc[ia],
                    "delta_accuracy_ci_low": alo,
                    "delta_accuracy_ci_high": ahi,
                    "delta_rho": point_rho[ib] - point_rho[ia],
                    "delta_rho_ci_low": rlo,
                    "delta_rho_ci_high": rhi,
                }
            )
    return pd.DataFrame(rows)

def fmt_delta(x, lo, hi):
    return f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Gold file:       {args.gold}")
    print(f"Prediction dir:  {args.pred_dir}")
    print(f"Output dir:      {args.out_dir}")
    print(f"Bootstrap reps:  {args.n_boot:,}")
    print(f"Seed:            {args.seed}")
    print(f"Stratified task: {args.stratify_by_task}")
    print()

    ids, y_soft, y_bin, tasks, scores, meta = load_data(args.gold, args.pred_dir)
    print(f"Validated {len(ids):,} gold examples and {scores.shape[1]} controlled model-condition files.")
    print("Task counts:", dict(pd.Series(tasks).value_counts().sort_index()))

    point_acc, point_rho = point_metrics(scores, y_soft, y_bin)

    # Useful sanity check against the published DeBERTa-base OOTB numbers.
    lookup = build_lookup(meta)
    deb_base = lookup[("DeBERTa CE", "base", "OOTB")]
    print(
        "Sanity check, DeBERTa-base OOTB: "
        f"accuracy={point_acc[deb_base]:.6f}, rho={point_rho[deb_base]:.6f} "
        "(paper rounds to 0.769 / 0.640)"
    )
    print()

    stratify_tasks = tasks if args.stratify_by_task else None
    acc_boot, rho_boot = bootstrap_metrics(
        scores=scores,
        y_soft=y_soft,
        y_bin=y_bin,
        n_boot=args.n_boot,
        seed=args.seed,
        batch_size=args.batch_size,
        tasks=stratify_tasks,
    )

    checkpoints = checkpoint_table(meta, point_acc, point_rho, acc_boot, rho_boot)
    endpoints = smallest_to_largest_table(meta, point_acc, point_rho, acc_boot, rho_boot)
    lora_pairs = lora_vs_ootb_table(meta, lookup, point_acc, point_rho, acc_boot, rho_boot)
    family_lora = family_mean_lora_table(meta, lookup, point_acc, point_rho, acc_boot, rho_boot)
    qwen = qwen_diagnostics_table(lookup, point_acc, point_rho, acc_boot, rho_boot)

    checkpoints.to_csv(args.out_dir / "checkpoint_metrics_with_ci.csv", index=False)
    endpoints.to_csv(args.out_dir / "smallest_to_largest_deltas.csv", index=False)
    lora_pairs.to_csv(args.out_dir / "lora_minus_ootb_deltas.csv", index=False)
    family_lora.to_csv(args.out_dir / "family_mean_lora_gain.csv", index=False)
    qwen.to_csv(args.out_dir / "qwen_3b_diagnostics.csv", index=False)

    print("\n=== Smallest -> largest delta rho (main camera-ready table candidate) ===")
    for _, r in endpoints.iterrows():
        print(
            f"{r['family']:11s} {r['condition']:4s}: "
            + fmt_delta(r["delta_rho"], r["delta_rho_ci_low"], r["delta_rho_ci_high"])
        )

    print("\n=== Family mean LoRA - OOTB delta rho ===")
    for _, r in family_lora.iterrows():
        print(
            f"{r['family']:11s}: "
            + fmt_delta(r["mean_delta_rho"], r["mean_delta_rho_ci_low"], r["mean_delta_rho_ci_high"])
        )

    print("\n=== Qwen adjacent-checkpoint diagnostics ===")
    for _, r in qwen.iterrows():
        print(
            f"{r['condition']:4s} {r['from_checkpoint']} -> {r['to_checkpoint']}: "
            f"dAcc={fmt_delta(r['delta_accuracy'], r['delta_accuracy_ci_low'], r['delta_accuracy_ci_high'])}; "
            f"dRho={fmt_delta(r['delta_rho'], r['delta_rho_ci_low'], r['delta_rho_ci_high'])}"
        )

    print(f"\nSaved CSV outputs to: {args.out_dir.resolve()}")
    print(
        "Interpretation reminder: these CIs quantify uncertainty from the finite SHROOM test sample. "
        "They do NOT quantify variation across LoRA training seeds."
    )

if __name__ == "__main__":
    main()