#!/usr/bin/env python3
"""
Create OOTB vs. fine-tuned comparison tables from experiment metadata.

Expected project layout, when run from the project root:

    outputs/
      metadata/
        *.json
      tables/                       # created by this script

Outputs:
    outputs/tables/ootb_vs_finetuned_comparison.csv
    outputs/tables/ootb_vs_finetuned_comparison.md
    outputs/tables/ootb_vs_finetuned_family_summary.csv
    outputs/tables/ootb_vs_finetuned_family_summary.md
    outputs/tables/normalized_metadata_rows.csv

The pairing logic matches OOTB and fine-tuned rows by:
    family + subfamily + size_rung

Examples:
    FLAN + main + base
    DeBERTa + cross_encoder + small
    Qwen + main + 1.5b
    Gemma + main + 4b

Run:
    python summarize_ootb_vs_finetuned_comparison.py

Optional:
    python summarize_ootb_vs_finetuned_comparison.py --selection best_rho
    python summarize_ootb_vs_finetuned_comparison.py --metadata-dir outputs/metadata --output-dir outputs/tables
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_METADATA_DIR = Path("outputs/metadata")
DEFAULT_OUTPUT_DIR = Path("outputs/tables")

FINETUNED_MARKERS = (
    "finetuned",
    "fine_tuned",
    "fine-tuned",
    "lora",
    "adapter",
)

CSV_FIELDNAMES = [
    "family",
    "subfamily",
    "size_rung",
    "base_model_name",
    "ootb_metadata_file",
    "finetuned_metadata_file",
    "ootb_model_type",
    "finetuned_model_type",
    "ootb_parameter_count",
    "finetuned_parameter_count",
    "ootb_accuracy",
    "finetuned_accuracy",
    "delta_accuracy_ft_minus_ootb",
    "pct_delta_accuracy_ft_vs_ootb",
    "ootb_rho",
    "finetuned_rho",
    "delta_rho_ft_minus_ootb",
    "pct_delta_rho_ft_vs_ootb",
    "ootb_mean_latency_seconds_per_example",
    "finetuned_mean_latency_seconds_per_example",
    "delta_mean_latency_seconds_ft_minus_ootb",
    "latency_ratio_ft_div_ootb",
    "ootb_total_inference_runtime_seconds",
    "finetuned_total_inference_runtime_seconds",
    "finetuned_total_training_runtime_seconds",
    "ootb_num_examples",
    "finetuned_num_examples",
]

NORMALIZED_FIELDNAMES = [
    "adaptation",
    "family",
    "subfamily",
    "size_rung",
    "base_model_name",
    "model_name",
    "model_type",
    "accuracy",
    "rho",
    "parameter_count",
    "mean_latency_seconds_per_example",
    "total_inference_runtime_seconds",
    "total_training_runtime_seconds",
    "num_examples",
    "timestamp",
    "metadata_file",
]

FAMILY_SUMMARY_FIELDNAMES = [
    "family",
    "subfamily",
    "n_paired_models",
    "mean_ootb_accuracy",
    "mean_finetuned_accuracy",
    "mean_delta_accuracy_ft_minus_ootb",
    "mean_ootb_rho",
    "mean_finetuned_rho",
    "mean_delta_rho_ft_minus_ootb",
    "mean_ootb_latency_seconds_per_example",
    "mean_finetuned_latency_seconds_per_example",
    "mean_latency_ratio_ft_div_ootb",
    "sum_finetuned_training_runtime_seconds",
]


def as_float(value: Any) -> float | None:
    """Safely coerce a value to float, preserving None for missing/unusable values."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_nested(data: dict[str, Any], *paths: str) -> Any:
    """Return the first non-None value found across dot-separated dictionary paths."""
    for path in paths:
        current: Any = data
        found = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found and current is not None:
            return current
    return None


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Accept common ISO variants, including trailing Z.
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def is_finetuned_row(data: dict[str, Any], model_type: str, model_name: str, path: Path) -> bool:
    """Detect whether metadata belongs to a fine-tuned/adapted run."""
    joined = " ".join(
        str(x).lower()
        for x in [
            model_type,
            model_name,
            data.get("base_model_name", ""),
            data.get("adaptation", ""),
            data.get("training_method", ""),
            path.as_posix(),
        ]
    )
    if any(marker in joined for marker in FINETUNED_MARKERS):
        return True
    # Many fine-tuning metadata files include final_prediction_metrics even when
    # participant_scores are also present. This is a fallback, not the main rule.
    if data.get("final_prediction_metrics") and data.get("base_model_name"):
        return True
    return False


def extract_model_names(data: dict[str, Any]) -> tuple[str, str, str]:
    """
    Return model_type, model_name, base_model_name.

    For fine-tuned runs, base_model_name is the checkpoint used for pairing.
    For OOTB runs, base_model_name falls back to model_name.
    """
    model_type = str(data.get("model_type") or "")
    model_name = str(data.get("model_name") or data.get("checkpoint_path") or data.get("safe_model_name") or "")
    base_model_name = str(data.get("base_model_name") or model_name)
    return model_type, model_name, base_model_name


def extract_scores(data: dict[str, Any]) -> tuple[float | None, float | None]:
    """Normalize accuracy and Spearman rho across OOTB and fine-tuned schemas."""
    accuracy = get_nested(
        data,
        "scores.agnostic_acc",
        "scores.acc_agnostic",
        "scores.accuracy",
        "participant_scores.agnostic_acc",
        "participant_scores.acc_agnostic",
        "participant_scores.accuracy",
        "final_prediction_metrics.accuracy",
        "final_prediction_metrics.acc",
        "final_prediction_metrics.test_acc",
        "metrics.accuracy",
        "accuracy",
        "agnostic_acc",
        "acc_agnostic",
    )
    rho = get_nested(
        data,
        "scores.agnostic_rho",
        "scores.rho_agnostic",
        "scores.spearman_rho",
        "participant_scores.agnostic_rho",
        "participant_scores.rho_agnostic",
        "participant_scores.spearman_rho",
        "final_prediction_metrics.rho",
        "final_prediction_metrics.spearman_rho",
        "final_prediction_metrics.test_rho",
        "metrics.rho",
        "metrics.spearman_rho",
        "rho",
        "agnostic_rho",
        "rho_agnostic",
    )
    return as_float(accuracy), as_float(rho)


def extract_costs(data: dict[str, Any]) -> dict[str, float | None]:
    """Normalize runtime/size fields across metadata schemas."""
    return {
        "parameter_count": as_float(
            get_nested(
                data,
                "computational_cost.parameter_count",
                "model_info.parameter_count",
                "parameter_count",
            )
        ),
        "mean_latency_seconds_per_example": as_float(
            get_nested(
                data,
                "computational_cost.mean_inference_latency_seconds_per_example",
                "computational_cost.mean_latency_seconds_per_example",
                "computational_cost.mean_latency_seconds",
                "inference_metrics.mean_inference_latency_seconds_per_example",
                "inference_metrics.mean_latency_seconds_per_example",
                "mean_inference_latency_seconds_per_example",
                "mean_latency_seconds_per_example",
                "mean_latency_seconds",
            )
        ),
        "total_inference_runtime_seconds": as_float(
            get_nested(
                data,
                "computational_cost.total_inference_runtime_seconds",
                "inference_metrics.total_inference_runtime_seconds",
                "total_inference_runtime_seconds",
            )
        ),
        "total_training_runtime_seconds": as_float(
            get_nested(
                data,
                "computational_cost.total_training_runtime_seconds",
                "training_metrics.total_training_runtime_seconds",
                "train_metrics.total_training_runtime_seconds",
                "train_results.total_training_runtime_seconds",
                "train_results.train_runtime",
                "trainer_metrics.train_runtime",
                "train_runtime_seconds",
                "total_training_runtime_seconds",
            )
        ),
    }


def infer_family(model_type: str, base_model_name: str) -> str:
    text = f"{model_type} {base_model_name}".lower()
    if "flan" in text or "t5" in text:
        return "flan"
    if "deberta" in text:
        return "deberta"
    if "qwen" in text:
        return "qwen"
    if "gemma" in text:
        return "gemma"
    return model_type.lower() or "unknown"


def infer_subfamily(family: str, base_model_name: str) -> str:
    name = base_model_name.lower()
    if family == "deberta":
        if name.startswith("cross-encoder/nli-deberta-v3-") or "cross-encoder" in name:
            return "cross_encoder"
        return "other_nli"
    return "main"


def short_model_label(base_model_name: str) -> str:
    """Return compact size/rung label used to pair OOTB and fine-tuned rows."""
    name = (base_model_name or "").lower().replace("_", "-")

    if "flan-t5-" in name:
        return name.split("flan-t5-")[-1].split("/")[-1]

    if "cross-encoder/nli-deberta-v3-" in name:
        return name.split("cross-encoder/nli-deberta-v3-")[-1]

    if "deberta-v3-base-tasksource-nli" in name:
        return "tasksource-base"

    if "deberta-v3-large-mnli-fever-anli" in name:
        return "ml-fever-anli-large"

    if "qwen2.5-" in name and "-instruct" in name:
        return name.split("qwen2.5-")[-1].replace("-instruct", "")

    if "qwen3-" in name and "-instruct" in name:
        return name.split("qwen3-")[-1].replace("-instruct", "")

    if "gemma-3-" in name and "-it" in name:
        return name.split("gemma-3-")[-1].replace("-it", "")

    if "gemma-2-" in name and "-it" in name:
        return name.split("gemma-2-")[-1].replace("-it", "")

    # Common fallback for local fine-tuned checkpoint names, if base_model_name
    # was not stored and only a safe model/checkpoint name is available.
    for marker in [
        "flan-lora-finetuned-",
        "gemma-lora-finetuned-",
        "qwen-lora-finetuned-",
        "deberta-lora-finetuned-",
    ]:
        if marker in name:
            tail = name.split(marker)[-1]
            return (
                tail.replace("-seq2seq-verbalizer", "")
                .replace("-causal-lm-verbalizer", "")
                .replace("-binary-classifier", "")
            )

    return base_model_name.split("/")[-1] if base_model_name else "unknown"


def normalize_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    model_type, model_name, base_model_name = extract_model_names(data)
    adaptation = "finetuned" if is_finetuned_row(data, model_type, model_name, path) else "ootb"
    family = infer_family(model_type, base_model_name)
    subfamily = infer_subfamily(family, base_model_name)
    accuracy, rho = extract_scores(data)
    costs = extract_costs(data)
    timestamp = str(get_nested(data, "timestamp", "run_timestamp", "finished_at", "created_at") or "")

    return {
        "adaptation": adaptation,
        "family": family,
        "subfamily": subfamily,
        "size_rung": short_model_label(base_model_name),
        "base_model_name": base_model_name,
        "model_name": model_name,
        "model_type": model_type,
        "accuracy": accuracy,
        "rho": rho,
        "parameter_count": costs["parameter_count"],
        "mean_latency_seconds_per_example": costs["mean_latency_seconds_per_example"],
        "total_inference_runtime_seconds": costs["total_inference_runtime_seconds"],
        "total_training_runtime_seconds": costs["total_training_runtime_seconds"],
        "num_examples": as_int(get_nested(data, "num_examples", "n_examples", "dataset.num_examples")),
        "timestamp": timestamp,
        "timestamp_dt": parse_timestamp(timestamp),
        "metadata_file": str(path),
        "metadata_mtime": path.stat().st_mtime,
    }


def load_metadata(metadata_dir: Path) -> list[dict[str, Any]]:
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {metadata_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(metadata_dir.glob("*.json")):
        try:
            rows.append(normalize_row(path))
        except Exception as exc:  # noqa: BLE001 - keep going and report bad metadata files.
            print(f"WARNING: skipping {path}: {exc}")
    return rows


def sort_key_for_selection(row: dict[str, Any], selection: str) -> tuple[Any, ...]:
    """Return a key where max(key) is the selected preferred row."""
    timestamp = row.get("timestamp_dt") or datetime.min
    mtime = row.get("metadata_mtime") or 0.0

    if selection == "latest":
        return (timestamp, mtime)
    if selection == "best_rho":
        return (row.get("rho") if row.get("rho") is not None else float("-inf"), timestamp, mtime)
    if selection == "best_acc":
        return (row.get("accuracy") if row.get("accuracy") is not None else float("-inf"), timestamp, mtime)
    if selection == "fastest":
        latency = row.get("mean_latency_seconds_per_example")
        return (-(latency if latency is not None else float("inf")), timestamp, mtime)
    raise ValueError(f"Unsupported selection: {selection}")


def select_one_row(rows: list[dict[str, Any]], selection: str) -> dict[str, Any]:
    if len(rows) == 1:
        return rows[0]
    return max(rows, key=lambda row: sort_key_for_selection(row, selection))


TEXT_SIZE_ORDER = {
    "xxsmall": 0.0,
    "xsmall": 1.0,
    "small": 2.0,
    "base": 3.0,
    "large": 4.0,
    "xl": 5.0,
    "xxl": 6.0,
}


def size_rung_sort_value(size_rung: Any) -> float:
    """
    Fallback size ordering when parameter_count is missing.

    Prefer real parameter_count whenever it is available. This parser only exists
    so labels such as xsmall/small/base/large/xl, 270m, 1.5b, or tasksource-base
    still sort in a reasonable model-size order if a metadata file lacks params.
    """
    label = str(size_rung or "").lower().strip().replace("_", "-")
    if not label:
        return float("inf")

    # Handles labels like 270m, 0.5b, 1.5b, 7b, etc.
    numeric_match = re.search(r"(\d+(?:\.\d+)?)\s*([kmb])\b", label)
    if numeric_match:
        value = float(numeric_match.group(1))
        suffix = numeric_match.group(2)
        multiplier = {"k": 1_000.0, "m": 1_000_000.0, "b": 1_000_000_000.0}[suffix]
        return value * multiplier

    # Handles textual rungs and compound labels such as tasksource-base or
    # ml-fever-anli-large.
    for token, order in TEXT_SIZE_ORDER.items():
        if token in label:
            return order

    return float("inf")


def first_numeric(*values: Any) -> float | None:
    for value in values:
        numeric = as_float(value)
        if numeric is not None:
            return numeric
    return None


def comparison_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """
    Sort paired comparison rows by family, subfamily, then actual model size.

    The paired table has both OOTB and fine-tuned parameter counts. OOTB params
    are the primary key because they reflect the base model size; fine-tuned
    params are used as a fallback, then the size_rung parser if params are absent.
    """
    size_value = first_numeric(
        row.get("ootb_parameter_count"),
        row.get("finetuned_parameter_count"),
    )
    if size_value is None:
        size_value = size_rung_sort_value(row.get("size_rung"))

    return (
        str(row.get("family") or ""),
        str(row.get("subfamily") or ""),
        size_value,
        str(row.get("size_rung") or ""),
    )


def normalized_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """
    Sort normalized rows by family/subfamily/model size, with OOTB before fine-tuned.
    """
    size_value = first_numeric(row.get("parameter_count"))
    if size_value is None:
        size_value = size_rung_sort_value(row.get("size_rung"))

    adaptation_order = {"ootb": 0, "finetuned": 1}.get(str(row.get("adaptation") or ""), 9)

    return (
        str(row.get("family") or ""),
        str(row.get("subfamily") or ""),
        size_value,
        str(row.get("size_rung") or ""),
        adaptation_order,
        str(row.get("base_model_name") or ""),
    )


def family_summary_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("family") or ""),
        str(row.get("subfamily") or ""),
    )


def pair_rows(rows: list[dict[str, Any]], selection: str, include_unpaired: bool) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        key = (row["family"], row["subfamily"], row["size_rung"])
        grouped[key][row["adaptation"]].append(row)

    comparison_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for key in sorted(grouped):
        family, subfamily, size_rung = key
        ootb_candidates = grouped[key].get("ootb", [])
        ft_candidates = grouped[key].get("finetuned", [])

        if len(ootb_candidates) > 1:
            warnings.append(
                f"Multiple OOTB rows for {family}/{subfamily}/{size_rung}; selected {selection}."
            )
        if len(ft_candidates) > 1:
            warnings.append(
                f"Multiple fine-tuned rows for {family}/{subfamily}/{size_rung}; selected {selection}."
            )

        ootb = select_one_row(ootb_candidates, selection) if ootb_candidates else None
        ft = select_one_row(ft_candidates, selection) if ft_candidates else None

        if ootb is None or ft is None:
            if include_unpaired:
                comparison_rows.append(build_comparison_row(family, subfamily, size_rung, ootb, ft))
            continue

        comparison_rows.append(build_comparison_row(family, subfamily, size_rung, ootb, ft))

    comparison_rows.sort(key=comparison_sort_key)
    return comparison_rows, warnings


def delta(ft_value: Any, ootb_value: Any) -> float | None:
    ft = as_float(ft_value)
    ootb = as_float(ootb_value)
    if ft is None or ootb is None:
        return None
    return ft - ootb


def pct_delta(ft_value: Any, ootb_value: Any) -> float | None:
    ft = as_float(ft_value)
    ootb = as_float(ootb_value)
    if ft is None or ootb is None or ootb == 0:
        return None
    return (ft - ootb) / abs(ootb) * 100.0


def ratio(numerator: Any, denominator: Any) -> float | None:
    num = as_float(numerator)
    den = as_float(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def build_comparison_row(
    family: str,
    subfamily: str,
    size_rung: str,
    ootb: dict[str, Any] | None,
    ft: dict[str, Any] | None,
) -> dict[str, Any]:
    base_model_name = ""
    if ft is not None:
        base_model_name = ft.get("base_model_name") or ""
    if not base_model_name and ootb is not None:
        base_model_name = ootb.get("base_model_name") or ""

    ootb_acc = ootb.get("accuracy") if ootb else None
    ft_acc = ft.get("accuracy") if ft else None
    ootb_rho = ootb.get("rho") if ootb else None
    ft_rho = ft.get("rho") if ft else None
    ootb_latency = ootb.get("mean_latency_seconds_per_example") if ootb else None
    ft_latency = ft.get("mean_latency_seconds_per_example") if ft else None

    return {
        "family": family,
        "subfamily": subfamily,
        "size_rung": size_rung,
        "base_model_name": base_model_name,
        "ootb_metadata_file": ootb.get("metadata_file") if ootb else "",
        "finetuned_metadata_file": ft.get("metadata_file") if ft else "",
        "ootb_model_type": ootb.get("model_type") if ootb else "",
        "finetuned_model_type": ft.get("model_type") if ft else "",
        "ootb_parameter_count": ootb.get("parameter_count") if ootb else None,
        "finetuned_parameter_count": ft.get("parameter_count") if ft else None,
        "ootb_accuracy": ootb_acc,
        "finetuned_accuracy": ft_acc,
        "delta_accuracy_ft_minus_ootb": delta(ft_acc, ootb_acc),
        "pct_delta_accuracy_ft_vs_ootb": pct_delta(ft_acc, ootb_acc),
        "ootb_rho": ootb_rho,
        "finetuned_rho": ft_rho,
        "delta_rho_ft_minus_ootb": delta(ft_rho, ootb_rho),
        "pct_delta_rho_ft_vs_ootb": pct_delta(ft_rho, ootb_rho),
        "ootb_mean_latency_seconds_per_example": ootb_latency,
        "finetuned_mean_latency_seconds_per_example": ft_latency,
        "delta_mean_latency_seconds_ft_minus_ootb": delta(ft_latency, ootb_latency),
        "latency_ratio_ft_div_ootb": ratio(ft_latency, ootb_latency),
        "ootb_total_inference_runtime_seconds": ootb.get("total_inference_runtime_seconds") if ootb else None,
        "finetuned_total_inference_runtime_seconds": ft.get("total_inference_runtime_seconds") if ft else None,
        "finetuned_total_training_runtime_seconds": ft.get("total_training_runtime_seconds") if ft else None,
        "ootb_num_examples": ootb.get("num_examples") if ootb else None,
        "finetuned_num_examples": ft.get("num_examples") if ft else None,
    }


def mean(values: Iterable[Any]) -> float | None:
    numeric = [as_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def total(values: Iterable[Any]) -> float | None:
    numeric = [as_float(value) for value in values]
    numeric = [value for value in numeric if value is not None]
    if not numeric:
        return None
    return sum(numeric)


def make_family_summary(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        # Keep this family summary strictly paired, so missing/unpaired rows do not
        # distort mean deltas.
        if not row.get("ootb_metadata_file") or not row.get("finetuned_metadata_file"):
            continue
        grouped[(row["family"], row["subfamily"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (family, subfamily), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "family": family,
                "subfamily": subfamily,
                "n_paired_models": len(rows),
                "mean_ootb_accuracy": mean(row["ootb_accuracy"] for row in rows),
                "mean_finetuned_accuracy": mean(row["finetuned_accuracy"] for row in rows),
                "mean_delta_accuracy_ft_minus_ootb": mean(row["delta_accuracy_ft_minus_ootb"] for row in rows),
                "mean_ootb_rho": mean(row["ootb_rho"] for row in rows),
                "mean_finetuned_rho": mean(row["finetuned_rho"] for row in rows),
                "mean_delta_rho_ft_minus_ootb": mean(row["delta_rho_ft_minus_ootb"] for row in rows),
                "mean_ootb_latency_seconds_per_example": mean(row["ootb_mean_latency_seconds_per_example"] for row in rows),
                "mean_finetuned_latency_seconds_per_example": mean(row["finetuned_mean_latency_seconds_per_example"] for row in rows),
                "mean_latency_ratio_ft_div_ootb": mean(row["latency_ratio_ft_div_ootb"] for row in rows),
                "sum_finetuned_training_runtime_seconds": total(row["finetuned_total_training_runtime_seconds"] for row in rows),
            }
        )
    return summary_rows


def format_value(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def cleaned_for_csv(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for field in fieldnames:
        value = row.get(field)
        if isinstance(value, float):
            cleaned[field] = format_value(value, digits=6)
        elif value is None:
            cleaned[field] = ""
        else:
            cleaned[field] = value
    return cleaned


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(cleaned_for_csv(row, fieldnames))


def markdown_table(rows: list[dict[str, Any]], fieldnames: list[str], digits: int = 4) -> str:
    if not rows:
        return "No rows found.\n"

    headers = fieldnames
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [format_value(row.get(field), digits=digits).replace("|", "\\|") for field in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_comparison(path: Path, rows: list[dict[str, Any]]) -> None:
    compact_fields = [
        "family",
        "subfamily",
        "size_rung",
        "ootb_accuracy",
        "finetuned_accuracy",
        "delta_accuracy_ft_minus_ootb",
        "ootb_rho",
        "finetuned_rho",
        "delta_rho_ft_minus_ootb",
        "ootb_mean_latency_seconds_per_example",
        "finetuned_mean_latency_seconds_per_example",
        "latency_ratio_ft_div_ootb",
        "finetuned_total_training_runtime_seconds",
    ]
    text = (
        "# OOTB vs. fine-tuned comparison\n\n"
        "Rows are matched by `family + subfamily + size_rung`. "
        "Deltas are calculated as fine-tuned minus OOTB.\n\n"
        + markdown_table(rows, compact_fields, digits=4)
    )
    path.write_text(text, encoding="utf-8")


def write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    text = (
        "# OOTB vs. fine-tuned family summary\n\n"
        "Means are calculated across paired model rungs only. "
        "Deltas are calculated as fine-tuned minus OOTB before averaging.\n\n"
        + markdown_table(rows, FAMILY_SUMMARY_FIELDNAMES, digits=4)
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OOTB vs fine-tuned comparison tables from metadata JSON files.")
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--selection",
        choices=["latest", "best_rho", "best_acc", "fastest"],
        default="latest",
        help="How to select a single row when duplicate runs exist for the same family/subfamily/size/adaptation.",
    )
    parser.add_argument(
        "--include-unpaired",
        action="store_true",
        help="Include rows that only have OOTB or only have fine-tuned metadata. By default, only paired rows are written to the comparison table.",
    )
    args = parser.parse_args()

    rows = load_metadata(args.metadata_dir)
    if not rows:
        print(f"No metadata JSON files found in {args.metadata_dir}")
        return

    rows.sort(key=normalized_sort_key)

    comparison_rows, warnings = pair_rows(
        rows=rows,
        selection=args.selection,
        include_unpaired=args.include_unpaired,
    )
    summary_rows = make_family_summary(comparison_rows)
    summary_rows.sort(key=family_summary_sort_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(args.output_dir / "normalized_metadata_rows.csv", rows, NORMALIZED_FIELDNAMES)
    write_csv(args.output_dir / "ootb_vs_finetuned_comparison.csv", comparison_rows, CSV_FIELDNAMES)
    write_markdown_comparison(args.output_dir / "ootb_vs_finetuned_comparison.md", comparison_rows)
    write_csv(args.output_dir / "ootb_vs_finetuned_family_summary.csv", summary_rows, FAMILY_SUMMARY_FIELDNAMES)
    write_markdown_summary(args.output_dir / "ootb_vs_finetuned_family_summary.md", summary_rows)

    print(f"Loaded metadata rows: {len(rows)}")
    print(f"Paired comparison rows written: {len(comparison_rows)}")
    print(f"Family summary rows written: {len(summary_rows)}")
    print(f"Selection strategy for duplicate runs: {args.selection}")
    print(f"Saved outputs to: {args.output_dir}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
