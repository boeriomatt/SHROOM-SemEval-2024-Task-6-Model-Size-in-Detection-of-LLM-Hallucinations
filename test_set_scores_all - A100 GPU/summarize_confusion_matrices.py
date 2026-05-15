#!/usr/bin/env python3
"""
Generate confusion-matrix statistics for SHROOM model prediction files.

Default project layout, when run from the project root:

    data/
      SHROOM_test-labeled/
        test.model-agnostic.json
    outputs/
      predictions/
        archive/
          test.model-agnostic__<model>.json
      tables/                         # created by this script

Outputs:
    outputs/tables/confusion_matrix_stats_by_model_task.csv
    outputs/tables/confusion_matrix_stats_by_model_task.md
    outputs/tables/confusion_matrix_errors.csv

The script computes confusion-matrix counts and classification metrics for:
    - each prediction file / model
    - overall
    - each SHROOM task: DM, PG, MT

Positive class:
    Hallucination

Run:
    python summarize_confusion_matrices.py

Optional:
    python summarize_confusion_matrices.py \
        --dataset-path data/SHROOM_test-labeled/test.model-agnostic.json \
        --predictions-dir outputs/predictions/archive \
        --prediction-glob "test.model-agnostic*.json" \
        --output-dir outputs/tables

If you want to ignore the predicted label field and recompute labels from
p(Hallucination), use:

    python summarize_confusion_matrices.py --use-prob-threshold --threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DATASET_PATH = Path("data/SHROOM_test-labeled/test.model-agnostic.json")
DEFAULT_PREDICTIONS_DIR = Path("outputs/predictions/archive")
DEFAULT_OUTPUT_DIR = Path("outputs/tables")
DEFAULT_PREDICTION_GLOB = "test.model-agnostic*.json"

POSITIVE_LABEL = "Hallucination"
NEGATIVE_LABEL = "Not Hallucination"
TASK_ORDER = {"overall": 0, "DM": 1, "PG": 2, "MT": 3}

STATS_FIELDNAMES = [
    "model_key",
    "prediction_file",
    "family",
    "adaptation",
    "size_rung",
    "task",
    "n_dataset_in_scope",
    "n_predictions_matched_in_scope",
    "n_missing_predictions",
    "n_extra_predictions",
    "gold_hallucination",
    "gold_not_hallucination",
    "predicted_hallucination",
    "predicted_not_hallucination",
    "tp",
    "fp",
    "tn",
    "fn",
    "accuracy",
    "precision_hallucination",
    "recall_hallucination",
    "f1_hallucination",
    "precision_not_hallucination",
    "recall_not_hallucination",
    "f1_not_hallucination",
    "macro_f1",
    "balanced_accuracy",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "negative_predictive_value",
    "mcc",
    "gold_hallucination_rate",
    "predicted_hallucination_rate",
    "mean_predicted_p_hallucination",
    "mean_predicted_p_hallucination_when_gold_hallucination",
    "mean_predicted_p_hallucination_when_gold_not_hallucination",
]

ERROR_FIELDNAMES = [
    "model_key",
    "prediction_file",
    "family",
    "adaptation",
    "size_rung",
    "task",
    "id",
    "error_type",
    "gold_label",
    "predicted_label",
    "predicted_p_hallucination",
    "gold_p_hallucination",
    "src",
    "tgt",
    "hyp",
]

MD_FIELDNAMES = [
    "model_key",
    "task",
    "n_predictions_matched_in_scope",
    "tp",
    "fp",
    "tn",
    "fn",
    "accuracy",
    "precision_hallucination",
    "recall_hallucination",
    "f1_hallucination",
    "recall_not_hallucination",
    "macro_f1",
    "balanced_accuracy",
    "predicted_hallucination_rate",
]


def read_json_or_jsonl(path: Path) -> Any:
    """Read either a normal JSON file or JSONL file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse {path} as JSON or JSONL; bad line {line_no}") from exc
        return rows


def unwrap_records(obj: Any) -> list[dict[str, Any]]:
    """
    Normalize common prediction/dataset wrappers to a list of dictionaries.

    Supported:
        [ {...}, {...} ]
        { "predictions": [ {...}, ... ] }
        { "data": [ {...}, ... ] }
        { "examples": [ {...}, ... ] }
    """
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]

    if isinstance(obj, dict):
        for key in ("predictions", "data", "examples", "records", "rows"):
            value = obj.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]

    raise ValueError("Expected a JSON list or a dict containing a predictions/data/examples list.")


def record_id(record: dict[str, Any]) -> str:
    if "id" not in record:
        raise ValueError(f"Record is missing required 'id' field: {record}")
    return str(record["id"])


def load_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records = unwrap_records(read_json_or_jsonl(path))
    by_id: dict[str, dict[str, Any]] = {}

    for record in records:
        rid = record_id(record)
        if rid in by_id:
            raise ValueError(f"Duplicate id={rid!r} in {path}")
        by_id[rid] = record

    return by_id


def normalize_label_to_bool(label: Any) -> bool | None:
    """
    Return True for Hallucination, False for Not Hallucination, None if unknown.

    The checks are intentionally conservative because "Not Hallucination" contains
    the substring "Hallucination".
    """
    if label is None:
        return None

    text = str(label).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    negative_values = {
        "not hallucination",
        "non hallucination",
        "nonhallucination",
        "not hallucinated",
        "not hall",
        "no",
        "false",
        "0",
        "negative",
    }
    positive_values = {
        "hallucination",
        "hallucinated",
        "hall",
        "yes",
        "true",
        "1",
        "positive",
    }

    if text in negative_values:
        return False
    if text in positive_values:
        return True

    if text.startswith("not ") and "hallucination" in text:
        return False
    if text.startswith("non") and "hallucination" in text:
        return False

    return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def find_probability(record: dict[str, Any]) -> float | None:
    for key in (
        "p(Hallucination)",
        "p_hallucination",
        "p_hallucination_gold",
        "probability_hallucination",
        "hallucination_probability",
        "score",
    ):
        if key in record:
            return as_float(record.get(key))
    return None


def label_bool_from_record(
    record: dict[str, Any],
    *,
    use_prob_threshold: bool,
    threshold: float,
) -> bool:
    """
    Convert a prediction/gold record to boolean class.

    If use_prob_threshold=True, p(Hallucination) is preferred.
    Otherwise, the label field is preferred and probability is used as fallback.
    """
    if use_prob_threshold:
        probability = find_probability(record)
        if probability is not None:
            return probability >= threshold

    label_value = normalize_label_to_bool(record.get("label"))
    if label_value is not None:
        return label_value

    probability = find_probability(record)
    if probability is not None:
        return probability >= threshold

    raise ValueError(f"Could not infer label from record id={record.get('id')!r}: {record}")


def label_name(value: bool) -> str:
    return POSITIVE_LABEL if value else NEGATIVE_LABEL


def safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def mean(values: Iterable[float | None]) -> float | None:
    numeric = [value for value in values if value is not None and math.isfinite(value)]
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def mcc_score(tp: int, fp: int, tn: int, fn: int) -> float | None:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return None
    return ((tp * tn) - (fp * fn)) / denominator


def f1_from_counts(true_positive_for_class: int, predicted_positive_for_class_errors: int, missed_positive_for_class_errors: int) -> float | None:
    """
    Compute one-vs-rest F1 directly from counts.

    This intentionally returns 0.0 when a class has gold support but receives
    no predictions. That matches the usual sklearn-style zero_division=0
    behavior and avoids inflating macro-F1 for degenerate all-one-class models.

    For the Hallucination class:
        true_positive_for_class = TP
        predicted_positive_for_class_errors = FP
        missed_positive_for_class_errors = FN

    For the Not Hallucination class:
        true_positive_for_class = TN
        predicted_positive_for_class_errors = FN
        missed_positive_for_class_errors = FP
    """
    denominator = (2 * true_positive_for_class) + predicted_positive_for_class_errors + missed_positive_for_class_errors
    if denominator == 0:
        return None
    return (2 * true_positive_for_class) / denominator


def infer_model_key(prediction_path: Path, dataset_path: Path) -> str:
    """
    Strip the dataset stem from a prediction filename.

    Example:
        test.model-agnostic__cross-encoder__nli-deberta-v3-base.json
        -> cross-encoder__nli-deberta-v3-base
    """
    stem = prediction_path.stem
    dataset_stem = dataset_path.stem

    prefixes = [
        f"{dataset_stem}__",
        f"{dataset_stem}_",
    ]
    for prefix in prefixes:
        if stem.startswith(prefix):
            return stem[len(prefix) :]

    return stem


def infer_family(text: str) -> str:
    low = text.lower()
    if "flan" in low or "t5" in low:
        return "flan"
    if "deberta" in low or "cross-encoder" in low or "tasksource" in low:
        return "deberta"
    if "qwen" in low:
        return "qwen"
    if "gemma" in low:
        return "gemma"
    return "unknown"


def infer_adaptation(text: str) -> str:
    low = text.lower()
    if any(marker in low for marker in ("finetuned", "fine-tuned", "fine_tuned", "lora", "adapter")):
        return "finetuned"
    return "ootb"


def infer_size_rung(text: str) -> str:
    low = text.lower().replace("_", "-")

    # FLAN / T5
    for size in ("small", "base", "large", "xl", "xxl"):
        if re.search(rf"(?:flan-t5-|flan-|t5-){size}(?:\b|[-_/])", low):
            return size

    # DeBERTa cross-encoder
    for size in ("xsmall", "small", "base", "large"):
        if re.search(rf"deberta-v3-{size}(?:\b|[-_/])", low):
            return size
    if "tasksource" in low and "base" in low:
        return "tasksource-base"
    if "fever" in low and "anli" in low and "large" in low:
        return "ml-fever-anli-large"

    # Qwen / Gemma style sizes
    match = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?[bm])(?:\b|[-_/])", low)
    if match:
        return match.group(1)

    # Safe fallback for filenames like qwen2.5-0.5b-instruct.
    match = re.search(r"(?:qwen\d*(?:\.\d+)?-|gemma-\d+-)(\d+(?:\.\d+)?[bm])", low)
    if match:
        return match.group(1)

    return ""


def sort_size_value(size_rung: str) -> float:
    text = (size_rung or "").lower()
    order = {
        "xsmall": 1,
        "small": 2,
        "base": 3,
        "tasksource-base": 3,
        "large": 4,
        "ml-fever-anli-large": 4,
        "xl": 5,
        "xxl": 6,
    }
    if text in order:
        return float(order[text])

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([bm])", text)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        return value * (1_000_000_000 if unit == "b" else 1_000_000)

    return float("inf")


def calculate_stats_for_scope(
    *,
    model_key: str,
    prediction_file: str,
    family: str,
    adaptation: str,
    size_rung: str,
    task: str,
    gold_records: dict[str, dict[str, Any]],
    prediction_records: dict[str, dict[str, Any]],
    ids_in_scope: list[str],
    extra_predictions_total: int,
    threshold: float,
    use_prob_threshold: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tp = fp = tn = fn = 0
    missing = 0

    p_values: list[float | None] = []
    p_values_gold_pos: list[float | None] = []
    p_values_gold_neg: list[float | None] = []
    errors: list[dict[str, Any]] = []

    for rid in ids_in_scope:
        gold = gold_records[rid]
        pred = prediction_records.get(rid)
        if pred is None:
            missing += 1
            continue

        gold_bool = label_bool_from_record(gold, use_prob_threshold=False, threshold=threshold)
        pred_bool = label_bool_from_record(pred, use_prob_threshold=use_prob_threshold, threshold=threshold)
        pred_prob = find_probability(pred)

        if gold_bool and pred_bool:
            tp += 1
        elif (not gold_bool) and pred_bool:
            fp += 1
        elif (not gold_bool) and (not pred_bool):
            tn += 1
        elif gold_bool and (not pred_bool):
            fn += 1

        p_values.append(pred_prob)
        if gold_bool:
            p_values_gold_pos.append(pred_prob)
        else:
            p_values_gold_neg.append(pred_prob)

        if gold_bool != pred_bool:
            errors.append(
                {
                    "model_key": model_key,
                    "prediction_file": prediction_file,
                    "family": family,
                    "adaptation": adaptation,
                    "size_rung": size_rung,
                    "task": gold.get("task", ""),
                    "id": gold.get("id", rid),
                    "error_type": "FN" if gold_bool and not pred_bool else "FP",
                    "gold_label": label_name(gold_bool),
                    "predicted_label": label_name(pred_bool),
                    "predicted_p_hallucination": pred_prob,
                    "gold_p_hallucination": find_probability(gold),
                    "src": gold.get("src", ""),
                    "tgt": gold.get("tgt", ""),
                    "hyp": gold.get("hyp", ""),
                }
            )

    n_matched = tp + fp + tn + fn
    gold_pos = tp + fn
    gold_neg = tn + fp
    pred_pos = tp + fp
    pred_neg = tn + fn

    accuracy = safe_div(tp + tn, n_matched)
    precision_pos = safe_div(tp, tp + fp)
    recall_pos = safe_div(tp, tp + fn)
    precision_neg = safe_div(tn, tn + fn)
    recall_neg = safe_div(tn, tn + fp)
    # Compute class-specific F1 from counts rather than from precision/recall.
    # This keeps F1_hallucination = 0 when the model predicts no hallucinations
    # but the gold data contains hallucinations. Using precision=None would
    # otherwise cause macro_f1 to average over only the negative class.
    f1_pos = f1_from_counts(tp, fp, fn)
    f1_neg = f1_from_counts(tn, fn, fp)
    macro_f1 = mean([f1_pos, f1_neg])
    balanced_accuracy = mean([recall_pos, recall_neg])

    row = {
        "model_key": model_key,
        "prediction_file": prediction_file,
        "family": family,
        "adaptation": adaptation,
        "size_rung": size_rung,
        "task": task,
        "n_dataset_in_scope": len(ids_in_scope),
        "n_predictions_matched_in_scope": n_matched,
        "n_missing_predictions": missing,
        "n_extra_predictions": extra_predictions_total if task == "overall" else "",
        "gold_hallucination": gold_pos,
        "gold_not_hallucination": gold_neg,
        "predicted_hallucination": pred_pos,
        "predicted_not_hallucination": pred_neg,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision_hallucination": precision_pos,
        "recall_hallucination": recall_pos,
        "f1_hallucination": f1_pos,
        "precision_not_hallucination": precision_neg,
        "recall_not_hallucination": recall_neg,
        "f1_not_hallucination": f1_neg,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "specificity": recall_neg,
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
        "negative_predictive_value": precision_neg,
        "mcc": mcc_score(tp, fp, tn, fn),
        "gold_hallucination_rate": safe_div(gold_pos, n_matched),
        "predicted_hallucination_rate": safe_div(pred_pos, n_matched),
        "mean_predicted_p_hallucination": mean(p_values),
        "mean_predicted_p_hallucination_when_gold_hallucination": mean(p_values_gold_pos),
        "mean_predicted_p_hallucination_when_gold_not_hallucination": mean(p_values_gold_neg),
    }

    return row, errors


def format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def cleaned_for_csv(row: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for field in fieldnames:
        value = row.get(field, "")
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

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        values = [format_value(row.get(field), digits=digits).replace("|", "\\|") for field in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown_stats(path: Path, rows: list[dict[str, Any]]) -> None:
    text = (
        "# Confusion matrix statistics by model and task\n\n"
        "Positive class: `Hallucination`. "
        "`TP` means gold hallucination predicted as hallucination; "
        "`FN` means gold hallucination predicted as not hallucination; "
        "`FP` means gold not hallucination predicted as hallucination; "
        "`TN` means gold not hallucination predicted as not hallucination.\n\n"
        + markdown_table(rows, MD_FIELDNAMES, digits=4)
    )
    path.write_text(text, encoding="utf-8")


def summarize_prediction_file(
    *,
    prediction_path: Path,
    dataset_path: Path,
    gold_records: dict[str, dict[str, Any]],
    gold_ids_by_task: dict[str, list[str]],
    threshold: float,
    use_prob_threshold: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    prediction_records = load_records_by_id(prediction_path)

    model_key = infer_model_key(prediction_path, dataset_path)
    family = infer_family(model_key)
    adaptation = infer_adaptation(model_key)
    size_rung = infer_size_rung(model_key)

    gold_ids = set(gold_records)
    pred_ids = set(prediction_records)
    missing_ids = sorted(gold_ids - pred_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))
    extra_ids = sorted(pred_ids - gold_ids, key=lambda x: int(x) if str(x).isdigit() else str(x))

    warnings: list[str] = []
    if missing_ids:
        warnings.append(f"{prediction_path.name}: missing {len(missing_ids)} predictions from dataset ids.")
    if extra_ids:
        warnings.append(f"{prediction_path.name}: has {len(extra_ids)} predictions whose ids are not in the dataset.")

    stats_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    scopes = [("overall", sorted(gold_records.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)))]
    for task in ("DM", "PG", "MT"):
        scopes.append((task, gold_ids_by_task.get(task, [])))

    for task, ids_in_scope in scopes:
        row, errors = calculate_stats_for_scope(
            model_key=model_key,
            prediction_file=prediction_path.name,
            family=family,
            adaptation=adaptation,
            size_rung=size_rung,
            task=task,
            gold_records=gold_records,
            prediction_records=prediction_records,
            ids_in_scope=ids_in_scope,
            extra_predictions_total=len(extra_ids),
            threshold=threshold,
            use_prob_threshold=use_prob_threshold,
        )
        stats_rows.append(row)
        # Avoid double-counting errors in confusion_matrix_errors.csv.
        # The overall scope contains the same item-level mistakes as the
        # task-specific scopes, so keeping only DM/PG/MT errors gives one
        # row per mistaken prediction while preserving the task label.
        if task != "overall":
            error_rows.extend(errors)

    return stats_rows, error_rows, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confusion-matrix stats for SHROOM prediction files.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--prediction-glob", default=DEFAULT_PREDICTION_GLOB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used when converting p(Hallucination) to a label.",
    )
    parser.add_argument(
        "--use-prob-threshold",
        action="store_true",
        help="Use p(Hallucination) >= threshold for predicted labels instead of the prediction file's label field.",
    )
    parser.add_argument(
        "--no-error-file",
        action="store_true",
        help="Do not write confusion_matrix_errors.csv.",
    )
    args = parser.parse_args()

    if not args.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
    if not args.predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {args.predictions_dir}")

    gold_records = load_records_by_id(args.dataset_path)

    gold_ids_by_task: dict[str, list[str]] = defaultdict(list)
    for rid, record in gold_records.items():
        task = str(record.get("task", "")).strip()
        if task:
            gold_ids_by_task[task].append(rid)

    for task in gold_ids_by_task:
        gold_ids_by_task[task].sort(key=lambda x: int(x) if str(x).isdigit() else str(x))

    prediction_paths = sorted(args.predictions_dir.glob(args.prediction_glob))
    # Avoid accidentally treating the gold dataset as a prediction file if the
    # user points predictions-dir at the dataset folder.
    prediction_paths = [path for path in prediction_paths if path.resolve() != args.dataset_path.resolve()]

    if not prediction_paths:
        print(f"No prediction files found in {args.predictions_dir} matching {args.prediction_glob!r}")
        return

    all_stats_rows: list[dict[str, Any]] = []
    all_error_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for prediction_path in prediction_paths:
        try:
            stats_rows, error_rows, file_warnings = summarize_prediction_file(
                prediction_path=prediction_path,
                dataset_path=args.dataset_path,
                gold_records=gold_records,
                gold_ids_by_task=gold_ids_by_task,
                threshold=args.threshold,
                use_prob_threshold=args.use_prob_threshold,
            )
        except Exception as exc:  # noqa: BLE001 - continue processing other model files.
            warnings.append(f"Skipping {prediction_path.name}: {exc}")
            continue

        all_stats_rows.extend(stats_rows)
        all_error_rows.extend(error_rows)
        warnings.extend(file_warnings)

    all_stats_rows.sort(
        key=lambda row: (
            row["family"],
            row["adaptation"],
            sort_size_value(row["size_rung"]),
            row["model_key"],
            TASK_ORDER.get(row["task"], 99),
        )
    )
    all_error_rows.sort(
        key=lambda row: (
            row["family"],
            row["adaptation"],
            sort_size_value(row["size_rung"]),
            row["model_key"],
            TASK_ORDER.get(row["task"], 99),
            row["error_type"],
            int(row["id"]) if str(row["id"]).isdigit() else str(row["id"]),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats_csv = args.output_dir / "confusion_matrix_stats_by_model_task.csv"
    stats_md = args.output_dir / "confusion_matrix_stats_by_model_task.md"
    errors_csv = args.output_dir / "confusion_matrix_errors.csv"

    write_csv(stats_csv, all_stats_rows, STATS_FIELDNAMES)
    write_markdown_stats(stats_md, all_stats_rows)

    if not args.no_error_file:
        write_csv(errors_csv, all_error_rows, ERROR_FIELDNAMES)

    print(f"Loaded gold dataset rows: {len(gold_records)}")
    print(f"Prediction files processed: {len(prediction_paths)}")
    print(f"Stats rows written: {len(all_stats_rows)}")
    print(f"Saved: {stats_csv}")
    print(f"Saved: {stats_md}")
    if not args.no_error_file:
        print(f"Saved: {errors_csv}")
    print(f"Positive class: {POSITIVE_LABEL}")
    print(f"Prediction labels from: {'p(Hallucination) threshold' if args.use_prob_threshold else 'label field, with probability fallback'}")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
