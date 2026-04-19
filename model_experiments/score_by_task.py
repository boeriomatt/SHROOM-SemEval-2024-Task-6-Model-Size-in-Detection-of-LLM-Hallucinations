import argparse
import collections
import json
import pathlib
from typing import Any
from scipy.stats import spearmanr

HALL = "Hallucination"
NONHALL = "Not Hallucination"
VALID_LABELS = {HALL, NONHALL}

def safe_spearman(x: list[float], y: list[float]) -> float:
    """Return Spearman rho, allowing NaN if the input is constant or too small."""
    rho = spearmanr(x, y)[0]
    return float(rho) if rho == rho else float("nan")

def load_json(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def compute_metrics(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, float | int]:
    sub_items = [sub for sub, _ in items]
    ref_items = [ref for _, ref in items]

    hall_items = [(sub, ref) for sub, ref in items if ref["label"] == HALL]
    nonhall_items = [(sub, ref) for sub, ref in items if ref["label"] == NONHALL]

    def class_recall(class_items: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
        if not class_items:
            return float("nan")
        return sum(sub["label"] == ref["label"] for sub, ref in class_items) / len(class_items)

    hall_recall = class_recall(hall_items)
    nonhall_recall = class_recall(nonhall_items)

    if hall_recall == hall_recall and nonhall_recall == nonhall_recall:
        balanced_acc = (hall_recall + nonhall_recall) / 2
    else:
        balanced_acc = float("nan")

    return {
        "n": len(items),
        "hall_n": len(hall_items),
        "nonhall_n": len(nonhall_items),
        "acc": sum(sub["label"] == ref["label"] for sub, ref in items) / len(items),
        "rho": safe_spearman(
            [float(sub["p(Hallucination)"]) for sub in sub_items],
            [float(ref["p(Hallucination)"]) for ref in ref_items],
        ),
        "hall_recall": hall_recall,
        "nonhall_recall": nonhall_recall,
        "balanced_acc": balanced_acc,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute overall and per-task SHROOM metrics (PG / MT / DM), including class-wise recall."
    )
    parser.add_argument(
        "submission_file",
        type=pathlib.Path,
        help="Path to one prediction json file, e.g. outputs/predictions/archive/val.model-agnostic__MODEL.json",
    )
    parser.add_argument(
        "reference_file",
        type=pathlib.Path,
        help="Path to one reference json file, e.g. data/SHROOM_dev-v2/val.model-agnostic.json",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional path to save metrics as a txt file.",
    )
    parser.add_argument(
        "--json-output",
        type=pathlib.Path,
        default=None,
        help="Optional path to save metrics as json.",
    )
    args = parser.parse_args()

    sub_data = load_json(args.submission_file)
    ref_data = load_json(args.reference_file)

    if len(sub_data) != len(ref_data):
        raise ValueError(
            f"Mismatched lengths: submission has {len(sub_data)} items, reference has {len(ref_data)}"
        )

    expected_keys = {"label", "p(Hallucination)"}
    for i, sub in enumerate(sub_data):
        missing = expected_keys - set(sub.keys())
        if missing:
            raise ValueError(f"Submission item {i} is missing keys: {sorted(missing)}")

    for i, ref in enumerate(ref_data):
        if "task" not in ref or "label" not in ref or "p(Hallucination)" not in ref:
            raise ValueError(
                f"Reference item {i} must contain 'task', 'label', and 'p(Hallucination)'"
            )

    for i, sub in enumerate(sub_data):
        if sub["label"] not in VALID_LABELS:
            raise ValueError(f"Invalid label at submission item {i}: {sub['label']}")

    pairs = list(zip(sub_data, ref_data))
    task_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for sub, ref in pairs:
        task_groups[str(ref["task"])].append((sub, ref))

    results: dict[str, float | int] = {}

    overall_metrics = compute_metrics(pairs)
    for metric_name, metric_value in overall_metrics.items():
        results[f"overall_{metric_name}"] = metric_value

    for task in ["PG", "MT", "DM"]:
        items = task_groups.get(task, [])
        if not items:
            continue
        task_metrics = compute_metrics(items)
        for metric_name, metric_value in task_metrics.items():
            results[f"{task}_{metric_name}"] = metric_value

    ordered_keys = [
        "overall_n",
        "overall_hall_n",
        "overall_nonhall_n",
        "overall_acc",
        "overall_rho",
        "overall_hall_recall",
        "overall_nonhall_recall",
        "overall_balanced_acc",
        "PG_n",
        "PG_hall_n",
        "PG_nonhall_n",
        "PG_acc",
        "PG_rho",
        "PG_hall_recall",
        "PG_nonhall_recall",
        "PG_balanced_acc",
        "MT_n",
        "MT_hall_n",
        "MT_nonhall_n",
        "MT_acc",
        "MT_rho",
        "MT_hall_recall",
        "MT_nonhall_recall",
        "MT_balanced_acc",
        "DM_n",
        "DM_hall_n",
        "DM_nonhall_n",
        "DM_acc",
        "DM_rho",
        "DM_hall_recall",
        "DM_nonhall_recall",
        "DM_balanced_acc",
    ]

    for key in ordered_keys:
        if key in results:
            print(f"{key}:{results[key]}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for key in ordered_keys:
                if key in results:
                    f.write(f"{key}:{results[key]}\n")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()