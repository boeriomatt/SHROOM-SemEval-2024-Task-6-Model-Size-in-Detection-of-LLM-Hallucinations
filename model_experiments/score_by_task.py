import argparse
import collections
import json
import pathlib
from typing import Any
from scipy.stats import spearmanr

def safe_spearman(x: list[float], y: list[float]) -> float:
    """Return Spearman rho, allowing NaN if the input is constant or too small."""
    rho = spearmanr(x, y)[0]
    return float(rho) if rho == rho else float('nan')

def load_json(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute overall and per-task SHROOM metrics (PG / MT / DM)."
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

    valid_labels = {"Hallucination", "Not Hallucination"}
    for i, sub in enumerate(sub_data):
        if sub["label"] not in valid_labels:
            raise ValueError(f"Invalid label at submission item {i}: {sub['label']}")

    pairs = list(zip(sub_data, ref_data))
    task_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for sub, ref in pairs:
        task_groups[str(ref["task"])].append((sub, ref))

    results: dict[str, float | int] = {}

    # Overall metrics
    results["overall_n"] = len(pairs)
    results["overall_acc"] = sum(
        sub["label"] == ref["label"] for sub, ref in pairs
    ) / len(pairs)
    results["overall_rho"] = safe_spearman(
        [float(sub["p(Hallucination)"]) for sub, _ in pairs],
        [float(ref["p(Hallucination)"]) for _, ref in pairs],
    )

    # Per-task metrics
    for task in ["PG", "MT", "DM"]:
        items = task_groups.get(task, [])
        if not items:
            continue

        results[f"{task}_n"] = len(items)
        results[f"{task}_acc"] = sum(
            sub["label"] == ref["label"] for sub, ref in items
        ) / len(items)
        results[f"{task}_rho"] = safe_spearman(
            [float(sub["p(Hallucination)"]) for sub, _ in items],
            [float(ref["p(Hallucination)"]) for _, ref in items],
        )

    # Print to terminal
    ordered_keys = [
        "overall_n",
        "overall_acc",
        "overall_rho",
        "PG_n",
        "PG_acc",
        "PG_rho",
        "MT_n",
        "MT_acc",
        "MT_rho",
        "DM_n",
        "DM_acc",
        "DM_rho",
    ]

    for key in ordered_keys:
        if key in results:
            print(f"{key}:{results[key]}")

    # Optional txt output
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for key in ordered_keys:
                if key in results:
                    f.write(f"{key}:{results[key]}\n")

    # Optional json output
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()