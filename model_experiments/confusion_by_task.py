import argparse
import collections
import json
import pathlib
from typing import Any

HALL = "Hallucination"
NONHALL = "Not Hallucination"
VALID_LABELS = {HALL, NONHALL}

def load_json(path: pathlib.Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def compute_confusion(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, float | int]:
    tp = sum(1 for sub, ref in items if ref["label"] == HALL and sub["label"] == HALL)
    fn = sum(1 for sub, ref in items if ref["label"] == HALL and sub["label"] == NONHALL)
    fp = sum(1 for sub, ref in items if ref["label"] == NONHALL and sub["label"] == HALL)
    tn = sum(1 for sub, ref in items if ref["label"] == NONHALL and sub["label"] == NONHALL)

    n = len(items)
    pred_hall_n = tp + fp
    pred_nonhall_n = tn + fn

    return {
        "n": n,
        "gold_hall_n": tp + fn,
        "gold_nonhall_n": tn + fp,
        "pred_hall_n": pred_hall_n,
        "pred_nonhall_n": pred_nonhall_n,
        "pred_hall_rate": pred_hall_n / n if n else float("nan"),
        "pred_nonhall_rate": pred_nonhall_n / n if n else float("nan"),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute confusion-matrix counts and predicted class proportions overall and by SHROOM task."
    )
    parser.add_argument("submission_file", type=pathlib.Path)
    parser.add_argument("reference_file", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    parser.add_argument("--json-output", type=pathlib.Path, default=None)
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
        if sub["label"] not in VALID_LABELS:
            raise ValueError(f"Invalid prediction label at item {i}: {sub['label']}")

    for i, ref in enumerate(ref_data):
        if "task" not in ref or "label" not in ref:
            raise ValueError(f"Reference item {i} must contain 'task' and 'label'")
        if ref["label"] not in VALID_LABELS:
            raise ValueError(f"Invalid gold label at item {i}: {ref['label']}")

    pairs = list(zip(sub_data, ref_data))
    task_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for sub, ref in pairs:
        task_groups[str(ref["task"])].append((sub, ref))

    results: dict[str, float | int] = {}

    overall = compute_confusion(pairs)
    for key, value in overall.items():
        results[f"overall_{key}"] = value

    for task in ["PG", "MT", "DM"]:
        items = task_groups.get(task, [])
        if not items:
            continue
        metrics = compute_confusion(items)
        for key, value in metrics.items():
            results[f"{task}_{key}"] = value

    ordered_keys = [
        "overall_n",
        "overall_gold_hall_n",
        "overall_gold_nonhall_n",
        "overall_pred_hall_n",
        "overall_pred_nonhall_n",
        "overall_pred_hall_rate",
        "overall_pred_nonhall_rate",
        "overall_TP",
        "overall_FN",
        "overall_FP",
        "overall_TN",
        "PG_n",
        "PG_gold_hall_n",
        "PG_gold_nonhall_n",
        "PG_pred_hall_n",
        "PG_pred_nonhall_n",
        "PG_pred_hall_rate",
        "PG_pred_nonhall_rate",
        "PG_TP",
        "PG_FN",
        "PG_FP",
        "PG_TN",
        "MT_n",
        "MT_gold_hall_n",
        "MT_gold_nonhall_n",
        "MT_pred_hall_n",
        "MT_pred_nonhall_n",
        "MT_pred_hall_rate",
        "MT_pred_nonhall_rate",
        "MT_TP",
        "MT_FN",
        "MT_FP",
        "MT_TN",
        "DM_n",
        "DM_gold_hall_n",
        "DM_gold_nonhall_n",
        "DM_pred_hall_n",
        "DM_pred_nonhall_n",
        "DM_pred_hall_rate",
        "DM_pred_nonhall_rate",
        "DM_TP",
        "DM_FN",
        "DM_FP",
        "DM_TN",
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