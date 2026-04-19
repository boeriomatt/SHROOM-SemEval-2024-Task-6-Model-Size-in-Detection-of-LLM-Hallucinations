import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import binomtest, spearmanr

METADATA_DIR = Path("outputs/metadata")
OUTPUT_CSV = Path("outputs/summary_table.csv")
OUTPUT_MD = Path("outputs/summary_table.md")
OUTPUT_STATS_JSON = Path("outputs/family_significance.json")
OUTPUT_STATS_MD = Path("outputs/family_significance.md")

def format_params(n: int | None) -> str:
    if n is None:
        return ""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return str(n)

def format_float(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}"

def format_signed(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:+.{decimals}f}"

def format_multiplier(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{decimals}f}x"

def format_p(value: float | None) -> str:
    if value is None:
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"

def infer_family(model_type: str, model_name: str) -> str:
    model_type = (model_type or "").lower()
    model_name_lower = (model_name or "").lower()

    if model_type == "flan" or "flan-t5" in model_name_lower:
        return "flan"
    if model_type == "deberta" or "deberta" in model_name_lower:
        return "deberta"
    if model_type.startswith("qwen") or "qwen" in model_name_lower:
        return "qwen"
    return model_type if model_type else "unknown"

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_metadata_files(metadata_dir: Path) -> list[dict]:
    rows = []

    for path in metadata_dir.glob("*.json"):
        data = load_json(path)

        scores = data.get("scores", {})
        cost = data.get("computational_cost", {})

        model_type = data.get("model_type", "")
        model_name = data.get("model_name", "")

        row = {
            "metadata_path": str(path),
            "family": infer_family(model_type, model_name),
            "model_type": model_type,
            "model_name": model_name,
            "parameter_count": cost.get("parameter_count"),
            "parameter_count_readable": format_params(cost.get("parameter_count")),
            "agnostic_acc": scores.get("agnostic_acc", scores.get("acc_agnostic")),
            "agnostic_rho": scores.get("agnostic_rho", scores.get("rho_agnostic")),
            "mean_latency_seconds": cost.get("mean_inference_latency_seconds_per_example"),
            "total_runtime_seconds": cost.get("total_inference_runtime_seconds"),
            "archive_prediction_path": data.get("archive_prediction_path"),
            "input_path": data.get("input_path"),
            "prompt_version": data.get("prompt_version"),
            "timestamp": data.get("timestamp"),
        }
        rows.append(row)

    return rows

def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda x: (
            x["family"],
            x["parameter_count"] is None,
            x["parameter_count"] if x["parameter_count"] is not None else float("inf"),
        ),
    )

def add_family_comparison_columns(rows: list[dict]) -> list[dict]:
    previous_by_family: dict[str, dict] = {}

    for row in rows:
        family = row["family"]
        previous = previous_by_family.get(family)

        comparable = (
            previous is not None
            and row.get("input_path") == previous.get("input_path")
            and row.get("prompt_version") == previous.get("prompt_version")
        )

        if not comparable:
            row["acc_gain_vs_prev_family"] = None
            row["rho_gain_vs_prev_family"] = None
            row["latency_multiplier_vs_prev_family"] = None
            row["params_multiplier_vs_prev_family"] = None
        else:
            acc = row["agnostic_acc"]
            prev_acc = previous["agnostic_acc"]
            rho = row["agnostic_rho"]
            prev_rho = previous["agnostic_rho"]
            latency = row["mean_latency_seconds"]
            prev_latency = previous["mean_latency_seconds"]
            params = row["parameter_count"]
            prev_params = previous["parameter_count"]

            row["acc_gain_vs_prev_family"] = (
                acc - prev_acc if acc is not None and prev_acc is not None else None
            )
            row["rho_gain_vs_prev_family"] = (
                rho - prev_rho if rho is not None and prev_rho is not None else None
            )
            row["latency_multiplier_vs_prev_family"] = (
                latency / prev_latency
                if latency is not None and prev_latency not in (None, 0)
                else None
            )
            row["params_multiplier_vs_prev_family"] = (
                params / prev_params
                if params is not None and prev_params not in (None, 0)
                else None
            )

        previous_by_family[family] = row

    return rows

def compute_accuracy_from_items(pred_items: list[dict], ref_items: list[dict]) -> float:
    correct = sum(pred["label"] == ref["label"] for pred, ref in zip(pred_items, ref_items))
    return correct / len(ref_items)

def compute_rho_from_items(pred_items: list[dict], ref_items: list[dict]) -> float:
    pred_scores = [float(pred["p(Hallucination)"]) for pred in pred_items]
    gold_scores = [float(ref["p(Hallucination)"]) for ref in ref_items]
    return float(spearmanr(pred_scores, gold_scores)[0])

def mcnemar_exact_test(pred_a: list[dict], pred_b: list[dict], ref_items: list[dict]) -> dict:
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0

    for pa, pb, ref in zip(pred_a, pred_b, ref_items):
        a_correct = pa["label"] == ref["label"]
        b_correct = pb["label"] == ref["label"]

        if a_correct and not b_correct:
            a_correct_b_wrong += 1
        elif not a_correct and b_correct:
            a_wrong_b_correct += 1

    n_discordant = a_correct_b_wrong + a_wrong_b_correct

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                k=min(a_correct_b_wrong, a_wrong_b_correct),
                n=n_discordant,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }

def bootstrap_delta_rho(pred_a: list[dict], pred_b: list[dict], ref_items: list[dict], n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(ref_items)

    gold = np.array([float(ref["p(Hallucination)"]) for ref in ref_items], dtype=float)
    a_scores = np.array([float(pred["p(Hallucination)"]) for pred in pred_a], dtype=float)
    b_scores = np.array([float(pred["p(Hallucination)"]) for pred in pred_b], dtype=float)

    observed_a = float(spearmanr(a_scores, gold)[0])
    observed_b = float(spearmanr(b_scores, gold)[0])
    observed_delta = observed_a - observed_b

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho_a = float(spearmanr(a_scores[idx], gold[idx])[0])
        rho_b = float(spearmanr(b_scores[idx], gold[idx])[0])
        if np.isnan(rho_a) or np.isnan(rho_b):
            continue
        deltas.append(rho_a - rho_b)

    if not deltas:
        return {
            "observed_delta_rho": observed_delta,
            "ci_lower": None,
            "ci_upper": None,
            "p_value_approx": None,
            "n_boot_effective": 0,
        }

    deltas = np.array(deltas, dtype=float)
    ci_lower = float(np.percentile(deltas, 2.5))
    ci_upper = float(np.percentile(deltas, 97.5))
    p_left = float(np.mean(deltas <= 0.0))
    p_right = float(np.mean(deltas >= 0.0))
    p_value_approx = float(min(1.0, 2 * min(p_left, p_right)))

    return {
        "observed_delta_rho": float(observed_delta),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_approx": p_value_approx,
        "n_boot_effective": int(len(deltas)),
    }

def compute_family_significance(rows: list[dict]) -> list[dict]:
    """
    Compare each model against the previous model in the same family, but only when
    input_path and prompt_version match.
    """
    family_groups: dict[str, list[dict]] = {}
    for row in rows:
        family_groups.setdefault(row["family"], []).append(row)

    results = []

    for family, family_rows in family_groups.items():
        family_rows = sorted(
            family_rows,
            key=lambda x: x["parameter_count"] if x["parameter_count"] is not None else float("inf"),
        )

        previous = None
        for row in family_rows:
            comparable = (
                previous is not None
                and row.get("input_path") == previous.get("input_path")
                and row.get("prompt_version") == previous.get("prompt_version")
                and row.get("archive_prediction_path")
                and previous.get("archive_prediction_path")
                and Path(row["archive_prediction_path"]).exists()
                and Path(previous["archive_prediction_path"]).exists()
                and row.get("input_path")
                and Path(row["input_path"]).exists()
            )

            if comparable:
                ref_items = load_json(Path(row["input_path"]))
                pred_curr = load_json(Path(row["archive_prediction_path"]))
                pred_prev = load_json(Path(previous["archive_prediction_path"]))

                acc_test = mcnemar_exact_test(pred_curr, pred_prev, ref_items)
                rho_test = bootstrap_delta_rho(pred_curr, pred_prev, ref_items)

                results.append(
                    {
                        "family": family,
                        "previous_model": previous["model_name"],
                        "current_model": row["model_name"],
                        "input_path": row.get("input_path"),
                        "prompt_version": row.get("prompt_version"),
                        "previous_acc": compute_accuracy_from_items(pred_prev, ref_items),
                        "current_acc": compute_accuracy_from_items(pred_curr, ref_items),
                        "delta_acc": compute_accuracy_from_items(pred_curr, ref_items) - compute_accuracy_from_items(pred_prev, ref_items),
                        "previous_rho": compute_rho_from_items(pred_prev, ref_items),
                        "current_rho": compute_rho_from_items(pred_curr, ref_items),
                        "delta_rho": compute_rho_from_items(pred_curr, ref_items) - compute_rho_from_items(pred_prev, ref_items),
                        "accuracy_significance": acc_test,
                        "rho_significance": rho_test,
                    }
                )

            previous = row

    return results

def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "family",
        "model_type",
        "model_name",
        "parameter_count",
        "parameter_count_readable",
        "agnostic_acc",
        "agnostic_rho",
        "mean_latency_seconds",
        "total_runtime_seconds",
        "acc_gain_vs_prev_family",
        "rho_gain_vs_prev_family",
        "latency_multiplier_vs_prev_family",
        "params_multiplier_vs_prev_family",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def write_markdown(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    family_groups: dict[str, list[dict]] = {}
    for row in rows:
        family_groups.setdefault(row["family"], []).append(row)

    lines = ["# Model Summary Tables", ""]

    for family, family_rows in family_groups.items():
        lines.append(f"## {family.capitalize()}")
        lines.append("")
        lines.append(
            "| Model | Params | Acc | Rho | Mean latency (s/example) | ΔAcc vs prev (family) | ΔRho vs prev (family) | Latency x prev (family) | Params x prev (family) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        for row in family_rows:
            lines.append(
                "| "
                f'{row["model_name"]} | '
                f'{row["parameter_count_readable"]} | '
                f'{format_float(row["agnostic_acc"])} | '
                f'{format_float(row["agnostic_rho"])} | '
                f'{format_float(row["mean_latency_seconds"])} | '
                f'{format_signed(row["acc_gain_vs_prev_family"])} | '
                f'{format_signed(row["rho_gain_vs_prev_family"])} | '
                f'{format_multiplier(row["latency_multiplier_vs_prev_family"])} | '
                f'{format_multiplier(row["params_multiplier_vs_prev_family"])} |'
            )

        lines.append("")

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_significance_outputs(results: list[dict], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "comparisons": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    family_groups: dict[str, list[dict]] = {}
    for row in results:
        family_groups.setdefault(row["family"], []).append(row)

    lines = ["# Family Significance Tests", ""]
    lines.append("Accuracy significance is based on McNemar's exact test.")
    lines.append("Rho significance is based on a paired bootstrap over examples.")
    lines.append("")

    for family, family_rows in family_groups.items():
        lines.append(f"## {family.capitalize()}")
        lines.append("")
        lines.append(
            "| Previous model | Current model | ΔAcc | Acc p-value | ΔRho | 95% CI for ΔRho | Rho p-value |"
        )
        lines.append("|---|---|---:|---:|---:|---|---:|")

        for row in family_rows:
            rho_sig = row["rho_significance"]
            ci_str = ""
            if rho_sig["ci_lower"] is not None and rho_sig["ci_upper"] is not None:
                ci_str = f'[{rho_sig["ci_lower"]:.3f}, {rho_sig["ci_upper"]:.3f}]'

            lines.append(
                "| "
                f'{row["previous_model"]} | '
                f'{row["current_model"]} | '
                f'{format_signed(row["delta_acc"])} | '
                f'{format_p(row["accuracy_significance"]["p_value"])} | '
                f'{format_signed(row["delta_rho"])} | '
                f'{ci_str} | '
                f'{format_p(rho_sig["p_value_approx"])} |'
            )

        lines.append("")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def print_significance_tables(results: list[dict]) -> None:
    if not results:
        print("\nNo within-family significance comparisons were produced.")
        return

    family_groups: dict[str, list[dict]] = {}
    for row in results:
        family_groups.setdefault(row["family"], []).append(row)

    for family, family_rows in family_groups.items():
        print(f"\n{family.upper()} Significance Table")
        print("-" * 160)
        print(
            f'{"Previous model":45} {"Current model":45} '
            f'{"ΔAcc":>8} {"Acc p":>8} {"ΔRho":>8} {"Rho p":>8} {"95% CI for ΔRho":>24}'
        )
        print("-" * 160)

        for row in family_rows:
            rho_sig = row["rho_significance"]

            if rho_sig["ci_lower"] is not None and rho_sig["ci_upper"] is not None:
                ci_str = f'[{rho_sig["ci_lower"]:.3f}, {rho_sig["ci_upper"]:.3f}]'
            else:
                ci_str = ""

            print(
                f'{row["previous_model"][:45]:45} '
                f'{row["current_model"][:45]:45} '
                f'{format_signed(row["delta_acc"]):>8} '
                f'{format_p(row["accuracy_significance"]["p_value"]):>8} '
                f'{format_signed(row["delta_rho"]):>8} '
                f'{format_p(rho_sig["p_value_approx"]):>8} '
                f'{ci_str:>24}'
            )

def print_family_tables(rows: list[dict]) -> None:
    family_groups: dict[str, list[dict]] = {}
    for row in rows:
        family_groups.setdefault(row["family"], []).append(row)

    for family, family_rows in family_groups.items():
        print(f"\n{family.upper()} Summary Table")
        print("-" * 160)
        print(
            f'{"Model":45} {"Params":>10} {"Acc":>8} {"Rho":>8} {"Latency":>10} '
            f'{"ΔAcc":>8} {"ΔRho":>8} {"Lat x":>8} {"Par x":>8}'
        )
        print("-" * 160)

        for row in family_rows:
            print(
                f'{row["model_name"]:45} '
                f'{row["parameter_count_readable"]:>10} '
                f'{format_float(row["agnostic_acc"]):>8} '
                f'{format_float(row["agnostic_rho"]):>8} '
                f'{format_float(row["mean_latency_seconds"]):>10} '
                f'{format_signed(row["acc_gain_vs_prev_family"]):>8} '
                f'{format_signed(row["rho_gain_vs_prev_family"]):>8} '
                f'{format_multiplier(row["latency_multiplier_vs_prev_family"]):>8} '
                f'{format_multiplier(row["params_multiplier_vs_prev_family"]):>8}'
            )

def main() -> None:
    if not METADATA_DIR.exists():
        raise FileNotFoundError(f"Metadata directory not found: {METADATA_DIR}")

    rows = load_metadata_files(METADATA_DIR)
    rows = sort_rows(rows)
    rows = add_family_comparison_columns(rows)

    if not rows:
        print("No metadata JSON files found.")
        return

    print_family_tables(rows)
    write_csv(rows, OUTPUT_CSV)
    write_markdown(rows, OUTPUT_MD)

    sig_results = compute_family_significance(rows)
    write_significance_outputs(sig_results, OUTPUT_STATS_JSON, OUTPUT_STATS_MD)
    print_significance_tables(sig_results)

    print(f"\nSaved CSV summary to: {OUTPUT_CSV}")
    print(f"Saved Markdown summary to: {OUTPUT_MD}")
    print(f"Saved significance JSON to: {OUTPUT_STATS_JSON}")
    print(f"Saved significance Markdown to: {OUTPUT_STATS_MD}")

if __name__ == "__main__":
    main()