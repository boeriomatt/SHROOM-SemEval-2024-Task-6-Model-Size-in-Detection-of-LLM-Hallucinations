import json
from pathlib import Path
import csv

METADATA_DIR = Path("outputs/metadata")
OUTPUT_CSV = Path("outputs/summary_table.csv")
OUTPUT_MD = Path("outputs/summary_table.md")

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

def load_metadata_files(metadata_dir: Path) -> list[dict]:
    rows = []

    for path in metadata_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        scores = data.get("scores", {})
        cost = data.get("computational_cost", {})

        row = {
            "model_name": data.get("model_name", ""),
            "parameter_count": cost.get("parameter_count"),
            "parameter_count_readable": format_params(cost.get("parameter_count")),
            "agnostic_acc": scores.get("agnostic_acc"),
            "agnostic_rho": scores.get("agnostic_rho"),
            "mean_latency_seconds": cost.get("mean_inference_latency_seconds_per_example"),
            "total_runtime_seconds": cost.get("total_inference_runtime_seconds"),
        }
        rows.append(row)

    return rows

def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda x: (
            x["parameter_count"] is None,
            x["parameter_count"] if x["parameter_count"] is not None else float("inf"),
        ),
    )

def add_comparison_columns(rows: list[dict]) -> list[dict]:
    previous = None

    for row in rows:
        if previous is None:
            row["acc_gain_vs_prev"] = None
            row["rho_gain_vs_prev"] = None
            row["latency_multiplier_vs_prev"] = None
            row["params_multiplier_vs_prev"] = None
        else:
            acc = row["agnostic_acc"]
            prev_acc = previous["agnostic_acc"]
            rho = row["agnostic_rho"]
            prev_rho = previous["agnostic_rho"]
            latency = row["mean_latency_seconds"]
            prev_latency = previous["mean_latency_seconds"]
            params = row["parameter_count"]
            prev_params = previous["parameter_count"]

            row["acc_gain_vs_prev"] = (
                acc - prev_acc
                if acc is not None and prev_acc is not None
                else None
            )
            row["rho_gain_vs_prev"] = (
                rho - prev_rho
                if rho is not None and prev_rho is not None
                else None
            )
            row["latency_multiplier_vs_prev"] = (
                latency / prev_latency
                if latency is not None and prev_latency not in (None, 0)
                else None
            )
            row["params_multiplier_vs_prev"] = (
                params / prev_params
                if params is not None and prev_params not in (None, 0)
                else None
            )

        previous = row

    return rows

def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "parameter_count",
        "parameter_count_readable",
        "agnostic_acc",
        "agnostic_rho",
        "mean_latency_seconds",
        "total_runtime_seconds",
        "acc_gain_vs_prev",
        "rho_gain_vs_prev",
        "latency_multiplier_vs_prev",
        "params_multiplier_vs_prev",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_markdown(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(
        "| Model | Params | Acc | Rho | Mean latency (s/example) | ΔAcc vs prev | ΔRho vs prev | Latency x prev | Params x prev |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    for row in rows:
        lines.append(
            "| "
            f'{row["model_name"]} | '
            f'{row["parameter_count_readable"]} | '
            f'{format_float(row["agnostic_acc"])} | '
            f'{format_float(row["agnostic_rho"])} | '
            f'{format_float(row["mean_latency_seconds"])} | '
            f'{format_signed(row["acc_gain_vs_prev"])} | '
            f'{format_signed(row["rho_gain_vs_prev"])} | '
            f'{format_multiplier(row["latency_multiplier_vs_prev"])} | '
            f'{format_multiplier(row["params_multiplier_vs_prev"])} |'
        )

    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def print_table(rows: list[dict]) -> None:
    print("\nSummary Table")
    print("-" * 150)
    print(
        f'{"Model":40} {"Params":>10} {"Acc":>8} {"Rho":>8} {"Latency":>10} '
        f'{"ΔAcc":>8} {"ΔRho":>8} {"Lat x":>8} {"Par x":>8}'
    )
    print("-" * 150)

    for row in rows:
        print(
            f'{row["model_name"]:40} '
            f'{row["parameter_count_readable"]:>10} '
            f'{format_float(row["agnostic_acc"]):>8} '
            f'{format_float(row["agnostic_rho"]):>8} '
            f'{format_float(row["mean_latency_seconds"]):>10} '
            f'{format_signed(row["acc_gain_vs_prev"]):>8} '
            f'{format_signed(row["rho_gain_vs_prev"]):>8} '
            f'{format_multiplier(row["latency_multiplier_vs_prev"]):>8} '
            f'{format_multiplier(row["params_multiplier_vs_prev"]):>8}'
        )

def main() -> None:
    if not METADATA_DIR.exists():
        raise FileNotFoundError(f"Metadata directory not found: {METADATA_DIR}")

    rows = load_metadata_files(METADATA_DIR)
    rows = sort_rows(rows)
    rows = add_comparison_columns(rows)

    if not rows:
        print("No metadata JSON files found.")
        return

    print_table(rows)
    write_csv(rows, OUTPUT_CSV)
    write_markdown(rows, OUTPUT_MD)

    print(f"\nSaved CSV summary to: {OUTPUT_CSV}")
    print(f"Saved Markdown summary to: {OUTPUT_MD}")

if __name__ == "__main__":
    main()