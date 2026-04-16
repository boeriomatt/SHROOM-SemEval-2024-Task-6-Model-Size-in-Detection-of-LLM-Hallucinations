import json
from pathlib import Path
import matplotlib.pyplot as plt

METADATA_DIR = Path("outputs/metadata")
PLOTS_DIR = Path("outputs/plots")

def infer_family(model_type: str, model_name: str) -> str:
    model_name_lower = model_name.lower()

    if model_type == "flan" or "flan-t5" in model_name_lower:
        return "flan"

    if model_type == "deberta" or "deberta" in model_name_lower:
        return "deberta"

    return model_type if model_type else "unknown"

def load_metadata(metadata_dir: Path) -> list[dict]:
    rows = []

    for path in metadata_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        scores = data.get("scores", {})
        cost = data.get("computational_cost", {})

        row = {
            "family": infer_family(data.get("model_type", ""), data.get("model_name", "")),
            "model_name": data.get("model_name", ""),
            "parameter_count": cost.get("parameter_count"),
            "agnostic_acc": scores.get("agnostic_acc"),
            "agnostic_rho": scores.get("agnostic_rho"),
            "mean_latency_seconds": cost.get("mean_inference_latency_seconds_per_example"),
        }
        rows.append(row)

    return rows

def short_model_label(model_name: str) -> str:
    name = model_name.lower()

    if "flan-t5-" in name:
        return name.split("flan-t5-")[-1]

    if "nli-deberta-v3-" in name:
        return name.split("nli-deberta-v3-")[-1]

    return model_name

def make_family_plot(
    rows: list[dict],
    family: str,
    y_key: str,
    y_label: str,
    output_name: str,
) -> None:
    family_rows = [row for row in rows if row["family"] == family]
    family_rows.sort(key=lambda x: x["parameter_count"])

    if not family_rows:
        return

    x = [row["parameter_count"] for row in family_rows]
    y = [row[y_key] for row in family_rows]
    labels = [short_model_label(row["model_name"]) for row in family_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")

    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points")

    plt.xscale("log")
    plt.xlabel("Parameter count (log scale)")
    plt.ylabel(y_label)
    plt.title(f"{family.capitalize()} model size vs {y_label}")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()

def make_combined_tradeoff_plot(
    rows: list[dict],
    y_key: str,
    y_label: str,
    output_name: str,
) -> None:
    valid_rows = [
        row for row in rows
        if row["mean_latency_seconds"] is not None and row[y_key] is not None
    ]

    if not valid_rows:
        return

    flan_rows = sorted(
        [row for row in valid_rows if row["family"] == "flan"],
        key=lambda x: x["parameter_count"]
    )

    deberta_rows = sorted(
        [row for row in valid_rows if row["family"] == "deberta"],
        key=lambda x: x["parameter_count"]
    )

    plt.figure(figsize=(8, 5))

    if flan_rows:
        x = [row["mean_latency_seconds"] for row in flan_rows]
        y = [row[y_key] for row in flan_rows]
        plt.plot(x, y, marker="o", label="FLAN")

        for row in flan_rows:
            plt.annotate(
                short_model_label(row["model_name"]),
                (row["mean_latency_seconds"], row[y_key]),
                xytext=(5, 5),
                textcoords="offset points",
            )

    if deberta_rows:
        x = [row["mean_latency_seconds"] for row in deberta_rows]
        y = [row[y_key] for row in deberta_rows]
        plt.plot(x, y, marker="o", label="DeBERTa")

        for row in deberta_rows:
            plt.annotate(
                short_model_label(row["model_name"]),
                (row["mean_latency_seconds"], row[y_key]),
                xytext=(5, -10),
                textcoords="offset points",
            )

    plt.xlabel("Mean inference latency (seconds/example)")
    plt.ylabel(y_label)
    plt.title(f"Performance-cost trade-off: {y_label}")
    plt.legend()
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()

def main() -> None:
    if not METADATA_DIR.exists():
        raise FileNotFoundError(f"Metadata directory not found: {METADATA_DIR}")

    rows = load_metadata(METADATA_DIR)

    if not rows:
        print("No metadata files found.")
        return

    # FLAN-only plots
    make_family_plot(
        rows=rows,
        family="flan",
        y_key="agnostic_acc",
        y_label="Agnostic accuracy",
        output_name="flan_size_vs_accuracy.png",
    )
    make_family_plot(
        rows=rows,
        family="flan",
        y_key="agnostic_rho",
        y_label="Agnostic Spearman rho",
        output_name="flan_size_vs_rho.png",
    )
    make_family_plot(
        rows=rows,
        family="flan",
        y_key="mean_latency_seconds",
        y_label="Mean inference latency (seconds/example)",
        output_name="flan_size_vs_latency.png",
    )

    # DeBERTa-only plots
    make_family_plot(
        rows=rows,
        family="deberta",
        y_key="agnostic_acc",
        y_label="Agnostic accuracy",
        output_name="deberta_size_vs_accuracy.png",
    )
    make_family_plot(
        rows=rows,
        family="deberta",
        y_key="agnostic_rho",
        y_label="Agnostic Spearman rho",
        output_name="deberta_size_vs_rho.png",
    )
    make_family_plot(
        rows=rows,
        family="deberta",
        y_key="mean_latency_seconds",
        y_label="Mean inference latency (seconds/example)",
        output_name="deberta_size_vs_latency.png",
    )

    # Combined trade-off plots
    make_combined_tradeoff_plot(
        rows=rows,
        y_key="agnostic_acc",
        y_label="Agnostic accuracy",
        output_name="tradeoff_latency_vs_accuracy.png",
    )
    make_combined_tradeoff_plot(
        rows=rows,
        y_key="agnostic_rho",
        y_label="Agnostic Spearman rho",
        output_name="tradeoff_latency_vs_rho.png",
    )

    print(f"Saved plots to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()