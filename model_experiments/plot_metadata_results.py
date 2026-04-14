import json
from pathlib import Path
import matplotlib.pyplot as plt

METADATA_DIR = Path("outputs/metadata")
PLOTS_DIR = Path("outputs/plots")

def load_metadata(metadata_dir: Path) -> list[dict]:
    rows = []

    for path in metadata_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        scores = data.get("scores", {})
        cost = data.get("computational_cost", {})

        row = {
            "model_name": data.get("model_name", ""),
            "parameter_count": cost.get("parameter_count"),
            "agnostic_acc": scores.get("agnostic_acc"),
            "agnostic_rho": scores.get("agnostic_rho"),
            "mean_latency_seconds": cost.get("mean_inference_latency_seconds_per_example"),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["parameter_count"])
    return rows

def short_model_label(model_name: str) -> str:
    return model_name.replace("google/", "").replace("flan-t5-", "")

def make_plot(
    rows: list[dict],
    y_key: str,
    y_label: str,
    output_name: str,
) -> None:
    x = [row["parameter_count"] for row in rows]
    y = [row[y_key] for row in rows]
    labels = [short_model_label(row["model_name"]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")

    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points")

    plt.xscale("log")
    plt.xlabel("Parameter count (log scale)")
    plt.ylabel(y_label)
    plt.title(f"FLAN-T5 model size vs {y_label}")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()

def make_tradeoff_plot(rows: list[dict]) -> None:
    x = [row["mean_latency_seconds"] for row in rows]
    y = [row["agnostic_acc"] for row in rows]
    labels = [short_model_label(row["model_name"]) for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")

    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points")

    plt.xlabel("Mean inference latency (seconds/example)")
    plt.ylabel("Agnostic accuracy")
    plt.title("Performance-cost trade-off across FLAN-T5 sizes")
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / "latency_vs_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()

def main() -> None:
    if not METADATA_DIR.exists():
        raise FileNotFoundError(f"Metadata directory not found: {METADATA_DIR}")

    rows = load_metadata(METADATA_DIR)

    if not rows:
        print("No metadata files found.")
        return

    make_plot(
        rows=rows,
        y_key="agnostic_acc",
        y_label="Agnostic accuracy",
        output_name="size_vs_accuracy.png",
    )

    make_plot(
        rows=rows,
        y_key="agnostic_rho",
        y_label="Agnostic Spearman rho",
        output_name="size_vs_rho.png",
    )

    make_plot(
        rows=rows,
        y_key="mean_latency_seconds",
        y_label="Mean inference latency (seconds/example)",
        output_name="size_vs_latency.png",
    )

    make_tradeoff_plot(rows)

    print(f"Saved plots to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()