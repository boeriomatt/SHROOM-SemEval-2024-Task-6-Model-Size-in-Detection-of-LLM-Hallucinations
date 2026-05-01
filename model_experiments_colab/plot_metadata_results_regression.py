import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METADATA_DIR = Path("outputs/metadata")
PLOTS_DIR = Path("outputs/plots")

# Plot styling defaults. These keep the figures readable in a thesis/report while
# avoiding external style dependencies.
COMBINED_FIGSIZE = (9, 6)
FAMILY_FIGSIZE = (8, 5)
DPI = 300
POINT_SIZE = 42
LINE_WIDTH = 1.8
ANNOTATION_FONT_SIZE = 8
CAPTION_FONT_SIZE = 8


def infer_family(model_type: str, model_name: str) -> str:
    model_type = (model_type or "").lower()
    model_name_lower = (model_name or "").lower()

    if model_type == "flan" or "flan-t5" in model_name_lower:
        return "flan"

    if model_type == "deberta" or "deberta" in model_name_lower:
        return "deberta"

    if model_type == "gemma" or "gemma" in model_name_lower:
        return "gemma"

    if model_type.startswith("qwen") or "qwen" in model_name_lower:
        return "qwen"

    return model_type if model_type else "unknown"


def infer_comparison_bucket(model_type: str, model_name: str) -> str:
    model_type = (model_type or "").lower()
    model_name_lower = (model_name or "").lower()

    if model_type == "flan" or "flan-t5" in model_name_lower:
        return "flan"

    if model_type == "gemma" or "gemma" in model_name_lower:
        return "gemma"

    if model_type.startswith("qwen") or "qwen" in model_name_lower:
        return "qwen"

    if model_type == "deberta" or "deberta" in model_name_lower:
        if model_name_lower.startswith("cross-encoder/nli-deberta-v3-"):
            return "deberta_cross_encoder"
        return "deberta_other_nli"

    return model_type if model_type else "unknown"


def bucket_title(bucket: str) -> str:
    mapping = {
        "flan": "FLAN-T5",
        "qwen": "Qwen2.5",
        "gemma": "Gemma 3",
        "deberta_cross_encoder": "DeBERTa cross-encoder",
        "deberta_other_nli": "DeBERTa NLI variants",
    }
    return mapping.get(bucket, bucket.replace("_", " ").title())


def load_metadata(metadata_dir: Path) -> list[dict[str, Any]]:
    rows = []

    for path in metadata_dir.glob("*.json"):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        scores = data.get("scores", {})
        cost = data.get("computational_cost", {})

        model_type = data.get("model_type", "")
        model_name = data.get("model_name", "")

        row = {
            "family": infer_family(model_type, model_name),
            "comparison_bucket": infer_comparison_bucket(model_type, model_name),
            "model_name": model_name,
            "parameter_count": cost.get("parameter_count"),
            "agnostic_acc": scores.get("agnostic_acc", scores.get("acc_agnostic")),
            "agnostic_rho": scores.get("agnostic_rho", scores.get("rho_agnostic")),
            "mean_latency_seconds": cost.get("mean_inference_latency_seconds_per_example"),
        }
        rows.append(row)

    return rows


def short_model_label(model_name: str) -> str:
    name = model_name.lower()

    if "flan-t5-" in name:
        return name.split("flan-t5-")[-1]

    if "cross-encoder/nli-deberta-v3-" in name:
        return name.split("cross-encoder/nli-deberta-v3-")[-1]

    if "deberta-v3-base-tasksource-nli" in name:
        return "tasksource-base"

    if "moritzlaurer/deberta-v3-large-mnli-fever-anli" in name:
        return "ml-fever-anli-large"

    if "qwen2.5-" in name and "-instruct" in name:
        return name.split("qwen2.5-")[-1].replace("-instruct", "")

    if "gemma-3-" in name and "-it" in name:
        return name.split("gemma-3-")[-1].replace("-it", "")

    return model_name


def _as_float_arrays(
    x_values: list[float],
    y_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def _transform_x(x: np.ndarray, x_transform: str) -> np.ndarray | None:
    if x_transform == "log10":
        if np.any(x <= 0):
            return None
        return np.log10(x)
    if x_transform == "identity":
        return x
    raise ValueError(f"Unsupported x_transform: {x_transform}")


def _fit_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_transform: str,
) -> dict[str, Any] | None:
    """Fit y ~ x, optionally after transforming x, and return plotting data."""
    if len(x) < 2:
        return None

    x_fit = _transform_x(x, x_transform)
    if x_fit is None:
        return None

    x_line_fit = np.linspace(float(x_fit.min()), float(x_fit.max()), 100)
    if x_transform == "log10":
        x_line = 10 ** x_line_fit
    else:
        x_line = x_line_fit

    slope, intercept = np.polyfit(x_fit, y, deg=1)
    y_hat = slope * x_fit + intercept
    y_line = slope * x_line_fit + intercept

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "x_line": x_line,
        "y_line": y_line,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "n": int(len(x)),
    }


def _format_r2(r_squared: float, n: int) -> str:
    if not np.isfinite(r_squared):
        return f"n={n}"
    return f"R²={r_squared:.2f}, n={n}"


def add_scatter_and_regression(
    x_values: list[float],
    y_values: list[float],
    *,
    label: str | None = None,
    x_transform: str = "identity",
    show_r2_in_legend: bool = True,
) -> dict[str, Any] | None:
    """
    Plot observed model points plus a simple least-squares linear regression line.

    For parameter-count plots, use x_transform="log10" so the fitted line is
    linear in log10(parameter_count), matching the log-scaled x-axis.
    """
    x, y = _as_float_arrays(x_values, y_values)
    if len(x) == 0:
        return None

    fit = _fit_linear_regression(x, y, x_transform=x_transform)

    legend_label = label
    if label is not None and fit is not None and show_r2_in_legend:
        legend_label = f"{label} ({_format_r2(fit['r_squared'], fit['n'])})"

    scatter = plt.scatter(x, y, s=POINT_SIZE, label=legend_label, zorder=3)

    if fit is None:
        return None

    facecolors = scatter.get_facecolors()
    color = facecolors[0] if len(facecolors) else None
    plt.plot(
        fit["x_line"],
        fit["y_line"],
        linestyle="--",
        linewidth=LINE_WIDTH,
        color=color,
        alpha=0.95,
        zorder=2,
    )
    return fit


# Manual label offsets for the dense combined plots. These are deliberately small
# adjustments to reduce overlap while preserving the point locations.
COMBINED_LABEL_OFFSETS: dict[tuple[str, str], tuple[int, int]] = {
    ("flan", "small"): (6, 7),
    ("flan", "base"): (6, 7),
    ("flan", "large"): (6, 7),
    ("flan", "xl"): (6, 7),
    ("deberta_cross_encoder", "xsmall"): (6, -12),
    ("deberta_cross_encoder", "small"): (6, -15),
    ("deberta_cross_encoder", "base"): (6, 6),
    ("deberta_cross_encoder", "large"): (6, -13),
    ("deberta_other_nli", "tasksource-base"): (6, 12),
    ("deberta_other_nli", "ml-fever-anli-large"): (6, 12),
    ("qwen", "0.5b"): (6, -13),
    ("qwen", "1.5b"): (6, -13),
    ("qwen", "3b"): (6, -13),
    ("qwen", "7b"): (6, -13),
    ("gemma", "270m"): (6, 6),
    ("gemma", "1b"): (6, 6),
    ("gemma", "4b"): (6, 6),
}

FAMILY_LABEL_OFFSETS: dict[str, tuple[int, int]] = {
    "xsmall": (6, -12),
    "small": (6, 7),
    "base": (6, 7),
    "large": (6, 7),
    "xl": (6, 7),
    "tasksource-base": (6, 10),
    "ml-fever-anli-large": (6, 10),
    "0.5b": (6, -12),
    "1.5b": (6, -12),
    "3b": (6, -12),
    "7b": (6, -12),
    "270m": (6, 6),
    "1b": (6, 6),
    "4b": (6, 6),
}


def annotate_point(
    x: float,
    y: float,
    text: str,
    *,
    offset: tuple[int, int] = (6, 6),
) -> None:
    plt.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=ANNOTATION_FONT_SIZE,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.65},
        clip_on=False,
        zorder=4,
    )


def add_caption(caption: str) -> None:
    plt.gcf().text(
        0.5,
        0.012,
        caption,
        ha="center",
        va="bottom",
        fontsize=CAPTION_FONT_SIZE,
    )


def metric_title(y_label: str) -> str:
    if y_label.lower() == "accuracy":
        return "Accuracy"
    if "rho" in y_label.lower():
        return "Spearman rho"
    if "latency" in y_label.lower():
        return "Mean inference latency"
    return y_label


def make_family_plot(
    rows: list[dict[str, Any]],
    bucket: str,
    y_key: str,
    y_label: str,
    output_name: str,
) -> None:
    bucket_rows = [row for row in rows if row["comparison_bucket"] == bucket]
    bucket_rows = [
        row for row in bucket_rows
        if row["parameter_count"] is not None and row[y_key] is not None
    ]
    bucket_rows.sort(key=lambda x: x["parameter_count"])

    if not bucket_rows:
        return

    x = [row["parameter_count"] for row in bucket_rows]
    y = [row[y_key] for row in bucket_rows]
    labels = [short_model_label(row["model_name"]) for row in bucket_rows]

    plt.figure(figsize=FAMILY_FIGSIZE)
    fit = add_scatter_and_regression(
        x,
        y,
        label=bucket_title(bucket),
        x_transform="log10",
        show_r2_in_legend=True,
    )

    for xi, yi, point_label in zip(x, y, labels):
        annotate_point(
            xi,
            yi,
            point_label,
            offset=FAMILY_LABEL_OFFSETS.get(point_label, (6, 6)),
        )

    plt.xscale("log")
    plt.xlabel("Parameter count (log scale)")
    plt.ylabel(y_label)
    plt.title(f"{bucket_title(bucket)}: model size (log scale) vs {metric_title(y_label)}")

    if fit is not None:
        plt.legend(loc="best", fontsize=8, frameon=True)

    add_caption(
        "Dashed line shows least-squares trend fitted over log10(parameter count); "
        "R² is descriptive because each family has few checkpoints."
    )
    plt.tight_layout(rect=(0, 0.055, 1, 1))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=DPI, bbox_inches="tight")
    plt.close()


def make_combined_tradeoff_plot(
    rows: list[dict[str, Any]],
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

    bucket_specs = [
        ("flan", "FLAN", (6, 6)),
        ("deberta_cross_encoder", "DeBERTa (cross-encoder)", (6, -12)),
        ("deberta_other_nli", "DeBERTa (other NLI)", (6, 10)),
        ("qwen", "Qwen", (6, -12)),
        ("gemma", "Gemma", (6, 6)),
    ]

    plt.figure(figsize=COMBINED_FIGSIZE)

    for bucket, legend_label, default_offset in bucket_specs:
        bucket_rows = sorted(
            [row for row in valid_rows if row["comparison_bucket"] == bucket],
            key=lambda x: x["parameter_count"] if x["parameter_count"] is not None else float("inf"),
        )

        if not bucket_rows:
            continue

        x = [row["mean_latency_seconds"] for row in bucket_rows]
        y = [row[y_key] for row in bucket_rows]
        add_scatter_and_regression(x, y, label=legend_label, x_transform="identity")

        for row in bucket_rows:
            point_label = short_model_label(row["model_name"])
            annotate_point(
                row["mean_latency_seconds"],
                row[y_key],
                point_label,
                offset=COMBINED_LABEL_OFFSETS.get((bucket, point_label), default_offset),
            )

    plt.xlabel("Mean inference latency (seconds/example)")
    plt.ylabel(y_label)
    plt.title(f"Latency-performance trade-off: {metric_title(y_label)}")
    plt.legend(loc="best", fontsize=8, frameon=True)
    add_caption(
        "Dashed lines show least-squares trend fitted within each family over mean inference latency; "
        "R² is descriptive."
    )
    plt.tight_layout(rect=(0, 0.055, 1, 1))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=DPI, bbox_inches="tight")
    plt.close()


def make_combined_size_plot(
    rows: list[dict[str, Any]],
    y_key: str,
    y_label: str,
    output_name: str,
) -> None:
    valid_rows = [
        row for row in rows
        if row["parameter_count"] is not None and row[y_key] is not None
    ]

    if not valid_rows:
        return

    bucket_specs = [
        ("flan", "FLAN", (6, 6)),
        ("deberta_cross_encoder", "DeBERTa (cross-encoder)", (6, -12)),
        ("deberta_other_nli", "DeBERTa (other NLI)", (6, 10)),
        ("qwen", "Qwen", (6, -12)),
        ("gemma", "Gemma", (6, 6)),
    ]

    plt.figure(figsize=COMBINED_FIGSIZE)

    for bucket, legend_label, default_offset in bucket_specs:
        bucket_rows = sorted(
            [row for row in valid_rows if row["comparison_bucket"] == bucket],
            key=lambda x: x["parameter_count"] if x["parameter_count"] is not None else float("inf"),
        )

        if not bucket_rows:
            continue

        x = [row["parameter_count"] for row in bucket_rows]
        y = [row[y_key] for row in bucket_rows]
        add_scatter_and_regression(x, y, label=legend_label, x_transform="log10")

        for row in bucket_rows:
            point_label = short_model_label(row["model_name"])
            annotate_point(
                row["parameter_count"],
                row[y_key],
                point_label,
                offset=COMBINED_LABEL_OFFSETS.get((bucket, point_label), default_offset),
            )

    plt.xscale("log")
    plt.xlabel("Parameter count (log scale)")
    plt.ylabel(y_label)
    plt.title(f"Model size (log scale) vs {metric_title(y_label)}")
    plt.legend(loc="best", fontsize=8, frameon=True)
    add_caption(
        "Dashed lines show least-squares trend fitted within each family over log10(parameter count); "
        "R² is descriptive."
    )
    plt.tight_layout(rect=(0, 0.055, 1, 1))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=DPI, bbox_inches="tight")
    plt.close()


def write_plot_captions() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    caption_path = PLOTS_DIR / "plot_captions.md"
    caption_path.write_text(
        "# Suggested figure captions\n\n"
        "- **Model size plots:** Parameter count is shown on a logarithmic x-axis. "
        "Dashed lines show family-wise least-squares regressions fitted over "
        "log10(parameter count). Reported R² values are descriptive because each "
        "family contains only a small number of checkpoints.\n\n"
        "- **Latency trade-off plots:** Mean inference latency is shown in seconds per "
        "example. Dashed lines show family-wise least-squares regressions fitted over "
        "latency. These lines summarize observed trade-offs and should not be "
        "interpreted as causal estimates.\n",
        encoding="utf-8",
    )


def main() -> None:
    if not METADATA_DIR.exists():
        raise FileNotFoundError(f"Metadata directory not found: {METADATA_DIR}")

    rows = load_metadata(METADATA_DIR)

    if not rows:
        print("No metadata files found.")
        return

    # FLAN
    make_family_plot(
        rows=rows,
        bucket="flan",
        y_key="agnostic_acc",
        y_label="Accuracy",
        output_name="flan_size_vs_accuracy.png",
    )
    make_family_plot(
        rows=rows,
        bucket="flan",
        y_key="agnostic_rho",
        y_label="Spearman rho",
        output_name="flan_size_vs_rho.png",
    )
    make_family_plot(
        rows=rows,
        bucket="flan",
        y_key="mean_latency_seconds",
        y_label="Mean inference latency (seconds/example)",
        output_name="flan_size_vs_latency.png",
    )

    # DeBERTa cross-encoder
    make_family_plot(
        rows=rows,
        bucket="deberta_cross_encoder",
        y_key="agnostic_acc",
        y_label="Accuracy",
        output_name="deberta_cross_encoder_size_vs_accuracy.png",
    )
    make_family_plot(
        rows=rows,
        bucket="deberta_cross_encoder",
        y_key="agnostic_rho",
        y_label="Spearman rho",
        output_name="deberta_cross_encoder_size_vs_rho.png",
    )
    make_family_plot(
        rows=rows,
        bucket="deberta_cross_encoder",
        y_key="mean_latency_seconds",
        y_label="Mean inference latency (seconds/example)",
        output_name="deberta_cross_encoder_size_vs_latency.png",
    )

    # DeBERTa other NLI variants
    make_family_plot(
        rows=rows,
        bucket="deberta_other_nli",
        y_key="agnostic_acc",
        y_label="Accuracy",
        output_name="deberta_other_nli_size_vs_accuracy.png",
    )
    make_family_plot(
        rows=rows,
        bucket="deberta_other_nli",
        y_key="agnostic_rho",
        y_label="Spearman rho",
        output_name="deberta_other_nli_size_vs_rho.png",
    )
    make_family_plot(
        rows=rows,
        bucket="deberta_other_nli",
        y_key="mean_latency_seconds",
        y_label="Mean inference latency (seconds/example)",
        output_name="deberta_other_nli_size_vs_latency.png",
    )

    # Qwen
    make_family_plot(
        rows=rows,
        bucket="qwen",
        y_key="agnostic_acc",
        y_label="Accuracy",
        output_name="qwen_size_vs_accuracy.png",
    )
    make_family_plot(
        rows=rows,
        bucket="qwen",
        y_key="agnostic_rho",
        y_label="Spearman rho",
        output_name="qwen_size_vs_rho.png",
    )
    make_family_plot(
        rows=rows,
        bucket="qwen",
        y_key="mean_latency_seconds",
        y_label="Mean inference latency (seconds/example)",
        output_name="qwen_size_vs_latency.png",
    )

    # Gemma
    make_family_plot(
        rows=rows,
        bucket="gemma",
        y_key="agnostic_acc",
        y_label="Accuracy",
        output_name="gemma_size_vs_accuracy.png",
    )
    make_family_plot(
        rows=rows,
        bucket="gemma",
        y_key="agnostic_rho",
        y_label="Spearman rho",
        output_name="gemma_size_vs_rho.png",
    )
    make_family_plot(
        rows=rows,
        bucket="gemma",
        y_key="mean_latency_seconds",
        y_label="Mean inference latency (seconds/example)",
        output_name="gemma_size_vs_latency.png",
    )

    # Combined trade-off plots
    make_combined_tradeoff_plot(
        rows=rows,
        y_key="agnostic_acc",
        y_label="Accuracy",
        output_name="tradeoff_latency_vs_accuracy.png",
    )
    make_combined_tradeoff_plot(
        rows=rows,
        y_key="agnostic_rho",
        y_label="Spearman rho",
        output_name="tradeoff_latency_vs_rho.png",
    )

    # Combined size plots
    make_combined_size_plot(
        rows=rows,
        y_key="agnostic_acc",
        y_label="Accuracy",
        output_name="combined_size_vs_accuracy.png",
    )
    make_combined_size_plot(
        rows=rows,
        y_key="agnostic_rho",
        y_label="Spearman rho",
        output_name="combined_size_vs_rho.png",
    )

    write_plot_captions()
    print(f"Saved plots to: {PLOTS_DIR}")
    print(f"Saved suggested captions to: {PLOTS_DIR / 'plot_captions.md'}")


if __name__ == "__main__":
    main()
