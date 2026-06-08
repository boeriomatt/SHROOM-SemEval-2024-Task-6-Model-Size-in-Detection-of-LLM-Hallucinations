import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

METADATA_DIR = Path("outputs/metadata")
PLOTS_DIR = Path("outputs/plots")

# Plot styling defaults. These keep the figures readable in a thesis/report while avoiding external style dependencies.
COMBINED_FIGSIZE = (9, 6)
FAMILY_FIGSIZE = (8, 5)
DPI = 300
POINT_SIZE = 42
LINE_WIDTH = 1.8
ANNOTATION_FONT_SIZE = 8
CAPTION_FONT_SIZE = 8

# Set to False if you want combined plots to show only OOTB families when both OOTB and fine-tuned metadata are present in outputs/metadata.
INCLUDE_FINETUNED_IN_COMBINED = True

# Buckets that should be shown as grouped scatter points, but without a fitted regression line.
NO_REGRESSION_BUCKETS = {
    "deberta_other_nli",
    "deberta_lora_finetuned_other_nli",
}

BUCKET_COLORS: dict[str, str] = {
    "flan": "#1f77b4",
    "flan_lora_finetuned": "#1f77b4",

    "deberta_cross_encoder": "#ff7f0e",
    "deberta_lora_finetuned_cross_encoder": "#ff7f0e",

    "deberta_other_nli": "#9467bd",
    "deberta_lora_finetuned_other_nli": "#9467bd",

    "qwen": "#2ca02c",
    "qwen_lora_finetuned": "#2ca02c",

    "gemma": "#d62728",
    "gemma_lora_finetuned": "#d62728",
}

FALLBACK_BUCKET_COLOR = "#7f7f7f"

def color_for_bucket(bucket: str) -> str:
    """Return the hard-coded plotting color for a comparison bucket."""
    return BUCKET_COLORS.get(bucket, FALLBACK_BUCKET_COLOR)

# Manual defaults for common rung labels.
FAMILY_LABEL_OFFSETS: dict[str, tuple[int, int]] = {
    "xsmall": (8, 8),
    "small": (8, -12),
    "base": (8, 8),
    "large": (8, 8),
    "xl": (8, 8),
    "xxl": (8, 8),
    "270m": (8, -8),
    "0.5b": (8, -8),
    "1b": (8, 8),
    "1.5b": (8, -10),
    "3b": (8, 8),
    "4b": (8, 8),
    "7b": (8, -8),
    "tasksource-base": (8, 12),
    "ml-fever-anli-large": (8, 12),
}

# Optional metric-specific label offsets for the combined size plots.
COMBINED_SIZE_LABEL_OFFSETS: dict[tuple[str, str, str], tuple[float, float]] = {
    # FLAN / FLAN LoRA: keep the small point below, and separate mid-sized points from nearby DeBERTa/Qwen labels.
    ("flan", "accuracy", "small"): (8, -14),
    ("flan", "accuracy", "base"): (8, 10),
    ("flan", "accuracy", "large"): (8, 12),
    ("flan", "accuracy", "xl"): (-8, 10),
    ("flan", "rho", "small"): (8, -14),
    ("flan", "rho", "base"): (8, 10),
    ("flan", "rho", "large"): (8, 12),
    ("flan", "rho", "xl"): (-8, 10),
    ("flan_lora_finetuned", "accuracy", "small"): (8, -14),
    ("flan_lora_finetuned", "accuracy", "base"): (8, 10),
    ("flan_lora_finetuned", "accuracy", "large"): (8, 12),
    ("flan_lora_finetuned", "accuracy", "xl"): (-8, 10),
    ("flan_lora_finetuned", "rho", "small"): (8, -14),
    ("flan_lora_finetuned", "rho", "base"): (8, 10),
    ("flan_lora_finetuned", "rho", "large"): (8, 12),
    ("flan_lora_finetuned", "rho", "xl"): (-8, 10),

    # DeBERTa cross-encoder OOTB / LoRA: stagger the compact xsmall-small-base area.
    ("deberta_cross_encoder", "accuracy", "xsmall"): (8, 10),
    ("deberta_cross_encoder", "accuracy", "small"): (8, -14),
    ("deberta_cross_encoder", "accuracy", "base"): (-8, 10),
    ("deberta_cross_encoder", "accuracy", "large"): (8, 8),
    ("deberta_cross_encoder", "rho", "xsmall"): (8, 10),
    ("deberta_cross_encoder", "rho", "small"): (8, -14),
    ("deberta_cross_encoder", "rho", "base"): (-8, 10),
    ("deberta_cross_encoder", "rho", "large"): (8, 8),
    ("deberta_lora_finetuned_cross_encoder", "accuracy", "xsmall"): (8, 10),
    ("deberta_lora_finetuned_cross_encoder", "accuracy", "small"): (8, -14),
    ("deberta_lora_finetuned_cross_encoder", "accuracy", "base"): (8, 12),
    ("deberta_lora_finetuned_cross_encoder", "accuracy", "large"): (8, 12),
    ("deberta_lora_finetuned_cross_encoder", "rho", "xsmall"): (8, 10),
    ("deberta_lora_finetuned_cross_encoder", "rho", "small"): (8, -16),
    ("deberta_lora_finetuned_cross_encoder", "rho", "base"): (8, 12),
    ("deberta_lora_finetuned_cross_encoder", "rho", "large"): (8, 12),

    # Other NLI DeBERTa models: place both labels above the points.
    ("deberta_other_nli", "accuracy", "tasksource-base"): (8, 14),
    ("deberta_other_nli", "accuracy", "ml-fever-anli-large"): (8, 10),
    ("deberta_other_nli", "rho", "tasksource-base"): (8, 14),
    ("deberta_other_nli", "rho", "ml-fever-anli-large"): (8, 10),
    ("deberta_lora_finetuned_other_nli", "accuracy", "tasksource-base"): (8, 14),
    ("deberta_lora_finetuned_other_nli", "accuracy", "ml-fever-anli-large"): (8, 10),
    ("deberta_lora_finetuned_other_nli", "rho", "tasksource-base"): (8, 14),
    ("deberta_lora_finetuned_other_nli", "rho", "ml-fever-anli-large"): (8, 10),

    # Qwen OOTB / LoRA: separate 1.5b from nearby FLAN/DeBERTa and keep 7b inside.
    ("qwen", "accuracy", "0.5b"): (8, -12),
    ("qwen", "accuracy", "1.5b"): (8, -16),
    ("qwen", "accuracy", "3b"): (8, 12),
    ("qwen", "accuracy", "7b"): (-8, -12),
    ("qwen", "rho", "0.5b"): (8, -12),
    ("qwen", "rho", "1.5b"): (8, -16),
    ("qwen", "rho", "3b"): (8, 12),
    ("qwen", "rho", "7b"): (-8, -12),
    ("qwen_lora_finetuned", "accuracy", "0.5b"): (8, -12),
    ("qwen_lora_finetuned", "accuracy", "1.5b"): (8, -16),
    ("qwen_lora_finetuned", "accuracy", "3b"): (8, 12),
    ("qwen_lora_finetuned", "accuracy", "7b"): (-8, -12),
    ("qwen_lora_finetuned", "rho", "0.5b"): (8, -12),
    ("qwen_lora_finetuned", "rho", "1.5b"): (8, -16),
    ("qwen_lora_finetuned", "rho", "3b"): (8, 14),
    ("qwen_lora_finetuned", "rho", "7b"): (-8, -12),

    # Gemma / Gemma LoRA: keep the smallest rung below the point and avoid edge clipping.
    ("gemma", "accuracy", "270m"): (8, -16),
    ("gemma", "accuracy", "1b"): (8, 10),
    ("gemma", "accuracy", "4b"): (8, 10),
    ("gemma", "rho", "270m"): (8, -16),
    ("gemma", "rho", "1b"): (8, 10),
    ("gemma", "rho", "4b"): (8, 10),
    ("gemma_lora_finetuned", "accuracy", "270m"): (8, -16),
    ("gemma_lora_finetuned", "accuracy", "1b"): (8, 10),
    ("gemma_lora_finetuned", "accuracy", "4b"): (8, 10),
    ("gemma_lora_finetuned", "rho", "270m"): (8, -16),
    ("gemma_lora_finetuned", "rho", "1b"): (8, 10),
    ("gemma_lora_finetuned", "rho", "4b"): (8, 10),
}

def metric_key_for_label_offsets(y_key: str) -> str:
    """Map row metric keys to the compact names used in label-offset tables."""
    lowered = str(y_key).lower()
    if "acc" in lowered or lowered == "accuracy":
        return "accuracy"
    if "rho" in lowered:
        return "rho"
    return lowered

def combined_size_label_offset(
    bucket: str,
    point_label: str,
    y_key: str,
    default_offset: tuple[float, float],
) -> tuple[float, float]:
    metric_key = metric_key_for_label_offsets(y_key)
    return COMBINED_SIZE_LABEL_OFFSETS.get((bucket, metric_key, point_label), default_offset)

def _as_float(value: Any) -> float | None:
    # Return value as float if possible, preserving None for missing values.
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _is_finetuned_deberta(model_type: str) -> bool:
    model_type_lower = (model_type or "").lower()
    return model_type_lower in {
        "deberta_finetuned_binary_classifier",
        "finetuned_deberta_binary_classifier",
        "deberta_finetuned",
        "deberta_lora_finetuned_binary_classifier",
        "deberta_lora_finetuned",
        "deberta_lora",
    }

def _is_finetuned_flan_lora(model_type: str) -> bool:
    model_type_lower = (model_type or "").lower()
    return model_type_lower in {
        "flan_lora_finetuned_seq2seq_verbalizer",
        "finetuned_flan_lora",
        "flan_lora_finetuned",
        "flan_lora",
        "flan_lora_ft",
        "flan_finetuned_lora",
    }

def _is_finetuned_gemma_lora(model_type: str) -> bool:
    model_type_lower = (model_type or "").lower()
    return model_type_lower in {
        "gemma_lora_finetuned_causal_lm_verbalizer",
        "finetuned_gemma_lora",
        "gemma_lora_finetuned",
        "gemma_lora",
    }

def _is_finetuned_qwen_lora(model_type: str) -> bool:
    model_type_lower = (model_type or "").lower()
    return model_type_lower in {
        "qwen_lora_finetuned_causal_lm_verbalizer",
        "finetuned_qwen_lora",
        "qwen_lora_finetuned",
        "qwen_lora",
    }

def extract_model_type_and_name(data: dict[str, Any]) -> tuple[str, str]:
    """
    Normalize model identity across OOTB and fine-tuned metadata schemas.

    OOTB metadata usually stores:
        model_type, model_name

    Fine-tuned metadata usually stores:
        model_type = deberta_finetuned_binary_classifier
        base_model_name = original Hugging Face checkpoint
    """
    model_type = data.get("model_type", "") or ""
    model_name = data.get("model_name") or data.get("base_model_name") or ""
    return str(model_type), str(model_name)

def extract_scores(data: dict[str, Any]) -> tuple[float | None, float | None]:
    """
    Normalize score fields across metadata schemas.

    OOTB metadata commonly uses:
        scores.agnostic_acc / scores.agnostic_rho
        or scores.acc_agnostic / scores.rho_agnostic

    Final fine-tuned metadata commonly uses:
        participant_scores.agnostic_acc / participant_scores.agnostic_rho
        and/or final_prediction_metrics.accuracy / final_prediction_metrics.rho
    """
    scores = data.get("scores") or {}
    participant_scores = data.get("participant_scores") or {}
    final_metrics = data.get("final_prediction_metrics") or {}

    acc = (
        scores.get("agnostic_acc")
        if scores.get("agnostic_acc") is not None
        else scores.get("acc_agnostic")
    )
    rho = (
        scores.get("agnostic_rho")
        if scores.get("agnostic_rho") is not None
        else scores.get("rho_agnostic")
    )

    if acc is None:
        acc = (
            participant_scores.get("agnostic_acc")
            if participant_scores.get("agnostic_acc") is not None
            else participant_scores.get("acc_agnostic")
        )
    if rho is None:
        rho = (
            participant_scores.get("agnostic_rho")
            if participant_scores.get("agnostic_rho") is not None
            else participant_scores.get("rho_agnostic")
        )

    if acc is None:
        acc = final_metrics.get("accuracy") or final_metrics.get("acc") or final_metrics.get("test_acc")
    if rho is None:
        rho = final_metrics.get("rho") or final_metrics.get("spearman_rho") or final_metrics.get("test_rho")

    return _as_float(acc), _as_float(rho)

def normalize_metadata_row(data: dict[str, Any], path: Path) -> dict[str, Any]:
    # Convert OOTB or fine-tuned metadata JSON into one plotting row
    model_type, model_name = extract_model_type_and_name(data)
    acc, rho = extract_scores(data)
    cost = data.get("computational_cost", {}) or {}

    return {
        "family": infer_family(model_type, model_name),
        "comparison_bucket": infer_comparison_bucket(model_type, model_name),
        "model_type": model_type,
        "model_name": model_name,
        "metadata_file": str(path),
        "parameter_count": _as_float(cost.get("parameter_count")),
        "agnostic_acc": acc,
        "agnostic_rho": rho,
        "mean_latency_seconds": _as_float(cost.get("mean_inference_latency_seconds_per_example")),
        "total_training_runtime_seconds": _as_float(cost.get("total_training_runtime_seconds")),
    }

def infer_family(model_type: str, model_name: str) -> str:
    model_type = (model_type or "").lower()
    model_name_lower = (model_name or "").lower()

    if _is_finetuned_deberta(model_type):
        return "deberta_lora_finetuned"
    
    if _is_finetuned_flan_lora(model_type):
        return "flan_lora_finetuned"
    
    if _is_finetuned_gemma_lora(model_type):
        return "gemma_lora_finetuned"
    
    if _is_finetuned_qwen_lora(model_type):
        return "qwen_lora_finetuned"

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

    # Fine-tuned DeBERTa gets its own adaptation-regime bucket so it is not mixed with OOTB DeBERTa in combined plots.
    if _is_finetuned_deberta(model_type):
        if model_name_lower.startswith("cross-encoder/nli-deberta-v3-"):
            return "deberta_lora_finetuned_cross_encoder"
        return "deberta_lora_finetuned_other_nli"
    
    if _is_finetuned_flan_lora(model_type):
        return "flan_lora_finetuned"

    if _is_finetuned_gemma_lora(model_type):
        return "gemma_lora_finetuned"

    if _is_finetuned_qwen_lora(model_type):
        return "qwen_lora_finetuned"

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
        "flan": "FLAN OOTB",
        "qwen": "Qwen OOTB",
        "gemma": "Gemma OOTB",
        "deberta_cross_encoder": "DeBERTa cross-encoder OOTB",
        "deberta_other_nli": "DeBERTa NLI variants OOTB",
        "deberta_lora_finetuned_cross_encoder": "DeBERTa cross-encoder LoRA fine-tuned",
        "deberta_lora_finetuned_other_nli": "DeBERTa NLI variants LoRA fine-tuned",
        "flan_lora_finetuned": "FLAN LoRA fine-tuned",
        "gemma_lora_finetuned": "Gemma LoRA fine-tuned",
        "qwen_lora_finetuned": "Qwen LoRA fine-tuned",
    }
    return mapping.get(bucket, bucket.replace("_", " ").title())

def load_metadata(metadata_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path in sorted(metadata_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        row = normalize_metadata_row(data, path)
        rows.append(row)

    return rows

def short_model_label(model_name: str) -> str:
    name = (model_name or "").lower()

    if "flan-t5-" in name:
        return name.split("flan-t5-")[-1]
    
    if "flan-lora-finetuned" in name:
        return name.split("flan-lora-finetuned")[-1].lstrip("-").replace("-seq2seq-verbalizer", "")
    
    if "gemma-lora-finetuned" in name:
        return name.split("gemma-lora-finetuned")[-1].lstrip("-").replace("-causal_lm_verbalizer", "")
    
    if "qwen-lora-finetuned" in name:
        return name.split("qwen-lora-finetuned")[-1].lstrip("-").replace("-causal_lm_verbalizer", "")

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
    label: str,
    x_transform: str | None = None,
    show_r2_in_legend: bool = True,
    draw_regression: bool = True,
    color: Any | None = None,
) -> dict[str, Any] | None:
    # Plot a series and return metadata, including its matplotlib color.
    if len(x_values) == 0:
        return None

    x = np.array(x_values, dtype=float)
    y = np.array(y_values, dtype=float)
    transform_name = x_transform or "identity"

    fit = (
        _fit_linear_regression(x, y, x_transform=transform_name)
        if draw_regression and len(x) >= 2
        else None
    )
    legend_label = label
    if fit and show_r2_in_legend:
        legend_label = f"{label} ({_format_r2(fit['r_squared'], fit['n'])})"
    elif show_r2_in_legend:
        legend_label = f"{label} (n={len(x)})"

    scatter = plt.scatter(x, y, label=legend_label, s=POINT_SIZE, color=color)
    facecolors = scatter.get_facecolors()
    series_color = color if color is not None else (facecolors[0] if len(facecolors) else None)

    if draw_regression and fit:
        plt.plot(fit["x_line"], fit["y_line"], linestyle="--", linewidth=LINE_WIDTH, color=series_color)

    return {
        "fit": fit,
        "color": series_color,
        "label": label,
        "legend_label": legend_label,
        "n": len(x),
    }

def annotate_point(
    x: float,
    y: float,
    label: str,
    offset: tuple[int, int],
    color: Any | None = None,
):
    text_color = color if color is not None else "black"
    return plt.annotate(
        label,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=ANNOTATION_FONT_SIZE,
        color=text_color,
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.76},
        zorder=5,
    )

def _set_annotation_alignment_from_offset(annotation) -> None:
    """Keep label alignment consistent when an offset is moved by collision handling."""
    x_offset, y_offset = annotation.get_position()
    if x_offset > 1:
        annotation.set_ha("left")
    elif x_offset < -1:
        annotation.set_ha("right")
    else:
        annotation.set_ha("center")

    if y_offset > 1:
        annotation.set_va("bottom")
    elif y_offset < -1:
        annotation.set_va("top")
    else:
        annotation.set_va("center")

def _move_annotation_by_pixels(
    annotation,
    dx_pixels: float,
    dy_pixels: float,
    max_abs_x_offset_points: float,
    max_abs_y_offset_points: float,
) -> None:
    # Move a point-label annotation by display pixels while preserving point offsets.
    figure = annotation.figure
    points_per_pixel = 72.0 / figure.dpi
    x_offset, y_offset = annotation.get_position()
    new_x = max(
        -max_abs_x_offset_points,
        min(max_abs_x_offset_points, x_offset + dx_pixels * points_per_pixel),
    )
    new_y = max(
        -max_abs_y_offset_points,
        min(max_abs_y_offset_points, y_offset + dy_pixels * points_per_pixel),
    )
    annotation.set_position((new_x, new_y))
    _set_annotation_alignment_from_offset(annotation)

def _expanded_annotation_bbox(annotation, renderer, x_pad: float = 1.04, y_pad: float = 1.12):
    return annotation.get_window_extent(renderer=renderer).expanded(x_pad, y_pad)

def _keep_annotations_inside_axes(
    annotations: list,
    ax,
    renderer,
    pad_pixels: float,
    max_abs_x_offset_points: float,
    max_abs_y_offset_points: float,
) -> None:
    # Pull labels back into the axes area so edge labels do not get clipped.
    axes_bbox = ax.get_window_extent(renderer=renderer)
    for annotation in annotations:
        bbox = _expanded_annotation_bbox(annotation, renderer)
        dx_pixels = 0.0
        dy_pixels = 0.0
        if bbox.x0 < axes_bbox.x0 + pad_pixels:
            dx_pixels = axes_bbox.x0 + pad_pixels - bbox.x0
        elif bbox.x1 > axes_bbox.x1 - pad_pixels:
            dx_pixels = axes_bbox.x1 - pad_pixels - bbox.x1
        if bbox.y0 < axes_bbox.y0 + pad_pixels:
            dy_pixels = axes_bbox.y0 + pad_pixels - bbox.y0
        elif bbox.y1 > axes_bbox.y1 - pad_pixels:
            dy_pixels = axes_bbox.y1 - pad_pixels - bbox.y1
        if dx_pixels or dy_pixels:
            _move_annotation_by_pixels(
                annotation,
                dx_pixels=dx_pixels,
                dy_pixels=dy_pixels,
                max_abs_x_offset_points=max_abs_x_offset_points,
                max_abs_y_offset_points=max_abs_y_offset_points,
            )

def relax_annotations(
    annotations: list,
    ax=None,
    iterations: int = 140,
    pad_pixels: float = 3.0,
    max_step_pixels: float = 8.0,
    max_abs_x_offset_points: float = 48.0,
    max_abs_y_offset_points: float = 44.0,
) -> None:
    # Deterministically reduce point-label overlaps without external dependencies.
    if len(annotations) < 2:
        return

    if ax is None:
        ax = plt.gca()

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for annotation in annotations:
        _set_annotation_alignment_from_offset(annotation)

    for _ in range(iterations):
        moved = False
        bboxes = [_expanded_annotation_bbox(annotation, renderer) for annotation in annotations]

        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                bbox_i = bboxes[i]
                bbox_j = bboxes[j]
                overlap_x = min(bbox_i.x1, bbox_j.x1) - max(bbox_i.x0, bbox_j.x0)
                overlap_y = min(bbox_i.y1, bbox_j.y1) - max(bbox_i.y0, bbox_j.y0)
                if overlap_x <= pad_pixels or overlap_y <= pad_pixels:
                    continue

                cx_i = (bbox_i.x0 + bbox_i.x1) / 2.0
                cy_i = (bbox_i.y0 + bbox_i.y1) / 2.0
                cx_j = (bbox_j.x0 + bbox_j.x1) / 2.0
                cy_j = (bbox_j.y0 + bbox_j.y1) / 2.0

                sx = 1.0 if cx_i >= cx_j else -1.0
                sy = 1.0 if cy_i >= cy_j else -1.0
                if abs(cx_i - cx_j) < 1e-6:
                    sx = 1.0 if i % 2 == 0 else -1.0
                if abs(cy_i - cy_j) < 1e-6:
                    sy = 1.0 if i % 2 == 0 else -1.0

                x_step = min(max_step_pixels, max(1.0, overlap_x * 0.22))
                y_step = min(max_step_pixels, max(1.5, overlap_y * 0.55))

                _move_annotation_by_pixels(
                    annotations[i],
                    dx_pixels=sx * x_step,
                    dy_pixels=sy * y_step,
                    max_abs_x_offset_points=max_abs_x_offset_points,
                    max_abs_y_offset_points=max_abs_y_offset_points,
                )
                _move_annotation_by_pixels(
                    annotations[j],
                    dx_pixels=-sx * x_step,
                    dy_pixels=-sy * y_step,
                    max_abs_x_offset_points=max_abs_x_offset_points,
                    max_abs_y_offset_points=max_abs_y_offset_points,
                )
                moved = True

        _keep_annotations_inside_axes(
            annotations,
            ax=ax,
            renderer=renderer,
            pad_pixels=pad_pixels,
            max_abs_x_offset_points=max_abs_x_offset_points,
            max_abs_y_offset_points=max_abs_y_offset_points,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        if not moved:
            break

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
    series_info = add_scatter_and_regression(
        x,
        y,
        label=bucket_title(bucket),
        x_transform="log10",
        show_r2_in_legend=True,
        color=color_for_bucket(bucket),
    )
    label_color = series_info.get("color") if series_info else None

    for xi, yi, point_label in zip(x, y, labels):
        annotate_point(
            xi,
            yi,
            point_label,
            offset=FAMILY_LABEL_OFFSETS.get(point_label, (6, 6)),
            color=label_color,
        )

    plt.xscale("log")
    plt.xlabel("Parameter count (log scale)")
    plt.ylabel(y_label)
    plt.title(f"{bucket_title(bucket)}: model size (log scale) vs {metric_title(y_label)}")

    if series_info is not None:
        plt.legend(loc="best", fontsize=8, frameon=True)

    add_caption(
        "Dashed line shows least-squares trend fitted over log10(parameter count); "
        "R² is descriptive because each family has few checkpoints."
    )
    plt.tight_layout(rect=(0, 0.055, 1, 1))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=DPI, bbox_inches="tight")
    plt.close()

def get_bucket_specs(include_finetuned: bool = INCLUDE_FINETUNED_IN_COMBINED) -> list[tuple[str, str, tuple[int, int]]]:
    bucket_specs = [
        ("flan", "FLAN OOTB", (6, 6)),
        ("deberta_cross_encoder", "DeBERTa CE OOTB", (6, -12)),
        ("deberta_other_nli", "DeBERTa other NLI OOTB", (6, 10)),
        ("qwen", "Qwen OOTB", (6, -12)),
        ("gemma", "Gemma OOTB", (6, 6)),
    ]
    if include_finetuned:
        bucket_specs.extend([
            ("flan_lora_finetuned", "FLAN LoRA fine-tuned", (6, 6)),
            ("gemma_lora_finetuned", "Gemma LoRA fine-tuned", (6, 6)),
            ("qwen_lora_finetuned", "Qwen LoRA fine-tuned", (6, 6)),
            ("deberta_lora_finetuned_cross_encoder", "DeBERTa CE LoRA fine-tuned", (6, 12)),
            ("deberta_lora_finetuned_other_nli", "DeBERTa other NLI LoRA fine-tuned", (6, 14)),
        ])
    return bucket_specs

def _short_metric_label(y_label: str) -> str:
    # Compact metric label used in bar annotations.
    if y_label.lower() == "accuracy":
        return "acc"
    if "rho" in y_label.lower():
        return "rho"
    return y_label.lower()

def _wrap_family_tick(label: str) -> str:
    # Wrap long family/adaptation names for grouped bar chart tick labels.
    replacements = {
        " LoRA fine-tuned": "\nLoRA fine-tuned",
        "DeBERTa other NLI": "DeBERTa other NLI",
        "DeBERTa CE": "DeBERTa CE",
    }
    wrapped = label
    for old, new in replacements.items():
        wrapped = wrapped.replace(old, new)
    return wrapped

def _metric_annotation_lines(row: dict[str, Any]) -> list[str]:
    # Return compact performance annotation lines available for a metadata row.
    metric_lines: list[str] = []
    acc_value = row.get("agnostic_acc")
    rho_value = row.get("agnostic_rho")

    if acc_value is not None:
        metric_lines.append(f"Acc={float(acc_value):.3f}")
    if rho_value is not None:
        metric_lines.append(f"rho={float(rho_value):.3f}")

    return metric_lines

def make_combined_tradeoff_plot(
    rows: list[dict[str, Any]],
    y_key: str,
    y_label: str,
    output_name: str,
) -> None:
    """
    Backward-compatible wrapper for the older one-metric latency bar plots.
    """
    make_combined_latency_tradeoff_plot(
        rows=rows,
        output_name=output_name,
        title_suffix=f" ({metric_title(y_label)} rows)",
        require_metric_key=y_key,
    )

def make_combined_latency_tradeoff_plot(
    rows: list[dict[str, Any]],
    output_name: str = "tradeoff_latency_by_family.png",
    title_suffix: str = "",
    require_metric_key: str | None = None,
) -> None:
    """
    Plot mean inference latency as grouped bars by family/adaptation condition.
    """
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("mean_latency_seconds") is None:
            continue
        if require_metric_key is not None and row.get(require_metric_key) is None:
            continue
        valid_rows.append(row)

    grouped_rows: list[tuple[str, str, list[dict[str, Any]]]] = []
    for bucket, legend_label, _default_offset in get_bucket_specs():
        bucket_rows = sorted(
            [row for row in valid_rows if row["comparison_bucket"] == bucket],
            key=lambda x: x["parameter_count"] if x["parameter_count"] is not None else float("inf"),
        )
        if bucket_rows:
            grouped_rows.append((bucket, legend_label, bucket_rows))

    if not grouped_rows:
        return

    fig_width = max(COMBINED_FIGSIZE[0], 1.85 * len(grouped_rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.6))

    max_bars_in_group = max(len(bucket_rows) for _bucket, _label, bucket_rows in grouped_rows)
    bar_width = min(0.18, 0.78 / max_bars_in_group)
    group_spacing = 1.0
    max_latency = max(float(row["mean_latency_seconds"]) for row in valid_rows)
    annotation_y_offset = max(max_latency * 0.012, 0.002)

    x_centers: list[float] = []
    x_tick_labels: list[str] = []

    for group_idx, (_bucket, legend_label, bucket_rows) in enumerate(grouped_rows):
        group_center = group_idx * group_spacing
        x_centers.append(group_center)
        x_tick_labels.append(_wrap_family_tick(legend_label))

        offsets = (np.arange(len(bucket_rows)) - (len(bucket_rows) - 1) / 2.0) * (bar_width * 1.25)
        x_positions = group_center + offsets
        latencies = [float(row["mean_latency_seconds"]) for row in bucket_rows]

        bars = ax.bar(
            x_positions,
            latencies,
            width=bar_width,
            label=legend_label,
            color=color_for_bucket(_bucket),
            alpha=0.92,
            zorder=3,
        )

        for bar, row in zip(bars, bucket_rows):
            size_label = short_model_label(row["model_name"])
            latency_value = float(row["mean_latency_seconds"])
            metric_lines = _metric_annotation_lines(row)
            annotation = "\n".join([size_label] + metric_lines)

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                latency_value + annotation_y_offset,
                annotation,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                clip_on=False,
                zorder=4,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_tick_labels, rotation=12, ha="right")
    ax.set_xlabel("Model family / adaptation condition")
    ax.set_ylabel("Mean inference latency (seconds/example)")
    ax.set_title(f"Mean inference latency by family and size rung{title_suffix}")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

    ax.set_ylim(0, max_latency * 1.48)
    ax.margins(x=0.03)

    add_caption(
        "Bars show mean inference latency in seconds/example, grouped by family/adaptation condition and broken out by size rung;\n"
        "bar annotations report the corresponding accuracy and Spearman rho values when available."
    )
    plt.tight_layout(rect=(0, 0.085, 1, 0.98))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS_DIR / output_name, dpi=DPI, bbox_inches="tight")
    plt.close()

def make_combined_training_time_tradeoff_plot(
    rows: list[dict[str, Any]],
    output_name: str = "tradeoff_training_time_by_family.png",
) -> None:
    """
    Plot fine-tuning training time as grouped bars by family/adaptation condition.
    """
    valid_rows = [
        row for row in rows
        if row.get("total_training_runtime_seconds") is not None
    ]

    grouped_rows: list[tuple[str, str, list[dict[str, Any]]]] = []
    for bucket, legend_label, _default_offset in get_bucket_specs():
        bucket_rows = sorted(
            [row for row in valid_rows if row["comparison_bucket"] == bucket],
            key=lambda x: x["parameter_count"] if x["parameter_count"] is not None else float("inf"),
        )
        if bucket_rows:
            grouped_rows.append((bucket, legend_label, bucket_rows))

    if not grouped_rows:
        return

    fig_width = max(COMBINED_FIGSIZE[0], 1.85 * len(grouped_rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))

    max_bars_in_group = max(len(bucket_rows) for _bucket, _label, bucket_rows in grouped_rows)
    bar_width = min(0.18, 0.78 / max_bars_in_group)
    group_spacing = 1.0
    max_runtime = max(float(row["total_training_runtime_seconds"]) for row in valid_rows)
    annotation_y_offset = max(max_runtime * 0.012, 1.0)

    x_centers: list[float] = []
    x_tick_labels: list[str] = []

    for group_idx, (_bucket, legend_label, bucket_rows) in enumerate(grouped_rows):
        group_center = group_idx * group_spacing
        x_centers.append(group_center)
        x_tick_labels.append(_wrap_family_tick(legend_label))

        offsets = (np.arange(len(bucket_rows)) - (len(bucket_rows) - 1) / 2.0) * (bar_width * 1.25)
        x_positions = group_center + offsets
        runtimes = [float(row["total_training_runtime_seconds"]) for row in bucket_rows]

        bars = ax.bar(
            x_positions,
            runtimes,
            width=bar_width,
            label=legend_label,
            color=color_for_bucket(_bucket),
            alpha=0.92,
            zorder=3,
        )

        for bar, row in zip(bars, bucket_rows):
            size_label = short_model_label(row["model_name"])
            runtime_value = float(row["total_training_runtime_seconds"])
            acc_value = row.get("agnostic_acc")
            rho_value = row.get("agnostic_rho")

            metric_lines = []
            if acc_value is not None:
                metric_lines.append(f"Acc={float(acc_value):.3f}")
            if rho_value is not None:
                metric_lines.append(f"rho={float(rho_value):.3f}")
            annotation = "\n".join([size_label] + metric_lines)

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                runtime_value + annotation_y_offset,
                annotation,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7,
                clip_on=False,
                zorder=4,
            )

    ax.set_xticks(x_centers)
    ax.set_xticklabels(x_tick_labels, rotation=12, ha="right")
    ax.set_xlabel("Model family / adaptation condition")
    ax.set_ylabel("Fine-tuning training time (seconds)")
    ax.set_title("Fine-tuning training time by family and size rung")
    ax.grid(axis="y", linestyle="--", alpha=0.25, zorder=0)

    ax.set_ylim(0, max_runtime * 1.45)
    ax.margins(x=0.03)

    add_caption(
        "Bars show total fine-tuning runtime in seconds, grouped by family/adaptation condition and broken out by size rung;\n"
        "bar annotations report the corresponding accuracy and Spearman rho values when available. OOTB rows without training-runtime metadata are skipped."
    )
    plt.tight_layout(rect=(0, 0.085, 1, 0.98))

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

    plt.figure(figsize=COMBINED_FIGSIZE)
    annotations: list[Any] = []

    for bucket, legend_label, default_offset in get_bucket_specs():
        bucket_rows = sorted(
            [row for row in valid_rows if row["comparison_bucket"] == bucket],
            key=lambda x: x["parameter_count"] if x["parameter_count"] is not None else float("inf"),
        )

        if not bucket_rows:
            continue

        x = [row["parameter_count"] for row in bucket_rows]
        y = [row[y_key] for row in bucket_rows]
        draw_regression = bucket not in NO_REGRESSION_BUCKETS
        series_info = add_scatter_and_regression(
            x,
            y,
            label=legend_label,
            x_transform="log10",
            draw_regression=draw_regression,
            color=color_for_bucket(bucket),
        )
        label_color = series_info.get("color") if series_info else None

        for row in bucket_rows:
            point_label = short_model_label(row["model_name"])
            annotations.append(
                annotate_point(
                    row["parameter_count"],
                    row[y_key],
                    point_label,
                    offset=combined_size_label_offset(bucket, point_label, y_key, default_offset),
                    color=label_color,
                )
            )

    plt.xscale("log")
    plt.margins(x=0.08, y=0.12)
    relax_annotations(annotations, ax=plt.gca())
    plt.xlabel("Parameter count (log scale)")
    plt.ylabel(y_label)
    plt.title(f"Model size (log scale) vs {metric_title(y_label)}")
    plt.legend(loc="best", fontsize=8, frameon=True)
    add_caption(
        "Dashed lines show least-squares trends fitted within eligible family/adaptation conditions over log10(parameter count);\n"
        "DeBERTa other-NLI variants are shown as grouped points without a fitted regression line because they combine heterogeneous checkpoints.\n"
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
        "Dashed lines show family-wise or adaptation-condition-wise least-squares regressions fitted over "
        "log10(parameter count). Reported R² values are descriptive because each "
        "family contains only a small number of checkpoints.\n\n"
        "- **Latency trade-off plot:** Mean inference latency is shown in seconds per "
        "example on the y-axis. Bars are grouped by model family/adaptation condition and "
        "broken out by size rung; bar annotations report the corresponding accuracy and "
        "Spearman rho values where available. This plot summarizes observed runtime/performance "
        "trade-offs and should not be interpreted as a causal estimate.\n\n"
        "- **Training-time trade-off plot:** Total fine-tuning runtime is shown in seconds on "
        "the y-axis. Bars are grouped by family/adaptation condition and broken out by size "
        "rung; bar annotations report the corresponding accuracy and Spearman rho values "
        "where available. OOTB rows without training-runtime metadata are skipped.\n\n"
        "- **Fine-tuned adaptation buckets:** Fine-tuned DeBERTa and FLAN LoRA rows are plotted as separate "
        "adaptation conditions from their OOTB counterparts, even when they share the same base checkpoint family.\n",
        encoding="utf-8",
    )

def make_all_family_plots(rows: list[dict[str, Any]]) -> None:
    family_plot_specs = [
        ("flan", "flan"),
        ("flan_lora_finetuned", "flan_lora_finetuned"),
        ("deberta_cross_encoder", "deberta_cross_encoder"),
        ("deberta_other_nli", "deberta_other_nli"),
        ("deberta_lora_finetuned_cross_encoder", "deberta_lora_finetuned_cross_encoder"),
        ("deberta_lora_finetuned_other_nli", "deberta_lora_finetuned_other_nli"),
        # Backward-compatible names for older full fine-tuned metadata.
        ("deberta_finetuned_cross_encoder", "deberta_finetuned_cross_encoder"),
        ("deberta_finetuned_other_nli", "deberta_finetuned_other_nli"),
        ("qwen", "qwen"),
        ("gemma", "gemma"),
        ("qwen_lora_finetuned", "qwen_lora_finetuned"),
        ("gemma_lora_finetuned", "gemma_lora_finetuned"),
    ]

    for bucket, prefix in family_plot_specs:
        make_family_plot(
            rows=rows,
            bucket=bucket,
            y_key="agnostic_acc",
            y_label="Accuracy",
            output_name=f"{prefix}_size_vs_accuracy.png",
        )
        make_family_plot(
            rows=rows,
            bucket=bucket,
            y_key="agnostic_rho",
            y_label="Spearman rho",
            output_name=f"{prefix}_size_vs_rho.png",
        )
        make_family_plot(
            rows=rows,
            bucket=bucket,
            y_key="mean_latency_seconds",
            y_label="Mean inference latency (seconds/example)",
            output_name=f"{prefix}_size_vs_latency.png",
        )

def main() -> None:
    if not METADATA_DIR.exists():
        raise FileNotFoundError(f"Metadata directory not found: {METADATA_DIR}")

    rows = load_metadata(METADATA_DIR)

    if not rows:
        print("No metadata files found.")
        return

    counts_by_bucket: dict[str, int] = {}
    for row in rows:
        counts_by_bucket[row["comparison_bucket"]] = counts_by_bucket.get(row["comparison_bucket"], 0) + 1
    print("Loaded metadata rows by bucket:")
    for bucket, count in sorted(counts_by_bucket.items()):
        print(f"  {bucket}: {count}")

    for stale_latency_plot in ("tradeoff_latency_vs_accuracy.png", "tradeoff_latency_vs_rho.png"):
        stale_path = PLOTS_DIR / stale_latency_plot
        if stale_path.exists():
            stale_path.unlink()

    make_combined_latency_tradeoff_plot(
        rows=rows,
        output_name="tradeoff_latency_by_family.png",
    )
    make_combined_training_time_tradeoff_plot(
        rows=rows,
        output_name="tradeoff_training_time_by_family.png",
    )

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