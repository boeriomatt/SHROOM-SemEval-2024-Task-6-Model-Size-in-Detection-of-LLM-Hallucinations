# For colab_vscode_all_baselines_test_runner.ipynb

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
from datetime import datetime
import sys
import time
from typing import Any

import numpy as np
import torch

from src.data import build_examples, load_json, preview_examples
from src.models_flan import FlanJudge
from src.models_deberta import DebertaJudge
from src.models_qwen import QwenJudge
from src.models_gemma import GemmaJudge
from src.prompts import support_prompt

# Output folders
PRED_ARCHIVE_DIR = Path("outputs/predictions/archive")
PRED_SUBMISSION_ROOT = Path("outputs/predictions/submissions")
SCORES_DIR = Path("outputs/scores")
METADATA_DIR = Path("outputs/metadata")

# Participant kit paths
CHECKER_PATH = Path("participant_kit/check_output.py")
SCORER_PATH = Path("participant_kit/score.py")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHROOM OOTB hallucination detection experiments.")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["flan", "deberta", "qwen", "gemma"],
        help="Model family / inference backend.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model identifier or checkpoint path.",
    )
    parser.add_argument(
        "--input-path",
        default="data/SHROOM_dev-v2/val.model-agnostic.json",
        help="Path to validation/test input JSON.",
    )
    parser.add_argument(
        "--score-split",
        choices=["val", "test"],
        default="val",
        help="Participant-kit split format to use. Use 'test' for test.model-agnostic.json without --is_val.",
    )
    parser.add_argument(
        "--track",
        choices=["model-agnostic", "model-aware"],
        default="model-agnostic",
        help="SHROOM track filename suffix.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Reference directory for participant_kit/score.py. Defaults to dev dir for val and test-labeled dir for test.",
    )
    parser.add_argument(
        "--skip-participant-scorer",
        action="store_true",
        help="Write predictions/metadata but do not run participant_kit/check_output.py and score.py.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--preview-n",
        type=int,
        default=2,
        help="Number of examples to preview.",
    )
    parser.add_argument(
        "--prompt-version",
        default="support_prompt_v1",
        help="Prompt version label to store in metadata.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-text notes stored in metadata.",
    )
    parser.add_argument(
        "--warmup-path",
        default="data/SHROOM_trial-v1.1/trial-v1.json",
        help="Optional dataset used only for warmup before latency measurement.",
    )
    parser.add_argument(
        "--warmup-n",
        type=int,
        default=10,
        help="Number of warmup examples to run before measured evaluation. Set to 0 to disable warmup.",
    )
    return parser.parse_args()

def safe_filename(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )

def get_reference_dir(score_split: str, reference_dir: str | None) -> Path:
    if reference_dir:
        return Path(reference_dir)
    if score_split == "test":
        return Path("data/SHROOM_test-labeled")
    return Path("data/SHROOM_dev-v2")

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_directories() -> None:
    PRED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    PRED_SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

def build_judge(model_type: str, model_name: str):
    """
    Factory for model-specific judge objects.
    Each judge should expose predict(...) -> (label, p_hall, raw_text).
    """
    if model_type == "flan":
        return FlanJudge(model_name=model_name)
    if model_type == "deberta":
        return DebertaJudge(model_name=model_name)
    if model_type == "qwen":
        return QwenJudge(
            model_name=model_name,
            max_input_length=512,
            enable_raw_generation=False,
            use_single_token_verbalizers=True,
            use_4bit=(os.getenv("SHROOM_USE_4BIT", "0") == "1"),
            reserve_answer_tokens=8,
            attn_implementation="sdpa",
        )
    if model_type == "gemma":
        return GemmaJudge(
            model_name=model_name,
            max_input_length=512,
            enable_raw_generation=False,
            use_single_token_verbalizers=True,
            use_4bit=(os.getenv("SHROOM_USE_4BIT", "0") == "1"),
            reserve_answer_tokens=8,
            attn_implementation="sdpa",
        )
    raise ValueError(f"Unsupported model_type: {model_type}")

def run_checker(submission_dir: Path, score_split: str) -> None:
    print("\nRunning output checker...")
    cmd = [sys.executable, str(CHECKER_PATH), str(submission_dir)]
    if score_split == "val":
        cmd.append("--is_val")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Output checker failed.")

def run_scorer(submission_dir: Path, reference_dir: Path, score_path: Path, score_split: str) -> None:
    print("Running scorer...")
    cmd = [
        sys.executable,
        str(SCORER_PATH),
        str(submission_dir),
        str(reference_dir),
        str(score_path),
    ]
    if score_split == "val":
        cmd.append("--is_val")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Scorer failed.")

    print(f"Saved score file to: {score_path}")

def read_score_file(score_path: Path) -> dict[str, float | str]:
    scores: dict[str, float | str] = {}
    if not score_path.exists():
        return scores

    with score_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                scores[key] = float(value)
            except ValueError:
                scores[key] = value
    return scores

def get_model_parameter_count(judge) -> int | None:
    if hasattr(judge, "get_num_parameters"):
        try:
            return int(judge.get_num_parameters())
        except Exception:
            return None
    return None

def compute_direct_metrics(predictions: list[dict[str, Any]], examples: list[dict[str, Any]]) -> dict[str, float]:
    labels_available = all(ex.get("label") in {"Hallucination", "Not Hallucination"} for ex in examples)
    probs_available = all(ex.get("p_hallucination_gold") is not None for ex in examples)
    metrics: dict[str, float] = {}

    if labels_available:
        metrics["accuracy"] = float(
            sum(pred["label"] == ex["label"] for pred, ex in zip(predictions, examples)) / len(examples)
        )

    if probs_available:
        try:
            from scipy.stats import spearmanr

            rho = spearmanr(
                [pred["p(Hallucination)"] for pred in predictions],
                [ex["p_hallucination_gold"] for ex in examples],
            )[0]
            metrics["rho"] = float(rho)
        except Exception:
            pass

    return metrics

def save_metadata(
    metadata_path: Path,
    model_type: str,
    model_name: str,
    input_path: Path,
    reference_dir: Path,
    score_split: str,
    track: str,
    archive_pred_path: Path,
    submission_dir: Path,
    submission_pred_path: Path,
    score_path: Path | None,
    seed: int,
    num_examples: int,
    mean_latency_seconds: float | None,
    total_runtime_seconds: float | None,
    parameter_count: int | None,
    warmup_path: Path | None,
    warmup_examples_used: int,
    direct_metrics: dict[str, float],
    prompt_version: str = "support_prompt_v1",
    notes: str = "",
) -> None:
    score_contents = read_score_file(score_path) if score_path is not None else {}

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_type,
        "model_name": model_name,
        "safe_model_name": safe_filename(model_name),
        "input_path": str(input_path),
        "score_split": score_split,
        "track": track,
        "reference_dir": str(reference_dir),
        "archive_prediction_path": str(archive_pred_path),
        "participant_submission_dir": str(submission_dir),
        "participant_submission_path": str(submission_pred_path),
        "score_path": str(score_path) if score_path is not None else None,
        "checker_path": str(CHECKER_PATH),
        "scorer_path": str(SCORER_PATH),
        "seed": seed,
        "num_examples": num_examples,
        "prompt_version": prompt_version,
        "notes": notes,
        "scores": score_contents,
        "direct_metrics": direct_metrics,
        "warmup": {
            "warmup_path": str(warmup_path) if warmup_path is not None else None,
            "warmup_examples_used": warmup_examples_used,
        },
        "computational_cost": {
            "parameter_count": parameter_count,
            "mean_inference_latency_seconds_per_example": mean_latency_seconds,
            "total_inference_runtime_seconds": total_runtime_seconds,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata file to: {metadata_path}")

def run_warmup(judge, model_type: str, warmup_path: Path, warmup_n: int) -> int:
    if warmup_n <= 0:
        print("\nWarmup disabled.")
        return 0

    if not warmup_path.exists():
        print(f"\nWarmup skipped: file not found at {warmup_path}")
        return 0

    print(f"\nLoading warmup data from: {warmup_path}")
    warmup_raw = load_json(warmup_path)
    warmup_examples = build_examples(warmup_raw)[:warmup_n]

    if not warmup_examples:
        print("Warmup skipped: no warmup examples available.")
        return 0

    print(f"Running {len(warmup_examples)} warmup examples...")
    for i, ex in enumerate(warmup_examples, start=1):
        if model_type == "flan":
            prompt = support_prompt(ex["context"], ex["hyp"])
            _label, _p_hall, _raw_text = judge.predict(prompt)
        elif model_type == "deberta":
            _label, _p_hall, _raw_text = judge.predict(ex["context"], ex["hyp"])
        elif model_type in {"qwen", "gemma"}:
            _label, _p_hall, _raw_text = judge.predict(ex["context"], ex["hyp"])
        else:
            raise ValueError(f"Unsupported model_type during warmup: {model_type}")

        if i <= 2:
            print(f"Warmup example {i} complete.")

    print("Warmup complete.")
    return len(warmup_examples)

def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    reference_dir = get_reference_dir(args.score_split, args.reference_dir)
    safe_model = safe_filename(args.model_name)

    file_stem = f"{args.score_split}.{args.track}"
    archive_pred_path = PRED_ARCHIVE_DIR / f"{file_stem}__{safe_model}.json"
    submission_dir = PRED_SUBMISSION_ROOT / f"{file_stem}__{safe_model}"
    submission_pred_path = submission_dir / f"{file_stem}.json"
    score_path = SCORES_DIR / f"{args.score_split}_scores__{safe_model}.txt"
    metadata_path = METADATA_DIR / f"run__{args.score_split}__{safe_model}.json"

    set_seed(args.seed)
    ensure_directories()
    submission_dir.mkdir(parents=True, exist_ok=True)

    for old_json in submission_dir.glob("*.json"):
        old_json.unlink()

    print(f"Loading data from: {input_path}")
    raw_data = load_json(input_path)
    examples = build_examples(raw_data)

    print(f"Loaded {len(examples)} examples.")
    preview_examples(examples, n=args.preview_n)

    judge = build_judge(args.model_type, args.model_name)

    warmup_path = Path(args.warmup_path) if args.warmup_path else None
    warmup_examples_used = 0
    if warmup_path is not None:
        warmup_examples_used = run_warmup(
            judge=judge,
            model_type=args.model_type,
            warmup_path=warmup_path,
            warmup_n=args.warmup_n,
        )

    parameter_count = get_model_parameter_count(judge)
    if parameter_count is not None:
        print(f"\nModel parameter count: {parameter_count:,}")

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []

    total_start = time.perf_counter()

    for i, ex in enumerate(examples, start=1):
        if args.model_type == "flan":
            model_input = support_prompt(ex["context"], ex["hyp"])
        elif args.model_type in {"deberta", "qwen", "gemma"}:
            model_input = (ex["context"], ex["hyp"])
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        if args.model_type == "flan":
            label, p_hall, raw_text = judge.predict(model_input)
        elif args.model_type == "deberta":
            context, hyp = model_input
            label, p_hall, raw_text = judge.predict(context, hyp)
        elif args.model_type in {"qwen", "gemma"}:
            context, hyp = model_input
            label, p_hall, raw_text = judge.predict(context, hyp)
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        latency = end_time - start_time
        latencies.append(latency)

        pred: dict[str, Any] = {
            "label": label,
            "p(Hallucination)": float(p_hall),
        }
        if args.score_split == "test":
            if ex.get("id") is None:
                raise ValueError("Test-mode predictions require an id for each example, but found id=None.")
            pred["id"] = int(ex["id"])

        predictions.append(pred)

        if i <= 3:
            print("\n--- Prediction preview ---")
            print(f"Example: {i}")
            print(f"id: {ex.get('id')}")
            print(f"Raw output: {raw_text}")
            print(f"Predicted label: {label}")
            print(f"Predicted p(Hallucination): {p_hall}")
            print(f"Latency (s): {latency:.6f}")

    total_end = time.perf_counter()
    total_runtime_seconds = total_end - total_start
    mean_latency_seconds = sum(latencies) / len(latencies) if latencies else None

    with archive_pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with submission_pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\nSaved archive predictions to: {archive_pred_path}")
    print(f"Saved participant submission predictions to: {submission_pred_path}")

    if mean_latency_seconds is not None:
        print(f"Mean inference latency per example: {mean_latency_seconds:.6f} seconds")
    print(f"Total inference runtime: {total_runtime_seconds:.6f} seconds")

    direct_metrics = compute_direct_metrics(predictions, examples)
    if direct_metrics:
        print("Direct metrics from labeled input:", direct_metrics)

    final_score_path: Path | None = None
    if not args.skip_participant_scorer:
        run_checker(submission_dir, args.score_split)
        run_scorer(submission_dir, reference_dir, score_path, args.score_split)
        final_score_path = score_path
    else:
        print("Participant checker/scorer skipped.")

    save_metadata(
        metadata_path=metadata_path,
        model_type=args.model_type,
        model_name=args.model_name,
        input_path=input_path,
        reference_dir=reference_dir,
        score_split=args.score_split,
        track=args.track,
        archive_pred_path=archive_pred_path,
        submission_dir=submission_dir,
        submission_pred_path=submission_pred_path,
        score_path=final_score_path,
        seed=args.seed,
        num_examples=len(examples),
        mean_latency_seconds=mean_latency_seconds,
        total_runtime_seconds=total_runtime_seconds,
        parameter_count=parameter_count,
        warmup_path=warmup_path,
        warmup_examples_used=warmup_examples_used,
        direct_metrics=direct_metrics,
        prompt_version=args.prompt_version,
        notes=args.notes,
    )

    print("Done.")

if __name__ == "__main__":
    main()