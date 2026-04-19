import argparse
import json
from pathlib import Path
import random
import subprocess
from datetime import datetime
import sys
import time

import numpy as np
import torch

from src.data import build_examples, load_json, preview_examples
from src.models_flan import FlanJudge
from src.models_deberta import DebertaJudge
from src.models_qwen import QwenJudge
from src.prompts import support_prompt

# Output folders
PRED_ARCHIVE_DIR = Path("outputs/predictions/archive") 
PRED_CURRENT_DIR = Path("outputs/predictions/current")
SCORES_DIR = Path("outputs/scores")
METADATA_DIR = Path("outputs/metadata")

# Participant kit paths
CHECKER_PATH = Path("participant_kit/check_output.py")
SCORER_PATH = Path("participant_kit/score.py")
REFERENCE_DIR = Path("data/SHROOM_dev-v2")

# Argument parsing for flexible experiment configuration
def parse_args():
    parser = argparse.ArgumentParser(description="Run SHROOM hallucination detection experiments.")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=["flan", "deberta", "qwen"],
        help="Model family / inference backend."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model identifier or checkpoint path."
    )
    parser.add_argument(
        "--input-path",
        default="data/SHROOM_dev-v2/val.model-agnostic.json",
        help="Path to validation/test input JSON."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--preview-n",
        type=int,
        default=2,
        help="Number of examples to preview."
    )
    parser.add_argument(
        "--prompt-version",
        default="support_prompt_v1",
        help="Prompt version label to store in metadata."
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional free-text notes stored in metadata."
    )
    parser.add_argument(
        "--warmup-path",
        default="data/SHROOM_trial-v1.1/trial-v1.json",
        help="Optional dataset used only for warmup before latency measurement."
    )
    parser.add_argument(
        "--warmup-n",
        type=int,
        default=10,
        help="Number of warmup examples to run before measured evaluation. Set to 0 to disable warmup."
    )
    return parser.parse_args()

# Clean the model name to create safe filenames for outputs
def safe_filename(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )

# Set random seeds for reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Ensure that all necessary output directories exist
def ensure_directories() -> None:
    PRED_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    PRED_CURRENT_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Model-specific judge factory
def build_judge(model_type: str, model_name: str):
    """
    - Factory for model-specific judge objects
    - Each judge should expose: predict(prompt_or_inputs) -> (label, p_hall, raw_text)
    """
    if model_type == "flan":
        return FlanJudge(model_name=model_name)
    elif model_type == "deberta":
        return DebertaJudge(model_name=model_name)
    elif model_type == "qwen":
        return QwenJudge(model_name=model_name, max_input_length=512, enable_raw_generation=False, use_single_token_verbalizers=True, use_4bit=False, reserve_answer_tokens=8, attn_implementation="sdpa")
    raise ValueError(f"Unsupported model_type: {model_type}")

# Run the output checker script from the participant kit
def run_checker(pred_current_dir: Path) -> None:
    print("\nRunning output checker...")
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER_PATH),
            str(pred_current_dir),
            "--is_val",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Output checker failed.")

# Run the scorer script from the participant kit
def run_scorer(pred_current_dir: Path, score_path: Path) -> None:
    print("Running scorer...")
    result = subprocess.run(
        [
            sys.executable,
            str(SCORER_PATH),
            str(pred_current_dir),
            str(REFERENCE_DIR),
            str(score_path),
            "--is_val",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Scorer failed.")

    print(f"Saved score file to: {score_path}")

# Read the score file and return a dictionary of scores
def read_score_file(score_path: Path) -> dict[str, float | str]:
    scores = {}

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

# Attempt to get the model's parameter count if the judge exposes that information
def get_model_parameter_count(judge) -> int | None:
    if hasattr(judge, "get_num_parameters"):
        try:
            return int(judge.get_num_parameters())
        except Exception:
            return None
    return None

# Save metadata about the experiment, including scores and parameters, to a JSON file
def save_metadata(
    metadata_path: Path,
    model_type: str,
    model_name: str,
    input_path: Path,
    archive_pred_path: Path,
    current_pred_path: Path,
    score_path: Path,
    seed: int,
    num_examples: int,
    mean_latency_seconds: float | None,
    total_runtime_seconds: float | None,
    parameter_count: int | None,
    warmup_path: Path | None,
    warmup_examples_used: int,
    prompt_version: str = "support_prompt_v1",
    notes: str = "",
) -> None:
    score_contents = read_score_file(score_path)

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_type,
        "model_name": model_name,
        "safe_model_name": safe_filename(model_name),
        "input_path": str(input_path),
        "reference_dir": str(REFERENCE_DIR),
        "archive_prediction_path": str(archive_pred_path),
        "current_prediction_path": str(current_pred_path),
        "score_path": str(score_path),
        "checker_path": str(CHECKER_PATH),
        "scorer_path": str(SCORER_PATH),
        "seed": seed,
        "num_examples": num_examples,
        "prompt_version": prompt_version,
        "notes": notes,
        "scores": score_contents,
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

# Run a warmup phase using a trial dataset to reduce startup-related latency confounds before the main evaluation loop
def run_warmup(judge, model_type: str, warmup_path: Path, warmup_n: int) -> int:
    """
    Run a short untimed warmup phase to reduce startup-related latency confounds.
    Returns:
    - number of warmup examples actually used
    """
    if warmup_n <= 0:
        print("\nWarmup disabled.")
        return 0

    if not warmup_path.exists():
        print(f"\nWarmup skipped: file not found at {warmup_path}")
        return 0

    print(f"\nLoading warmup data from: {warmup_path}")
    warmup_raw = load_json(warmup_path)
    warmup_examples = build_examples(warmup_raw)

    warmup_examples = warmup_examples[:warmup_n]

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

        elif model_type == "qwen":
            _label, _p_hall, _raw_text = judge.predict(ex["context"], ex["hyp"])

        else:
            raise ValueError(f"Unsupported model_type during warmup: {model_type}")

        if i <= 2:
            print(f"Warmup example {i} complete.")

    print("Warmup complete.")
    return len(warmup_examples)

# Main function to run the experiment
def main() -> None:
    args = parse_args()

    input_path = Path(args.input_path)
    safe_model = safe_filename(args.model_name)

    archive_pred_path = PRED_ARCHIVE_DIR / f"val.model-agnostic__{safe_model}.json"
    current_pred_path = PRED_CURRENT_DIR / "val.model-agnostic.json"
    score_path = SCORES_DIR / f"val_scores__{safe_model}.txt"
    metadata_path = METADATA_DIR / f"run__{safe_model}.json"

    set_seed(args.seed)
    ensure_directories()

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
    
    predictions = []
    latencies = []

    total_start = time.perf_counter()

    for i, ex in enumerate(examples, start=1):
        if args.model_type == "flan":
            model_input = support_prompt(ex["context"], ex["hyp"])
        elif args.model_type == "deberta":
            model_input = (ex["context"], ex["hyp"])
        elif args.model_type == "qwen":
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
        elif args.model_type == "qwen":
            context, hyp = model_input
            label, p_hall, raw_text = judge.predict(context, hyp)
        else:
            raise ValueError(f"Unsupported model_type: {args.model_type}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        latency = end_time - start_time
        latencies.append(latency)

        predictions.append(
            {
                "label": label,
                "p(Hallucination)": float(p_hall),
            }
        )

        if i <= 3:
            print("\n--- Prediction preview ---")
            print(f"Example: {i}")
            print(f"Raw output: {raw_text}")
            print(f"Predicted label: {label}")
            print(f"Predicted p(Hallucination): {p_hall}")
            print(f"Latency (s): {latency:.6f}")

    total_end = time.perf_counter()
    total_runtime_seconds = total_end - total_start
    mean_latency_seconds = sum(latencies) / len(latencies) if latencies else None

    with archive_pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with current_pred_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"\nSaved archive predictions to: {archive_pred_path}")
    print(f"Saved current scoring predictions to: {current_pred_path}")

    if mean_latency_seconds is not None:
        print(f"Mean inference latency per example: {mean_latency_seconds:.6f} seconds")
    print(f"Total inference runtime: {total_runtime_seconds:.6f} seconds")
    
    run_checker(PRED_CURRENT_DIR)
    run_scorer(PRED_CURRENT_DIR, score_path)

    save_metadata(
        metadata_path=metadata_path,
        model_type=args.model_type,
        model_name=args.model_name,
        input_path=input_path,
        archive_pred_path=archive_pred_path,
        current_pred_path=current_pred_path,
        score_path=score_path,
        seed=args.seed,
        num_examples=len(examples),
        mean_latency_seconds=mean_latency_seconds,
        total_runtime_seconds=total_runtime_seconds,
        parameter_count=parameter_count,
        warmup_path=warmup_path,
        warmup_examples_used=warmup_examples_used,
        prompt_version=args.prompt_version,
        notes=args.notes,
    )

    print("Done.")

if __name__ == "__main__":
    main()