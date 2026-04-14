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
from src.prompts import support_prompt

# Later, once implemented:
# from src.models_deberta import DebertaJudge
# from src.models_qwen_ollama import OllamaJudge

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
        choices=["flan", "deberta", "qwen_ollama"],
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

    # Uncomment once implemented
    # elif model_type == "deberta":
    #     return DebertaJudge(model_name=model_name)
    # elif model_type == "qwen_ollama":
    #     return OllamaJudge(model=model_name)

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
        "computational_cost": {
            "parameter_count": parameter_count,
            "mean_inference_latency_seconds_per_example": mean_latency_seconds,
            "total_inference_runtime_seconds": total_runtime_seconds,
        },
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata file to: {metadata_path}")

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
    parameter_count = get_model_parameter_count(judge)
    if parameter_count is not None:
        print(f"Model parameter count: {parameter_count:,}")
    
    predictions = []
    latencies = []

    total_start = time.perf_counter()

    for i, ex in enumerate(examples, start=1):
        prompt = support_prompt(ex["context"], ex["hyp"])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        label, p_hall, raw_text = judge.predict(prompt)

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
        prompt_version=args.prompt_version,
        notes=args.notes,
    )

    print("Done.")

if __name__ == "__main__":
    main()