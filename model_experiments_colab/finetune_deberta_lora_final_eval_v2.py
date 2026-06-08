"""
For colab_vscode_deberta_lora_finetune_test_runner_v2.ipynb

Final test-set fine-tune/evaluate DeBERTa-style cross-encoder hallucination judges on SHROOM with LoRA.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError as exc:  # pragma: no cover - user-facing dependency error
    raise ImportError("This script requires PEFT. Install it with: pip install peft") from exc

from src.data import build_examples, load_json, preview_examples

SCRIPT_VERSION = "deberta_lora_final_eval_v2_ids_20260510"

PRED_ARCHIVE_DIR = Path("outputs/predictions/archive")
PRED_CURRENT_DIR = Path("outputs/predictions/current")
PRED_SUBMISSIONS_DIR = Path("outputs/predictions/submissions")
SCORES_DIR = Path("outputs/scores")
METADATA_DIR = Path("outputs/metadata")
FINETUNED_DIR = Path("outputs/finetuned/deberta_lora")

CHECKER_PATH = Path("participant_kit/check_output.py")
SCORER_PATH = Path("participant_kit/score.py")
REFERENCE_DIR = Path("data/SHROOM_dev-v2")

LABEL2ID = {"Not Hallucination": 0, "Hallucination": 1}
ID2LABEL = {0: "Not Hallucination", 1: "Hallucination"}

def safe_filename(text: str) -> str:
    return (
        text.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_directories() -> None:
    for path in [
        PRED_ARCHIVE_DIR,
        PRED_CURRENT_DIR,
        PRED_SUBMISSIONS_DIR,
        SCORES_DIR,
        METADATA_DIR,
        FINETUNED_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

def label_to_id(label: Any) -> int:
    label_str = str(label)
    if label_str not in LABEL2ID:
        raise ValueError(f"Unsupported label {label!r}; expected one of {list(LABEL2ID)}")
    return LABEL2ID[label_str]

def get_gold_probability(example: dict[str, Any]) -> float:
    value = example.get("p_hallucination_gold")
    if value is None:
        return float(label_to_id(example["label"]))
    return float(value)

def spearmanr_safe(y_true: list[float], y_pred: list[float]) -> float | None:
    if len(y_true) < 2:
        return None
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return None

    try:
        from scipy.stats import spearmanr

        rho = spearmanr(y_true, y_pred).correlation
        if rho is None or math.isnan(float(rho)):
            return None
        return float(rho)
    except Exception:
        def rankdata(values: list[float]) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            sorter = np.argsort(arr, kind="mergesort")
            ranks = np.empty(len(arr), dtype=float)
            i = 0
            while i < len(arr):
                j = i
                while j + 1 < len(arr) and arr[sorter[j + 1]] == arr[sorter[i]]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0
                ranks[sorter[i : j + 1]] = avg_rank
                i = j + 1
            return ranks

        rx = rankdata(y_true)
        ry = rankdata(y_pred)
        corr = np.corrcoef(rx, ry)[0, 1]
        if math.isnan(float(corr)):
            return None
        return float(corr)

class ShroomPairDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer,
        max_length: int,
        label_mode: str,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mode = label_mode

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        encoding = self.tokenizer(
            str(ex["context"]),
            str(ex["hyp"]),
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        hard_label = label_to_id(ex["label"])
        gold_prob = get_gold_probability(ex)

        if self.label_mode == "soft":
            encoding["labels"] = float(gold_prob)
        else:
            encoding["labels"] = int(hard_label)

        encoding["hard_labels"] = int(hard_label)
        encoding["gold_probs"] = float(gold_prob)
        return encoding

@dataclass
class EvalMetrics:
    loss: float | None
    accuracy: float | None
    rho: float | None
    num_examples: int

def load_examples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    raw = load_json(path)
    examples = build_examples(raw)
    if limit is not None and limit > 0:
        examples = examples[:limit]
    return examples

def load_reference_ids(path: Path, limit: int | None = None) -> list[Any]:
    raw = load_json(path)
    if limit is not None and limit > 0:
        raw = raw[:limit]
    return [item.get("id") for item in raw]

def attach_reference_ids(examples: list[dict[str, Any]], reference_ids: list[Any] | None) -> None:
    if reference_ids is None:
        return
    if len(reference_ids) != len(examples):
        raise ValueError(
            f"Cannot attach reference ids: got {len(reference_ids)} ids for {len(examples)} examples."
        )
    for ex, ref_id in zip(examples, reference_ids):
        if ex.get("id") is None and ref_id is not None:
            ex["id"] = ref_id

def split_examples(
    examples: list[dict[str, Any]],
    eval_split: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (0.0 < eval_split < 1.0):
        raise ValueError("eval_split must be between 0 and 1 when splitting one file.")

    indices = list(range(len(examples)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    eval_size = max(1, int(round(len(indices) * eval_split)))
    eval_indices = set(indices[:eval_size])

    train_examples = [ex for i, ex in enumerate(examples) if i not in eval_indices]
    eval_examples = [ex for i, ex in enumerate(examples) if i in eval_indices]
    return train_examples, eval_examples

def build_dataloader(
    examples: list[dict[str, Any]],
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    label_mode: str,
) -> DataLoader:
    dataset = ShroomPairDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_length=max_length,
        label_mode=label_mode,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}

def split_batch_for_model(batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = batch.pop("labels")
    hard_labels = batch.pop("hard_labels")
    gold_probs = batch.pop("gold_probs")
    return batch, labels, hard_labels, gold_probs

def compute_loss_and_probs(outputs, labels: torch.Tensor, label_mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    if label_mode == "soft":
        logits = outputs.logits.squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        probs = torch.sigmoid(logits)
    else:
        loss = F.cross_entropy(outputs.logits, labels.long())
        probs = torch.softmax(outputs.logits, dim=-1)[:, LABEL2ID["Hallucination"]]
    return loss, probs

@torch.no_grad()
def evaluate_model(
    model,
    data_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    label_mode: str,
) -> EvalMetrics:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    correct = 0
    gold_probs_all: list[float] = []
    pred_probs: list[float] = []

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        model_inputs, labels, hard_labels, gold_probs = split_batch_for_model(batch)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(**model_inputs)
            loss, probs = compute_loss_and_probs(outputs, labels, label_mode)

        batch_size = int(hard_labels.shape[0])
        total_seen += batch_size
        total_loss += float(loss.item()) * batch_size

        preds = (probs >= 0.5).long()
        correct += int((preds == hard_labels.long()).sum().item())

        gold_probs_all.extend(gold_probs.detach().cpu().float().tolist())
        pred_probs.extend(probs.detach().cpu().float().tolist())

    if total_seen == 0:
        return EvalMetrics(loss=None, accuracy=None, rho=None, num_examples=0)

    return EvalMetrics(
        loss=total_loss / total_seen,
        accuracy=correct / total_seen,
        rho=spearmanr_safe(gold_probs_all, pred_probs),
        num_examples=total_seen,
    )

def metric_value(metrics: EvalMetrics, best_metric: str) -> float:
    if best_metric == "loss":
        return float("inf") if metrics.loss is None else float(metrics.loss)
    value = getattr(metrics, best_metric)
    return float("-inf") if value is None else float(value)

def is_better(metrics: EvalMetrics, best_metrics: EvalMetrics | None, best_metric: str) -> bool:
    if best_metrics is None:
        return True
    current = metric_value(metrics, best_metric)
    previous = metric_value(best_metrics, best_metric)
    if best_metric == "loss":
        return current < previous
    return current > previous

@torch.no_grad()
def run_warmup_examples(
    model,
    tokenizer,
    warmup_path: Path | None,
    warmup_n: int,
    max_length: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> int:
    if warmup_path is None or warmup_n <= 0:
        print("\nWarmup disabled.")
        return 0
    if not warmup_path.exists():
        print(f"\nWarmup skipped: file not found at {warmup_path}")
        return 0

    warmup_examples = load_examples(warmup_path, limit=warmup_n)
    if not warmup_examples:
        print("Warmup skipped: no examples available.")
        return 0

    print(f"\nRunning {len(warmup_examples)} warmup examples...")
    model.eval()
    for ex in warmup_examples:
        inputs = tokenizer(
            str(ex["context"]),
            str(ex["hyp"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            _ = model(**inputs)
    print("Warmup complete.")
    return len(warmup_examples)

@torch.no_grad()
def predict_examples_timed(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    max_length: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    label_mode: str,
    preview_n: int,
) -> tuple[list[dict[str, Any]], float | None, float | None]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []

    total_start = time.perf_counter()

    for i, ex in enumerate(examples, start=1):
        inputs = tokenizer(
            str(ex["context"]),
            str(ex["hyp"]),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = model(**inputs)
            if label_mode == "soft":
                p_hall = float(torch.sigmoid(outputs.logits.squeeze(-1))[0].detach().cpu().item())
            else:
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                p_hall = float(probs[LABEL2ID["Hallucination"]].detach().cpu().item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        label = "Hallucination" if p_hall >= 0.5 else "Not Hallucination"
        latency = end - start
        latencies.append(latency)

        pred: dict[str, Any] = {"label": label, "p(Hallucination)": p_hall}
        if ex.get("id") is not None:
            pred["id"] = ex["id"]
        predictions.append(pred)

        if i <= preview_n:
            print("\n--- DeBERTa LoRA prediction preview ---")
            print(f"Example: {i}")
            print(f"Gold label: {ex.get('label')}")
            print(f"Gold p(Hallucination): {get_gold_probability(ex):.3f}")
            print(f"Predicted label: {label}")
            print(f"Predicted p(Hallucination): {p_hall:.6f}")
            print(f"Latency (s): {latency:.6f}")

    total_end = time.perf_counter()
    mean_latency = sum(latencies) / len(latencies) if latencies else None
    total_runtime = total_end - total_start if latencies else None
    return predictions, mean_latency, total_runtime

def compute_prediction_metrics(
    examples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, float | None]:
    if len(examples) != len(predictions):
        return {"accuracy": None, "rho": None}

    gold_hard = [label_to_id(ex["label"]) for ex in examples]
    gold_probs = [get_gold_probability(ex) for ex in examples]
    pred_labels = [label_to_id(pred["label"]) for pred in predictions]
    pred_probs = [float(pred["p(Hallucination)"]) for pred in predictions]

    accuracy = sum(int(g == p) for g, p in zip(gold_hard, pred_labels)) / len(gold_hard) if gold_hard else None
    rho = spearmanr_safe(gold_probs, pred_probs)
    return {"accuracy": accuracy, "rho": rho}

def ensure_required_submission_ids(
    predictions: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    score_split: str,
) -> None:
    """
    Ensure final test submissions contain ids before writing/scoring.
    """
    if score_split == "val":
        return

    if len(predictions) != len(examples):
        raise ValueError(
            f"Cannot attach test ids: got {len(predictions)} predictions for {len(examples)} examples."
        )

    missing_positions: list[int] = []
    for idx, (pred, ex) in enumerate(zip(predictions, examples), start=1):
        if pred.get("id") is None and ex.get("id") is not None:
            pred["id"] = ex["id"]
        if pred.get("id") is None:
            missing_positions.append(idx)

    if missing_positions:
        preview = missing_positions[:10]
        raise ValueError(
            "Test submissions require an `id` field, but ids are missing for "
            f"{len(missing_positions)} predictions. First missing positions: {preview}. "
            "Check that --eval-path points to the labeled SHROOM test file with ids."
        )

def parse_score_file(score_path: Path) -> dict[str, float | str]:
    scores: dict[str, float | str] = {}
    if score_path.exists():
        for line in score_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                scores[key.strip()] = float(value.strip())
            except ValueError:
                scores[key.strip()] = value.strip()
    return scores

def run_checker_and_scorer(
    prediction_dir: Path,
    score_path: Path,
    reference_dir: Path,
    score_split: str,
) -> dict[str, float | str]:
    print("\nRunning participant output checker...")
    checker_cmd = [sys.executable, str(CHECKER_PATH), str(prediction_dir)]
    if score_split == "val":
        checker_cmd.append("--is_val")

    checker_result = subprocess.run(checker_cmd, capture_output=True, text=True)
    print(checker_result.stdout)
    if checker_result.returncode != 0:
        print(checker_result.stderr)
        raise RuntimeError("Participant output checker failed.")

    print("Running participant scorer...")
    scorer_cmd = [
        sys.executable,
        str(SCORER_PATH),
        str(prediction_dir),
        str(reference_dir),
        str(score_path),
    ]
    if score_split == "val":
        scorer_cmd.append("--is_val")

    scorer_result = subprocess.run(scorer_cmd, capture_output=True, text=True)
    print(scorer_result.stdout)
    if scorer_result.returncode != 0:
        print(scorer_result.stderr)
        raise RuntimeError("Participant scorer failed.")

    return parse_score_file(score_path)

def parse_variants(text: str | None) -> list[str]:
    if text is None:
        return []
    values: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        value = part.strip()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values

def count_trainable_parameters(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        n = parameter.numel()
        total += n
        if parameter.requires_grad:
            trainable += n
    return total, trainable

def resolve_dtype(args: argparse.Namespace, device: torch.device) -> tuple[torch.dtype, bool, bool]:
    if device.type != "cuda":
        return torch.float32, False, False
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --fp16 or --bf16.")
    if args.bf16:
        return torch.bfloat16, True, False
    if args.fp16:
        return torch.float16, True, True
    return torch.float32, False, False

def build_sequence_classifier(
    model_name: str,
    label_mode: str,
    device: torch.device,
    args: argparse.Namespace,
    amp_dtype: torch.dtype,
):
    model_kwargs: dict[str, Any] = {
        "ignore_mismatched_sizes": True,
    }

    if label_mode == "soft":
        model_kwargs["num_labels"] = 1
    else:
        model_kwargs["num_labels"] = 2
        model_kwargs["id2label"] = ID2LABEL
        model_kwargs["label2id"] = LABEL2ID

    if device.type == "cuda":
        model_kwargs["torch_dtype"] = amp_dtype
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    if amp_dtype == torch.float32:
        model = model.float()
    model.to(device)

    first_param = next(model.parameters(), None)
    if first_param is not None:
        print(f"Loaded model parameter dtype: {first_param.dtype}")
    return model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final test-set fine-tune/evaluate DeBERTa on SHROOM hallucination labels with LoRA.")
    parser.add_argument("--model-name", required=True, help="HF DeBERTa/cross-encoder model id or local checkpoint path.")
    parser.add_argument(
        "--train-path",
        default="data/SHROOM_dev-v2/val.model-agnostic.json",
        help="Labeled SHROOM JSON used for training.",
    )
    parser.add_argument(
        "--eval-path",
        default=None,
        help="Optional labeled SHROOM JSON used for final evaluation. If omitted, train-path is split.",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.2,
        help="Held-out split fraction when eval-path is omitted.",
    )
    parser.add_argument(
        "--final-eval-only",
        action="store_true",
        help=(
            "With --eval-path, train for the fixed number of epochs, save final checkpoint, "
            "and evaluate once. This avoids selecting checkpoints using the test set."
        ),
    )
    parser.add_argument(
        "--allow-train-eval-overlap",
        action="store_true",
        help="Allow training and evaluating on the same examples. Use only for debugging.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["soft", "hard"],
        default="soft",
        help="soft = BCEWithLogitsLoss on p(Hallucination); hard = 2-class cross-entropy on majority label.",
    )
    parser.add_argument("--output-dir", default=str(FINETUNED_DIR), help="Base output directory for checkpoints.")
    parser.add_argument("--run-tag", default=None, help="Optional run tag. Defaults to timestamp.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", help="Use CUDA fp16 autocast if a CUDA GPU is available.")
    parser.add_argument("--bf16", action="store_true", help="Use CUDA bf16 autocast if a CUDA GPU is available.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional transformers attention implementation, e.g. sdpa if supported by the model.",
    )
    parser.add_argument(
        "--best-metric",
        choices=["rho", "accuracy", "loss"],
        default="rho",
        help="Metric used to select best checkpoint during internal validation runs.",
    )
    parser.add_argument("--train-limit", type=int, default=None, help="Optional training subset size for smoke tests.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Optional eval subset size for smoke tests.")
    parser.add_argument("--preview-n", type=int, default=2)
    parser.add_argument(
        "--warmup-path",
        default="data/SHROOM_trial-v1.1/trial-v1.json",
        help="Optional warmup data for final timed inference.",
    )
    parser.add_argument("--warmup-n", type=int, default=10)
    parser.add_argument("--write-current", action="store_true", help="Also write outputs/predictions/current/<split>.model-agnostic.json.")
    parser.add_argument(
        "--run-participant-scorer",
        action="store_true",
        help="Run participant checker/scorer. Use only when predictions cover the full reference split.",
    )
    parser.add_argument("--score-split", choices=["val", "test"], default="val")
    parser.add_argument("--reference-dir", default=str(REFERENCE_DIR))
    parser.add_argument("--notes", default="")

    # LoRA settings. DeBERTa-v3 attention projection modules use these names.
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="query_proj,key_proj,value_proj",
        help=(
            "Comma-separated LoRA target module names. Default: query_proj,key_proj,value_proj. "
            "For a broader experiment, try query_proj,key_proj,value_proj,dense."
        ),
    )
    parser.add_argument("--lora-bias", default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora-modules-to-save",
        default="classifier,pooler",
        help=(
            "Comma-separated non-LoRA modules to keep trainable/save with the adapter. "
            "Default keeps the sequence-classification head/pooler trainable."
        ),
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    print(f"Running {SCRIPT_VERSION}")
    set_seed(args.seed)
    ensure_directories()

    train_path = Path(args.train_path)
    eval_path = Path(args.eval_path) if args.eval_path else None
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = safe_filename(args.model_name)
    run_dir = Path(args.output_dir) / f"{safe_model}__{run_tag}"
    best_checkpoint_dir = run_dir / "checkpoint-best"
    final_checkpoint_dir = run_dir / "checkpoint-final"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from: {train_path}")
    all_train_examples = load_examples(train_path, limit=args.train_limit)

    if eval_path is None:
        if args.eval_split > 0:
            train_examples, eval_examples = split_examples(all_train_examples, args.eval_split, args.seed)
            split_mode = f"train_path_split_{args.eval_split:.2f}"
            final_eval_only = False
        elif args.allow_train_eval_overlap:
            train_examples = all_train_examples
            eval_examples = all_train_examples[: args.eval_limit] if args.eval_limit else all_train_examples
            split_mode = "train_eval_overlap_debug"
            final_eval_only = False
        else:
            raise ValueError(
                "eval_path is omitted and eval_split <= 0. Provide --eval-path, "
                "set --eval-split > 0, or pass --allow-train-eval-overlap for debugging."
            )
    else:
        train_examples = all_train_examples
        eval_examples = load_examples(eval_path, limit=args.eval_limit)
        eval_reference_ids = load_reference_ids(eval_path, limit=args.eval_limit)
        attach_reference_ids(eval_examples, eval_reference_ids)
        split_mode = "separate_eval_path"
        final_eval_only = bool(args.final_eval_only)

    if args.eval_limit and eval_path is None and args.eval_split > 0:
        eval_examples = eval_examples[: args.eval_limit]

    if args.run_participant_scorer and (args.eval_limit is not None or (eval_path is None and args.eval_split > 0)):
        raise ValueError(
            "Participant scorer should only be run when predictions cover the full reference split. "
            "Do not use it with --eval-limit or internal eval splits."
        )

    print(f"Split mode: {split_mode}")
    print(f"Final eval only: {final_eval_only}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Evaluation examples: {len(eval_examples)}")
    print(f"Label mode: {args.label_mode}")
    preview_examples(train_examples, n=args.preview_n)
    if eval_examples:
        print(f"First eval id: {eval_examples[0].get('id')!r}")
        print(f"Eval examples with ids: {sum(1 for ex in eval_examples if ex.get('id') is not None)}/{len(eval_examples)}")

    requested_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_amp, _use_grad_scaler = resolve_dtype(args, requested_device)
    print(f"Requested device: {requested_device}; autocast: {use_amp}; amp dtype: {amp_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = build_sequence_classifier(
        args.model_name,
        label_mode=args.label_mode,
        device=requested_device,
        args=args,
        amp_dtype=amp_dtype,
    )

    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

    target_modules = parse_variants(args.lora_target_modules)
    modules_to_save = parse_variants(args.lora_modules_to_save)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type=TaskType.SEQ_CLS,
        modules_to_save=modules_to_save or None,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(requested_device)
    model.print_trainable_parameters()

    parameter_count, trainable_parameter_count = count_trainable_parameters(model)
    print(f"Total parameter count: {parameter_count:,}")
    print(f"Trainable parameter count: {trainable_parameter_count:,}")

    train_loader = build_dataloader(
        train_examples,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.train_batch_size,
        shuffle=True,
        label_mode=args.label_mode,
    )
    eval_loader = build_dataloader(
        eval_examples,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        shuffle=False,
        label_mode=args.label_mode,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_training_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and amp_dtype == torch.float16))

    history: list[dict[str, Any]] = []
    best_metrics: EvalMetrics | None = None
    checkpoint_selection = "best_validation_checkpoint"

    print("\nStarting DeBERTa LoRA fine-tuning...")
    global_step = 0
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_seen = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, requested_device)
            model_inputs, labels, _hard_labels, _gold_probs = split_batch_for_model(batch)
            batch_size = int(labels.shape[0])

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**model_inputs)
                loss, _probs = compute_loss_and_probs(outputs, labels, args.label_mode)
                loss = loss / max(1, args.grad_accum_steps)

            scaler.scale(loss).backward()
            epoch_loss += float(loss.item()) * max(1, args.grad_accum_steps) * batch_size
            epoch_seen += batch_size

            should_step = step % args.grad_accum_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss = epoch_loss / max(1, epoch_seen)

        if final_eval_only:
            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "eval": None,
            }
            history.append(row)
            print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | eval skipped (final-eval-only)")
        else:
            eval_metrics = evaluate_model(
                model,
                eval_loader,
                requested_device,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                label_mode=args.label_mode,
            )

            row = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "eval": asdict(eval_metrics),
            }
            history.append(row)
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"eval_loss={eval_metrics.loss if eval_metrics.loss is not None else 'NA'} | "
                f"eval_acc={eval_metrics.accuracy if eval_metrics.accuracy is not None else 'NA'} | "
                f"eval_rho={eval_metrics.rho if eval_metrics.rho is not None else 'NA'}"
            )

            if is_better(eval_metrics, best_metrics, args.best_metric):
                best_metrics = eval_metrics
                best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_checkpoint_dir)
                tokenizer.save_pretrained(best_checkpoint_dir)
                print(f"Saved new best LoRA adapter checkpoint to: {best_checkpoint_dir}")

    train_end = time.perf_counter()
    total_training_runtime_seconds = train_end - train_start
    print(f"Fine-tuning complete in {total_training_runtime_seconds:.2f}s")

    if final_eval_only:
        checkpoint_selection = "final_fixed_epoch_no_eval_selection"
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_checkpoint_dir)
        tokenizer.save_pretrained(final_checkpoint_dir)
        checkpoint_dir_for_eval = final_checkpoint_dir
        best_metrics = None
        print(f"Saved final LoRA adapter checkpoint to: {final_checkpoint_dir}")
    else:
        checkpoint_dir_for_eval = best_checkpoint_dir

    if checkpoint_dir_for_eval.exists():
        if final_eval_only:
            print(
                f"\nUsing in-memory final LoRA model for evaluation; "
                f"checkpoint also saved at: {checkpoint_dir_for_eval}"
            )

            import gc

            try:
                del optimizer
            except NameError:
                pass
            try:
                del scheduler
            except NameError:
                pass
            try:
                del scaler
            except NameError:
                pass
            try:
                del train_loader
            except NameError:
                pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            model.eval()
        else:
            print(f"\nReloading LoRA adapter checkpoint: {checkpoint_dir_for_eval}")
            import gc

            del model
            try:
                del base_model
            except NameError:
                pass
            try:
                del optimizer
            except NameError:
                pass
            try:
                del scheduler
            except NameError:
                pass
            try:
                del scaler
            except NameError:
                pass

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            base_model = build_sequence_classifier(
                args.model_name,
                label_mode=args.label_mode,
                device=requested_device,
                args=args,
                amp_dtype=amp_dtype,
            )
            model = PeftModel.from_pretrained(base_model, checkpoint_dir_for_eval)
            model.to(requested_device)
            model.eval()

    warmup_path = Path(args.warmup_path) if args.warmup_path else None
    warmup_examples_used = run_warmup_examples(
        model=model,
        tokenizer=tokenizer,
        warmup_path=warmup_path,
        warmup_n=args.warmup_n,
        max_length=args.max_length,
        device=requested_device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )

    predictions, mean_latency, total_inference_runtime = predict_examples_timed(
        model=model,
        tokenizer=tokenizer,
        examples=eval_examples,
        max_length=args.max_length,
        device=requested_device,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        label_mode=args.label_mode,
        preview_n=args.preview_n,
    )
    local_prediction_metrics = compute_prediction_metrics(eval_examples, predictions)
    ensure_required_submission_ids(
        predictions=predictions,
        examples=eval_examples,
        score_split=args.score_split,
    )

    eval_file_stem = "val.model-agnostic" if eval_path is None else Path(eval_path).stem
    archive_pred_path = PRED_ARCHIVE_DIR / f"{eval_file_stem}__finetuned__lora__{args.label_mode}__{safe_model}__{run_tag}.json"
    run_pred_path = run_dir / "predictions.json"
    score_path = SCORES_DIR / f"deberta_lora_scores__{args.label_mode}__{safe_model}__{run_tag}.txt"
    metadata_path = METADATA_DIR / f"run__deberta_lora__finetuned__{args.label_mode}__{safe_model}__{run_tag}.json"
    run_metadata_path = run_dir / "metadata.json"

    for path in [archive_pred_path, run_pred_path]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to: {run_pred_path}")
    print(f"Saved archive predictions to: {archive_pred_path}")

    participant_scores: dict[str, float | str] = {}
    current_pred_path: Path | None = None
    submission_dir: Path | None = None

    if args.write_current or args.run_participant_scorer:
        current_filename = f"{args.score_split}.model-agnostic.json"
        current_pred_path = PRED_CURRENT_DIR / current_filename
        with current_pred_path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"Saved current predictions to: {current_pred_path}")

    if args.run_participant_scorer:
        submission_dir = PRED_SUBMISSIONS_DIR / f"deberta_lora__{safe_model}__{run_tag}"
        submission_dir.mkdir(parents=True, exist_ok=True)
        submission_pred_path = submission_dir / f"{args.score_split}.model-agnostic.json"
        with submission_pred_path.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"Saved isolated scorer submission to: {submission_pred_path}")

        participant_scores = run_checker_and_scorer(
            prediction_dir=submission_dir,
            score_path=score_path,
            reference_dir=Path(args.reference_dir),
            score_split=args.score_split,
        )

    loss_name = (
        "BCEWithLogitsLoss_soft_p_hallucination"
        if args.label_mode == "soft"
        else "CrossEntropyLoss_hard_label"
    )

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_type": "deberta_lora_finetuned_binary_classifier",
        "base_model_name": args.model_name,
        "safe_model_name": safe_model,
        "checkpoint_dir": str(checkpoint_dir_for_eval),
        "train_path": str(train_path),
        "eval_path": str(eval_path) if eval_path is not None else str(train_path),
        "split_mode": split_mode,
        "final_eval_only": final_eval_only,
        "checkpoint_selection": checkpoint_selection,
        "seed": args.seed,
        "num_train_examples": len(train_examples),
        "num_eval_examples": len(eval_examples),
        "label_mode": args.label_mode,
        "loss": loss_name,
        "rho_target": "p_hallucination_gold",
        "accuracy_target": "majority_hard_label",
        "input_format": "cross_encoder_context_hypothesis_pair",
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
            "bias": args.lora_bias,
            "task_type": "SEQ_CLS",
        },
        "hyperparameters": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_train_batch_size": args.train_batch_size * max(1, args.grad_accum_steps),
            "fp16": bool(args.fp16 and requested_device.type == "cuda"),
            "bf16": bool(args.bf16 and requested_device.type == "cuda"),
            "amp_dtype": str(amp_dtype),
            "gradient_checkpointing": args.gradient_checkpointing,
            "attn_implementation": args.attn_implementation,
            "best_metric": args.best_metric,
        },
        "training_history": history,
        "best_eval_metrics_during_training": asdict(best_metrics) if best_metrics is not None else None,
        "final_prediction_metrics": local_prediction_metrics,
        "participant_scores": participant_scores,
        "prediction_paths": {
            "run_predictions": str(run_pred_path),
            "archive_predictions": str(archive_pred_path),
            "current_predictions": str(current_pred_path) if current_pred_path is not None else None,
            "scorer_submission_dir": str(submission_dir) if submission_dir is not None else None,
        },
        "score_path": str(score_path) if participant_scores else None,
        "computational_cost": {
            "parameter_count": parameter_count,
            "trainable_parameter_count": trainable_parameter_count,
            "trainable_parameter_fraction": trainable_parameter_count / parameter_count if parameter_count else None,
            "total_training_runtime_seconds": total_training_runtime_seconds,
            "mean_inference_latency_seconds_per_example": mean_latency,
            "total_inference_runtime_seconds": total_inference_runtime,
        },
        "warmup": {
            "warmup_path": str(warmup_path) if warmup_path is not None else None,
            "warmup_examples_used": warmup_examples_used,
        },
        "notes": args.notes,
    }

    for path in [metadata_path, run_metadata_path]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata to: {run_metadata_path}")
    print(f"Saved metadata archive to: {metadata_path}")
    print("Done.")

if __name__ == "__main__":
    main()