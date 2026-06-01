# SHROOM SemEval 2024 Task 6: Model Size in Hallucination Detection

This repository contains scripts and notebooks for studying how model size affects hallucination detection on the SHROOM SemEval 2024 Task 6 benchmark. The experiments compare out-of-the-box inference and LoRA fine-tuning for FLAN-T5, DeBERTa, Qwen, and Gemma model families.

Each prediction contains:

- `label`: `Hallucination` or `Not Hallucination`
- `p(Hallucination)`: a soft hallucination score

Generated datasets, checkpoints, predictions, scores, plots, and virtual environments are intentionally ignored by git. The README describes the files that are part of the current repository, plus the local ignored folders those scripts expect.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   `-- README.md
|-- EDA/
|   `-- Exploratory Data Analysis.ipynb
|-- model_experiments_colab/
|   |-- colab_vscode_*_runner.ipynb
|   |-- run_experiment.py
|   |-- run_experiment_ootb_eval.py
|   |-- finetune_*_lora*.py
|   |-- participant_kit/
|   |   |-- check_output.py
|   |   `-- score.py
|   `-- src/
|       |-- data.py
|       |-- models_deberta.py
|       |-- models_flan.py
|       |-- models_gemma.py
|       |-- models_qwen.py
|       `-- prompts.py
|-- test_set_scores_OOTB - A100 GPU/
|-- test_set_scores_finetuned - A100 GPU/
`-- test_set_scores_all - A100 GPU/
```

Folder contents:

- `data/`: tracked placeholder only. Put local SHROOM data here if useful for manual work; dataset files are ignored.
- `EDA/`: exploratory notebook for inspecting the SHROOM data.
- `model_experiments_colab/`: main experiment code. It contains Colab/VS Code runner notebooks, out-of-the-box evaluation scripts, LoRA fine-tuning scripts, model wrappers, prompt/data helpers, and the SHROOM participant-kit scoring scripts.
- `model_experiments_colab/src/`: shared Python helpers used by the experiment runners.
- `model_experiments_colab/participant_kit/`: official-style output validation and scoring scripts.
- `test_set_scores_OOTB - A100 GPU/`: summary and plotting scripts for out-of-the-box A100 test-set runs.
- `test_set_scores_finetuned - A100 GPU/`: summary and plotting scripts for fine-tuned A100 test-set runs.
- `test_set_scores_all - A100 GPU/`: combined summary, plotting, confusion-matrix, and OOTB-vs-fine-tuned comparison scripts.

The following local folders are expected during experiments but are ignored by git: `data/`, `model_experiments_colab/data/`, `model_experiments_colab/outputs/`, and matching `data/` / `outputs/` folders inside the `test_set_scores_* - A100 GPU/` directories.

## Setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU-backed runs are strongly recommended for the larger models and fine-tuning scripts. Some Hugging Face models, such as Gemma, may require accepting model terms and authenticating with Hugging Face before running.

## Data Layout

The active experiment scripts use paths relative to `model_experiments_colab/`. Place the SHROOM files locally as:

```text
model_experiments_colab/
`-- data/
    |-- SHROOM_dev-v2/
    |   `-- val.model-agnostic.json
    |-- SHROOM_test-labeled/
    |   `-- test.model-agnostic.json
    `-- SHROOM_trial-v1.1/
        `-- trial-v1.json
```

These files are ignored by git. The same dataset layout is useful inside the `test_set_scores_* - A100 GPU/` folders when running their local analysis scripts.

## Running Experiments

Run experiment scripts from `model_experiments_colab/` so their relative paths resolve correctly:

```powershell
cd model_experiments_colab
```

Out-of-the-box validation example:

```powershell
python run_experiment.py `
  --model-type flan `
  --model-name google/flan-t5-small `
  --notes "OOTB FLAN-T5 small"
```

Out-of-the-box test-set example:

```powershell
python run_experiment_ootb_eval.py `
  --model-type flan `
  --model-name google/flan-t5-small `
  --input-path data/SHROOM_test-labeled/test.model-agnostic.json `
  --score-split test
```

Supported `--model-type` values are `flan`, `deberta`, `qwen`, and `gemma`. Outputs are written under ignored local folders:

```text
model_experiments_colab/outputs/
|-- metadata/
|-- predictions/
|-- scores/
`-- finetuned/
```

## Fine-Tuning

Use the LoRA scripts in `model_experiments_colab/` for family-specific fine-tuning:

- `finetune_deberta_lora_v2.py`
- `finetune_flan_lora.py`
- `finetune_gemma_lora_v2.py`
- `finetune_qwen_lora.py`

Smoke-test example:

```powershell
python finetune_deberta_lora_v2.py `
  --model-name MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli `
  --train-limit 32 `
  --eval-limit 32 `
  --epochs 1 `
  --notes "smoke test"
```

For final test-set evaluation, use the corresponding `*_final_eval*.py` script for the model family. Add `--run-participant-scorer` only when the prediction file covers the full reference split.

## Notebooks

The `model_experiments_colab/colab_vscode_*_runner.ipynb` notebooks are convenience launchers for the same Python scripts. Use them when running in Colab or an interactive GPU notebook environment; keep data and generated outputs in the same ignored folder layout described above.

## Summaries and Plots

The `test_set_scores_* - A100 GPU/` folders contain post-processing scripts. Run them from inside the relevant folder after placing or generating local ignored `outputs/metadata/`, `outputs/predictions/`, and any needed `data/` files.

Examples:

```powershell
cd "test_set_scores_all - A100 GPU"
python plot_metadata_results_regression_finetuned_compatible.py
python summarize_ootb_vs_finetuned_comparisons.py
python summarize_confusion_matrices.py
```

For the OOTB-only and fine-tuned-only folders, run:

```powershell
python summarize_metadata_finetuned_compatible.py
python plot_metadata_results_regression_finetuned_compatible.py
```

The summary scripts read metadata from `outputs/metadata/`. The plot scripts write figures to `outputs/plots/`. The combined comparison and confusion-matrix scripts write tables to `outputs/tables/`.

## Scoring Utilities

The participant-kit scripts can be run directly from `model_experiments_colab/`:

```powershell
python participant_kit/check_output.py outputs/predictions/current --is_val
python participant_kit/score.py outputs/predictions/current data/SHROOM_dev-v2 outputs/scores/example.txt --is_val
```

For test-set scoring, omit `--is_val` and point the reference path at `data/SHROOM_test-labeled`.
