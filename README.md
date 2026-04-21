# SHROOM SemEval 2024 Task 6: Model Size in Detection of LLM Hallucinations
This repository contains experiments for evaluating how model size affects hallucination detection performance on the **SHROOM SemEval 2024 Task 6** benchmark. The project focuses on comparing out-of-the-box and, later, fine-tuned language models on the task of identifying whether a generated sentence is semantically supported by a given context.

For each record, the current implementation predicts:
- a binary label: `Hallucination` or `Not Hallucination`
- a soft score: `p(Hallucination)`

The project also records computational cost measurements for each model run, including:
- total parameter count
- mean inference latency per example

## Project structure (for Model Experiements)
```text
model_experiments/
├── data/                     # Local SHROOM datasets (ignored by git)
│   ├── SHROOM_dev-v2/
│   ├── SHROOM_test-labeled/
│   └── SHROOM_trial-v1.1/
├── outputs/                  # Generated predictions, scores, metadata, plots
│   ├── metadata/
|   ├── plots/
│   ├── predictions/
│   └── scores/
├── participant_kit/          # Official evaluation scripts
├── src/
│   ├── data.py
│   ├── models_deberta.py
│   ├── models_flan.py
│   ├── models_gemma.py
│   ├── models_qwen.py
│   └── prompts.py
├── run_experiment.py         # Main experiment runner
├── requirements.txt
└── README.md
```

## Setup and Execution
Create and activate a virtual environment, then install dependencies:
- python -m venv .venv
- .venv\Scripts\Activate.ps1
- python -m pip install -r requirements.txt

Place the SHROOM data folders inside:
- model_experiments/data/

Expected structure:
- model_experiments/data/SHROOM_dev-v2/
- model_experiments/data/SHROOM_test-labeled/
- model_experiments/data/SHROOM_trial-v1.1/

Running an experiment (w/ warmup, to reduce startup-related latency effects):
Example run with FLAN-T5 Small:
- python run_experiment.py --model-type flan --model-name google/flan-t5-small --notes "OOTB FLAN-T5 Small"

Outputs:
Each run produces:
- predictions: JSON file with model predictions
- scores: task metrics from the participant kit
- metadata: run configuration and computational cost information
Generated files are written under:
- outputs/predictions/
- outputs/scores/
- outputs/metadata/

Run python summarize_metadata.py to see table of latest model results
Run python plot_metadata_results.py to generate plots of model data
Run python score_by_task.py to see performance on task-based (PG, MT, DM) level
- CLI example:
python .\score_by_task.py `
  .\outputs\predictions\archive\val.model-agnostic__Qwen__Qwen2.5-1.5B-Instruct.json `
  .\data\SHROOM_dev-v2\val.model-agnostic.json `
  --output .\outputs\scores\task_scores__Qwen__Qwen2.5-1.5B-Instruct.txt
Run python confusion_by_task.py to see summarized predicted class proportions and confusion matrix
- CLI example:
python .\confusion_by_task.py `
  .\outputs\predictions\archive\val.model-agnostic__Qwen__Qwen2.5-3B-Instruct.json `
  .\data\SHROOM_dev-v2\val.model-agnostic.json `
  --output .\outputs\scores\confusion__Qwen__Qwen2.5-3B-Instruct.txt

## Current Progress and Notes
Currently implemented:
- FLAN-T5 prompt-based judges
- DeBERTa-v3 variants encoder-only models
- Qwen2.5 and Gemma 3 decoder-only instruction-tuned LLM judges

Planned / in progress:
- Fine-tuned checkpoints

Notes:
- Model outputs are intended to be deterministic under fixed seeds and greedy decoding.
- Warmup examples are excluded from reported latency measurements.
- Data and generated outputs are ignored in Git; only code and supporting files are tracked.