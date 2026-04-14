# SHROOM SemEval 2024 Task 6: Model Size in Detection of LLM Hallucinations
This repository contains experiments for evaluating how model size affects hallucination detection performance on the **SHROOM SemEval 2024 Task 6** benchmark. The project focuses on comparing out-of-the-box and, later, fine-tuned language models on the task of identifying whether a generated sentence is semantically supported by a given context.

The current implementation includes a prompt-based **FLAN-T5 judge** that predicts:
- a binary label: `Hallucination` or `Not Hallucination`
- a soft score: `p(Hallucination)`

The project also records computational cost measurements for each model run, including:
- total parameter count
- mean inference latency per example

## Project structure (for model experiements)
```text
model_experiments/
├── data/                     # Local SHROOM datasets (ignored by git)
│   ├── SHROOM_dev-v2/
│   ├── SHROOM_test-labeled/
│   └── SHROOM_trial-v1.1/
├── outputs/                  # Generated predictions, scores, metadata
│   ├── metadata/
│   ├── predictions/
│   └── scores/
├── participant_kit/          # Official evaluation scripts
├── src/
│   ├── data.py
│   ├── models_flan.py
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

Running an experiment:
Example run with FLAN-T5 Small:
- python run_experiment.py --model-type flan --model-name google/flan-t5-small --notes "OOTB FLAN-T5 Small"

Optional warmup run:
To reduce startup-related latency effects, a short warmup phase can be run on the SHROOM trial split before the measured validation pass.
- python run_experiment.py --model-type flan --model-name google/flan-t5-small --warmup-path data/SHROOM_trial-v1.1/trial-v1.json --warmup-n 10 --notes "FLAN small with warmup"

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

## Current Progress and Notes
Currently implemented:
- FLAN-T5 prompt-based judge

Planned / in progress:
- additional model families
- fine-tuned checkpoints
- broader scaling comparisons across architectures

Notes:
- Model outputs are intended to be deterministic under fixed seeds and greedy decoding.
- Warmup examples are excluded from reported latency measurements.
- Data and generated outputs are ignored in Git; only code and supporting files are tracked.