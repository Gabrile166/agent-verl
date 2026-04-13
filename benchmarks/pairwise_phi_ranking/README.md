# Pairwise Potential Ranking Benchmark

This directory contains the core framework for pairwise potential ranking evaluation.

## Directory Layout

- `core/`: benchmark data schema, label rules, and sample filtering helpers
- `eval/`: prompt builder, model interface layer, and metric computation helpers

## Current Scope

Implemented in this stage:

- Core data models with JSON-compatible serialization (`to_dict` / `from_dict`)
- Pairwise label rules for SciWorld and ALFWorld TextWorld
- Tie filtering, difficulty assignment, and A/B randomization
- Evaluation-side prompt building and robust response parsing
- Model interface abstraction with OpenAI-compatible and manual-input backends
- Aggregate benchmark metrics for overall and subset accuracies

Not included in this stage:

- `build/` pipeline modules
- `eval/eval.py`
- `eval/baselines.py`
- `eval/judge_wrapper.py`
