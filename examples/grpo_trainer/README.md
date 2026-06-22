# GRPO Trainer Release Scripts

This directory contains release-ready GRPO baseline entry points for ALFWorld
and SciWorld.

```bash
bash examples/grpo_trainer/run_alfworld.sh
bash examples/grpo_trainer/run_sciworld.sh
```

Before running, prepare `data/verl-agent/text/train.parquet` and
`data/verl-agent/text/test.parquet` as described in
`examples/RELEASE_REPRODUCTION.md`.

Common overrides:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
DATA_DIR=/path/to/data/verl-agent/text
ALFWORLD_DATA=/path/to/alfworld
EXP_NAME=grpo_qwen2_5_7b_alfworld_ood
TRAINER_LOGGER='[console]'
```

For Qwen3-4B actor experiments:

```bash
MODEL_PATH=Qwen/Qwen3-4B \
ENABLE_THINKING=False \
bash examples/grpo_trainer/run_sciworld.sh
```
