# Reproducing GRPO and Milestone-GAE Experiments

This document is the release entry point for reproducing the GRPO baseline and
Milestone-GAE experiments on ALFWorld and SciWorld.

## Scope

The release scripts cover four main training jobs:

| Method | Environment | Script |
|---|---|---|
| GRPO baseline | ALFWorld | `examples/grpo_trainer/run_alfworld.sh` |
| GRPO baseline | SciWorld | `examples/grpo_trainer/run_sciworld.sh` |
| Milestone-GAE | ALFWorld | `examples/milestone_gae_trainer/run_alfworld.sh` |
| Milestone-GAE | SciWorld | `examples/milestone_gae_trainer/run_sciworld.sh` |

All scripts use `examples/release_common.sh` for shared Ray cleanup, cache
directories, optional conda activation, rollout engine parsing, and Qwen3 chat
template overrides.

## Data Preparation

The RL dataset files are lightweight query placeholders. The actual tasks are
sampled by the ALFWorld and SciWorld environments during rollout.

Prepare the default text parquet files:

```bash
python3 -m examples.data_preprocess.prepare \
  --mode text \
  --local_dir ./data/verl-agent \
  --train_data_size 16 \
  --val_data_size 128
```

This creates:

```text
data/verl-agent/text/train.parquet
data/verl-agent/text/test.parquet
```

ALFWorld additionally needs its environment data. Point `ALFWORLD_DATA` to the
directory that contains `json_2.1.1/train`, `json_2.1.1/valid_seen`, and
`json_2.1.1/valid_unseen`.

SciWorld uses generalization level 1 in the scripts:

```text
env.sciworld.generalization_level=1
```

Training uses the L1 train variations and validation uses the L1 test
variations from the SciWorld environment package.

## Runtime Setup

The scripts no longer hard-code private cluster paths, proxies, conda
installations, or temp directories. Configure site-specific runtime through
environment variables:

```bash
export CONDA_SETUP_SCRIPT=/path/to/conda.sh   # optional
export CONDA_ENV_NAME=verl-agent             # optional
export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export DATA_DIR=$PWD/data/verl-agent/text
export ALFWORLD_DATA=$HOME/.cache/alfworld   # ALFWorld only
```

Useful shared overrides:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` | Actor policy model |
| `DATA_DIR` | `$REPO_ROOT/data/verl-agent/text` | Directory with `train.parquet` and `test.parquet` |
| `EXP_NAME` | script-specific | Experiment name |
| `TRAIN_BATCH_SIZE` | `16` | Number of query groups per training step |
| `VAL_BATCH_SIZE` | `128` or `64` | Number of validation episodes |
| `GROUP_SIZE` | `8` | Policy rollouts per training query |
| `NUM_GPUS_PER_NODE` | `8` | GPUs per node |
| `NNODES` | `1` | Node count |
| `RAY_NUM_CPUS` | script-specific | Ray CPU budget |
| `TRAINER_LOGGER` | `['console','wandb']` | Trainer loggers |
| `MY_TEMP_DIR` | `/tmp/verl-agent/$USER/$EXP_NAME` | Ray, vLLM, Triton, and torch cache root |

Hydra overrides can still be appended at the end of each command.

## GRPO Baseline

Run ALFWorld:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
ALFWORLD_DATA=/path/to/alfworld \
bash examples/grpo_trainer/run_alfworld.sh
```

Run SciWorld:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
bash examples/grpo_trainer/run_sciworld.sh
```

The first positional argument is reserved for the rollout engine. It defaults
to `vllm`:

```bash
bash examples/grpo_trainer/run_sciworld.sh vllm trainer.logger='[console]'
```

## Milestone-GAE

Milestone-GAE requires an OpenAI-compatible judge/generator service. By default
the scripts use two local endpoints for simple load balancing:

```text
http://127.0.0.1:8080/v1
http://127.0.0.1:8081/v1
```

Example vLLM launch commands:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --port 8080

python -m vllm.entrypoints.openai.api_server \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --port 8081
```

Run ALFWorld:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
ALFWORLD_DATA=/path/to/alfworld \
JUDGE_LLM_URL_1=http://127.0.0.1:8080/v1 \
JUDGE_LLM_URL_2=http://127.0.0.1:8081/v1 \
bash examples/milestone_gae_trainer/run_alfworld.sh
```

Run SciWorld:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
JUDGE_LLM_URL_1=http://127.0.0.1:8080/v1 \
JUDGE_LLM_URL_2=http://127.0.0.1:8081/v1 \
bash examples/milestone_gae_trainer/run_sciworld.sh
```

Milestone-specific overrides:

| Variable | Default | Meaning |
|---|---|---|
| `MILESTONE_GAMMA` | `0.99` | Milestone-GAE discount factor |
| `MILESTONE_LAMBDA` | `0.95` | GAE lambda |
| `MILESTONE_COST` | `0.05` | Per-step cost |
| `GENERATOR_ENABLE` | `true` | Enable dynamic milestone generation |
| `GENERATOR_NUM_MILESTONES` | `5` | Number of generated milestones |
| `MILESTONE_TEMPLATE` | `alfworld` or `none` | Fallback template |
| `JUDGE_LLM_MODEL` | `Qwen3-VL-32B-Instruct-FP8` | Judge and generator model name |
| `JUDGE_LLM_BASE_URLS` | built from `JUDGE_LLM_URL_1/2` | Explicit Hydra list override |

## Running Qwen2.5-3B

The 3B experiments use the same scripts and hyperparameters. Change only the
actor policy model and experiment name:

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
EXP_NAME=grpo_qwen2_5_3b_alfworld_ood \
ALFWORLD_DATA=/path/to/alfworld \
bash examples/grpo_trainer/run_alfworld.sh
```

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
EXP_NAME=milestone_gae_qwen2_5_3b_sciworld_l1 \
bash examples/milestone_gae_trainer/run_sciworld.sh
```

## Running Qwen3-4B

To switch the actor policy to Qwen3-4B, set `MODEL_PATH` and disable Qwen3
thinking in the tokenizer chat template:

```bash
MODEL_PATH=Qwen/Qwen3-4B \
ENABLE_THINKING=False \
EXP_NAME=grpo_qwen3_4b_sciworld_l1 \
bash examples/grpo_trainer/run_sciworld.sh
```

```bash
MODEL_PATH=Qwen/Qwen3-4B \
ENABLE_THINKING=False \
EXP_NAME=milestone_gae_qwen3_4b_alfworld_ood \
ALFWORLD_DATA=/path/to/alfworld \
bash examples/milestone_gae_trainer/run_alfworld.sh
```

`ENABLE_THINKING=False` expands to this Hydra override:

```text
+data.apply_chat_template_kwargs.enable_thinking=False
```

If GPU memory is tight, reduce `TRAIN_BATCH_SIZE`, `GROUP_SIZE`, or
`actor_rollout_ref.rollout.gpu_memory_utilization` through Hydra overrides.

## Main Reported Results

ALFWorld uses the average over three training seeds.

| Method | Model | val | step_length |
|---|---|---:|---:|
| GRPO baseline | Qwen2.5-3B | 53.9 | 17.36 |
| GRPO baseline | Qwen2.5-7B | 68.7 | 14.30 |
| Milestone-GAE | Qwen2.5-3B | 60.5 | 14.06 |
| Milestone-GAE | Qwen2.5-7B | 72.1 | 10.71 |

SciWorld:

| Method | Model | val | step_length |
|---|---|---:|---:|
| GRPO baseline | Qwen2.5-3B | 51.56 | 17.72 |
| GRPO baseline | Qwen2.5-7B | 66.40 | 19.52 |
| Milestone-GAE | Qwen2.5-3B | 55.39 | 12.79 |
| Milestone-GAE | Qwen2.5-7B | 74.73 | 10.66 |

`val` is reported as success rate percentage. `step_length` is the average
episode interaction length.
