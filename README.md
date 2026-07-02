# Milestone-GAE for Long-Horizon LLM Agent Training

This repository contains the code and reproduction scripts for our paper on
**Milestone-Guided Generalized Advantage Estimation (Milestone-GAE)** for
reinforcement learning of long-horizon language agents.

The project builds on `verl-agent` and `veRL`, and focuses on text-based
interactive environments where an LLM policy must complete multi-step tasks.
We compare a **GRPO baseline** with **Milestone-GAE** on ALFWorld and SciWorld
using Qwen2.5 instruction models.

## Overview

GRPO is a critic-free baseline that assigns the same outcome-level advantage to
all steps in a trajectory. This works when final task success is enough, but it
does not distinguish which intermediate actions actually move the agent closer
to success.

Milestone-GAE introduces a process-level credit signal. For each task, an LLM
generator can produce task-specific milestones, and an LLM judge evaluates each
trajectory step with a milestone potential:

```text
phi(s_t) in [0, 1]
```

The potential is then used as a value-like signal in a GAE-style recursion:

```text
delta_t = r_t - cost + gamma * phi(s_{t+1}) - phi(s_t)
A_t     = delta_t + gamma * lambda * A_{t+1}
```

This keeps the PPO actor update pipeline intact while replacing the learned
critic with an external milestone-based progress estimator.

## Main Results

`val` is the validation success rate in percent. `step_length` is the average
episode interaction length. ALFWorld reports the mean over three training seeds.

### ALFWorld

| Method | Model | val | step_length |
|---|---|---:|---:|
| GRPO baseline | Qwen2.5-3B | 53.9 | 17.36 |
| GRPO baseline | Qwen2.5-7B | 68.7 | 14.30 |
| Milestone-GAE | Qwen2.5-3B | 60.5 | 14.06 |
| Milestone-GAE | Qwen2.5-7B | 72.1 | 10.71 |

### SciWorld

| Method | Model | val | step_length |
|---|---|---:|---:|
| GRPO baseline | Qwen2.5-3B | 51.56 | 17.72 |
| GRPO baseline | Qwen2.5-7B | 66.40 | 19.52 |
| Milestone-GAE | Qwen2.5-3B | 55.39 | 12.79 |
| Milestone-GAE | Qwen2.5-7B | 74.73 | 10.66 |

## Repository Structure

| Path | Description |
|---|---|
| `examples/grpo_trainer/` | GRPO baseline scripts for ALFWorld and SciWorld |
| `examples/milestone_gae_trainer/` | Milestone-GAE scripts for ALFWorld and SciWorld |
| `examples/release_common.sh` | Shared release helper for environment setup and script overrides |
| `examples/RELEASE_REPRODUCTION.md` | Detailed reproduction guide |
| `examples/doc/grpo_algorithm.md` | GRPO implementation notes |
| `examples/doc/milestone_gae_algorithm.md` | Milestone-GAE implementation notes |
| `rlvmr/core_milestone_gae.py` | Core Milestone-GAE advantage computation |
| `rlvmr/milestone/` | Milestone generator and judge modules |
| `agent_system/environments/` | ALFWorld and SciWorld environment wrappers |

## Installation

The experiments are intended for a Linux GPU environment.

```bash
conda create -n verl-agent python=3.12 -y
conda activate verl-agent

pip install vllm==0.11.0
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

Install ALFWorld:

```bash
pip install gymnasium==0.29.1
pip install stable-baselines3==2.6.0
pip install alfworld
alfworld-download -f
```

Set the ALFWorld data path if it is not under the default cache directory:

```bash
export ALFWORLD_DATA=$HOME/.cache/alfworld
```

Install SciWorld. Java 1.8 or newer is required.

```bash
cd agent_system/environments/env_package/sciworld/ScienceWorld
pip install -e .
cd -
```

## Data Preparation

The parquet files used by the trainer are lightweight query placeholders. The
actual ALFWorld and SciWorld tasks are sampled by the environments during
rollout.

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

## Reproducing GRPO Baselines

ALFWorld:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
DATA_DIR=$PWD/data/verl-agent/text \
ALFWORLD_DATA=$HOME/.cache/alfworld \
bash examples/grpo_trainer/run_alfworld.sh
```

SciWorld:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
DATA_DIR=$PWD/data/verl-agent/text \
bash examples/grpo_trainer/run_sciworld.sh
```

For Qwen2.5-3B, keep the same script and change only `MODEL_PATH` and
`EXP_NAME`:

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
EXP_NAME=grpo_qwen2_5_3b_alfworld_ood \
ALFWORLD_DATA=$HOME/.cache/alfworld \
bash examples/grpo_trainer/run_alfworld.sh
```

## Reproducing Milestone-GAE

Milestone-GAE requires an OpenAI-compatible judge/generator service. The release
scripts default to two local endpoints:

```text
http://127.0.0.1:8080/v1
http://127.0.0.1:8081/v1
```

Example judge service launches:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --port 8080

python -m vllm.entrypoints.openai.api_server \
  --model Qwen3-VL-32B-Instruct-FP8 \
  --port 8081
```

ALFWorld:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
DATA_DIR=$PWD/data/verl-agent/text \
ALFWORLD_DATA=$HOME/.cache/alfworld \
JUDGE_LLM_URL_1=http://127.0.0.1:8080/v1 \
JUDGE_LLM_URL_2=http://127.0.0.1:8081/v1 \
bash examples/milestone_gae_trainer/run_alfworld.sh
```

SciWorld:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
DATA_DIR=$PWD/data/verl-agent/text \
JUDGE_LLM_URL_1=http://127.0.0.1:8080/v1 \
JUDGE_LLM_URL_2=http://127.0.0.1:8081/v1 \
bash examples/milestone_gae_trainer/run_sciworld.sh
```

For Qwen2.5-3B:

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
EXP_NAME=milestone_gae_qwen2_5_3b_sciworld_l1 \
bash examples/milestone_gae_trainer/run_sciworld.sh
```

## Running Qwen3-4B

To use Qwen3-4B as the actor policy, disable Qwen3 thinking in the chat
template:

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
ALFWORLD_DATA=$HOME/.cache/alfworld \
bash examples/milestone_gae_trainer/run_alfworld.sh
```

`ENABLE_THINKING=False` expands to:

```text
+data.apply_chat_template_kwargs.enable_thinking=False
```

## Important Runtime Overrides

All release scripts accept environment variables and trailing Hydra overrides.

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` | Actor policy model |
| `DATA_DIR` | `$REPO_ROOT/data/verl-agent/text` | Directory with `train.parquet` and `test.parquet` |
| `EXP_NAME` | script-specific | Experiment name |
| `TRAIN_BATCH_SIZE` | `16` | Number of query groups per training step |
| `VAL_BATCH_SIZE` | `128` or `64` | Number of validation episodes |
| `GROUP_SIZE` | `8` | Policy rollouts per training query |
| `TRAINER_LOGGER` | `['console','wandb']` | Trainer loggers |
| `NUM_GPUS_PER_NODE` | `8` | GPUs per node |
| `RAY_NUM_CPUS` | script-specific | Ray CPU budget |
| `CONDA_SETUP_SCRIPT` | unset | Optional conda bootstrap script |
| `CONDA_ENV_NAME` | unset | Optional conda environment to activate |
| `MY_TEMP_DIR` | `/tmp/verl-agent/$USER/$EXP_NAME` | Ray, vLLM, torch, and Triton cache root |

Milestone-GAE-specific variables:

| Variable | Default | Description |
|---|---|---|
| `MILESTONE_GAMMA` | `0.99` | Discount factor used by Milestone-GAE |
| `MILESTONE_LAMBDA` | `0.95` | GAE lambda |
| `MILESTONE_COST` | `0.05` | Per-step cost |
| `GENERATOR_ENABLE` | `true` | Enable dynamic milestone generation |
| `GENERATOR_NUM_MILESTONES` | `5` | Number of generated milestones |
| `JUDGE_LLM_MODEL` | `Qwen3-VL-32B-Instruct-FP8` | Judge and generator model |
| `JUDGE_LLM_URL_1` | `http://127.0.0.1:8080/v1` | First judge endpoint |
| `JUDGE_LLM_URL_2` | `http://127.0.0.1:8081/v1` | Second judge endpoint |

For low-resource smoke tests, use:

```bash
TRAIN_BATCH_SIZE=1 GROUP_SIZE=1 VAL_BATCH_SIZE=4 TRAINER_LOGGER='[console]' \
bash examples/grpo_trainer/run_sciworld.sh
```

## Evaluation Splits

- ALFWorld uses `alfworld/AlfredTWEnv` and evaluates on
  `eval_out_of_distribution`, corresponding to the `valid_unseen` split.
- SciWorld uses `sciworld/ScienceWorldEnv` with
  `env.sciworld.generalization_level=1`; training samples L1 train variations
  and validation samples L1 test variations.

## Additional Documentation

For a fuller reproduction record, see:

- `examples/RELEASE_REPRODUCTION.md`
- `examples/doc/release_readiness_review.md`
- `examples/doc/grpo_algorithm.md`
- `examples/doc/milestone_gae_algorithm.md`

## Acknowledgement

This codebase builds on `verl-agent` and `veRL`. The environments are adapted
from ALFWorld and ScienceWorld/SciWorld. We thank the authors and contributors
of these projects.

## Citation

If you use this repository, please cite the paper associated with
Milestone-GAE. The BibTeX entry will be updated when the paper metadata is
finalized.
