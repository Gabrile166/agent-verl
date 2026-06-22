# Milestone-GAE Trainer Release Scripts

This directory contains release-ready Milestone-GAE entry points for ALFWorld
and SciWorld.

```bash
bash examples/milestone_gae_trainer/run_alfworld.sh
bash examples/milestone_gae_trainer/run_sciworld.sh
```

Milestone-GAE uses `algorithm.adv_estimator=milestone_gae` and estimates
step-level milestone potentials with an OpenAI-compatible judge/generator
service.

## Judge Service

By default the scripts expect two local judge endpoints:

```text
http://127.0.0.1:8080/v1
http://127.0.0.1:8081/v1
```

Override them if your service uses different URLs:

```bash
JUDGE_LLM_URL_1=http://host-a:8080/v1 \
JUDGE_LLM_URL_2=http://host-b:8080/v1 \
bash examples/milestone_gae_trainer/run_sciworld.sh
```

## Main Configuration

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `Qwen/Qwen2.5-7B-Instruct` | Actor policy model |
| `MILESTONE_GAMMA` | `0.99` | Milestone-GAE discount factor |
| `MILESTONE_LAMBDA` | `0.95` | GAE lambda |
| `MILESTONE_COST` | `0.05` | Per-step cost |
| `GENERATOR_ENABLE` | `true` | Enable dynamic milestone generation |
| `GENERATOR_NUM_MILESTONES` | `5` | Number of generated milestones |
| `JUDGE_LLM_MODEL` | `Qwen3-VL-32B-Instruct-FP8` | Judge and generator model |
| `MILESTONE_TEMPLATE` | `alfworld` or `none` | Fallback milestone template |

For Qwen3-4B actor experiments:

```bash
MODEL_PATH=Qwen/Qwen3-4B \
ENABLE_THINKING=False \
bash examples/milestone_gae_trainer/run_alfworld.sh
```

See `examples/RELEASE_REPRODUCTION.md` for data preparation, full commands,
and reported experiment results.
