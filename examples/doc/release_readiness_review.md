# Release Readiness Review

This review covers the GRPO baseline and Milestone-GAE experiment entry points
intended for the open-source release branch.

## Reviewed Entry Points

| Area | Files |
|---|---|
| GRPO baseline | `examples/grpo_trainer/run_alfworld.sh`, `examples/grpo_trainer/run_sciworld.sh` |
| Milestone-GAE | `examples/milestone_gae_trainer/run_alfworld.sh`, `examples/milestone_gae_trainer/run_sciworld.sh` |
| Shared release helpers | `examples/release_common.sh` |
| Release docs | `examples/RELEASE_REPRODUCTION.md`, trainer README files |

## Issues Found And Fixed

| Severity | Finding | Fix |
|---|---|---|
| High | Training scripts hard-coded private cluster paths, conda paths, temp cache roots, and proxy URLs. | Replaced with environment variables such as `MODEL_PATH`, `DATA_DIR`, `ALFWORLD_DATA`, `CONDA_SETUP_SCRIPT`, `CONDA_ENV_NAME`, and `MY_TEMP_DIR`. |
| High | The first positional engine argument was also forwarded to Hydra via `$@`, which could make `vllm` appear as an invalid Hydra override. | Added `release_parse_engine` so the optional engine argument is consumed before forwarding remaining overrides. |
| Medium | Four scripts duplicated Ray cleanup and cache setup logic. | Added `examples/release_common.sh` and sourced it from all four release scripts. |
| Medium | Milestone-GAE README had stale defaults, including `MILESTONE_COST=0.01` and a single judge URL. | Rewrote the README to match current scripts: `MILESTONE_COST=0.05`, two judge URLs, and Qwen3-4B usage. |
| Medium | Shell scripts could be converted to CRLF on Windows, making release scripts fragile on Linux. | Added `.gitattributes` to keep `*.sh` files on LF line endings. |
| Low | Internal progress docs contained private machine paths. | Replaced the SciWorld handoff path with `/path/to/agent-verl` and converted one absolute benchmark link to a repository-relative path. |

## Current Release Interface

All four scripts now support the same pattern:

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
DATA_DIR=/path/to/data/verl-agent/text \
bash examples/grpo_trainer/run_sciworld.sh [vllm] [hydra overrides...]
```

Milestone-GAE additionally supports:

```bash
JUDGE_LLM_URL_1=http://127.0.0.1:8080/v1
JUDGE_LLM_URL_2=http://127.0.0.1:8081/v1
JUDGE_LLM_MODEL=Qwen3-VL-32B-Instruct-FP8
```

Qwen3-4B actor experiments can be launched with:

```bash
MODEL_PATH=Qwen/Qwen3-4B ENABLE_THINKING=False bash examples/grpo_trainer/run_sciworld.sh
```

## Remaining Risks Before Public Release

| Severity | Risk | Recommendation |
|---|---|---|
| High | End-to-end training was not run in this review because it requires GPUs, ALFWorld data, SciWorld Java runtime, and judge services. | Run one smoke job per script with `TRAIN_BATCH_SIZE=1`, `GROUP_SIZE=1`, and `TRAINER_LOGGER='[console]'`. |
| Medium | `examples/doc/grpo_milestone_gae_experiment_details.md` appears as mojibake in this Windows PowerShell session. It may be a terminal decoding issue, but it should be checked before publication. | Open it in a UTF-8 editor and rewrite or remove it if the file itself is corrupted. |
| Medium | Several long benchmark planning documents are useful internally but are not minimal reproduction docs. | Removed from the release branch to keep the package focused on paper reproduction. |
| Medium | SciWorld Milestone-GAE can launch many JVM processes when `TRAIN_BATCH_SIZE` and `GROUP_SIZE` are large. | Document recommended smoke-test settings and cluster resource requirements. |
| Low | Non-paper environments and legacy example scripts were outside the requested ALFWorld/SciWorld scope. | Removed from the release branch to reduce package size and avoid misleading entry points. |

## Validation Performed

```bash
bash -n examples/release_common.sh \
  examples/grpo_trainer/run_alfworld.sh \
  examples/grpo_trainer/run_sciworld.sh \
  examples/milestone_gae_trainer/run_alfworld.sh \
  examples/milestone_gae_trainer/run_sciworld.sh

rg -n "<private host, user, cluster, cache, or absolute workspace path patterns>" \
  examples/grpo_trainer examples/milestone_gae_trainer examples/release_common.sh examples/doc

git diff --check
```

The shell syntax check passed. The private-path scan no longer reports private
paths in release training scripts.
