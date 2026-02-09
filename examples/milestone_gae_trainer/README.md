# Milestone-Guided GAE Trainer

This example demonstrates how to use the Milestone-Guided GAE advantage estimator for training agents on ALFWorld environment.

## Key Features

- **LLM-as-Critic**: Uses LLM Judge to estimate milestone potentials Φ(s) instead of training a Critic network
- **Structured Milestone Judgment**: Transforms open-ended scoring into structured classification
- **Per-Trajectory Judge Call**: Calls Judge once per trajectory for efficiency
- **Global Advantage Normalization**: Step-level normalization across all trajectories

## Prerequisites

1. **Judge LLM Service**: Start a local LLM service that serves the Judge model
   ```bash
   # Example using vLLM
   python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-7B-Instruct \
       --port 8000
   ```

2. **Data Preparation**: The script will automatically prepare training data

## Usage

```bash
# Basic usage (uses default vLLM engine)
./run_alfworld.sh

# Specify Judge LLM URL
JUDGE_LLM_URL="http://localhost:8000/v1" ./run_alfworld.sh

# Use custom model
MODEL_PATH="your/model/path" ./run_alfworld.sh
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MILESTONE_GAMMA` | 0.99 | GAE discount factor |
| `MILESTONE_LAMBDA` | 0.95 | GAE lambda |
| `MILESTONE_COST` | 0.01 | Time cost per step |
| `JUDGE_LLM_URL` | http://127.0.0.1:8000/v1 | Judge LLM API URL |
| `JUDGE_LLM_MODEL` | qwen2.5-7b-instruct | Judge model name |
| `MILESTONE_TEMPLATE` | alfworld | Milestone template to use |

## Milestone Templates

Templates are stored in `rlvmr/milestone/templates/`. Currently supported:
- `alfworld.json` - ALFWorld household tasks

## Comparison with Other Methods

| Method | Advantage Estimation | Value Function |
|--------|---------------------|----------------|
| GRPO | Episode-level | None |
| GAE | Token-level | Critic Network |
| **MilestoneGAE** | Step-level | LLM Judge |
