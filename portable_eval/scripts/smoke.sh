#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-results/smoke}"

agent-eval preflight
agent-eval alfworld \
  --output "${output_root}/alfworld_ood" \
  --limit 2 \
  --max-steps 50 \
  --history-steps 10 \
  --resume
agent-eval sciworld \
  --output "${output_root}/sciworld_l1" \
  --limit 2 \
  --max-steps 50 \
  --history-steps 10 \
  --resume
