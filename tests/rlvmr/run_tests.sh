#!/bin/bash
# Test script for Hybrid Advantage Computation
# Run this from the agent-verl root directory

set -e

echo "=============================================="
echo "Running Hybrid GRPO Unit Tests"
echo "=============================================="

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Run the core_hybrid tests
echo ">>> Testing core_hybrid.py (Advantage Computation)"
python -m pytest tests/rlvmr/test_core_hybrid.py -v --tb=short 2>/dev/null || python tests/rlvmr/test_core_hybrid.py

echo ""
echo "=============================================="
echo "All tests completed!"
echo "=============================================="
