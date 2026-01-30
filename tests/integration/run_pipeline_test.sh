#!/bin/bash
# Integration Test Script for Agent-Verl Pipeline
# Based on run_alfworld.sh provided by user

# Cleanup existing Ray processes
echo "Cleaning up existing Ray processes..."
ray stop --force 2>/dev/null || true
sleep 2

# Network & Proxy Setup
export http_proxy="http://10.70.11.190:8412"
export https_proxy="http://10.70.11.190:8412"
export no_proxy="localhost,127.0.0.1,0.0.0.0"

# Directories & Cache Setup
export MY_TEMP_DIR="/workdir/temp_cache/${USER}/test_pipeline"
mkdir -p $MY_TEMP_DIR

export RAY_TMPDIR="${MY_TEMP_DIR}/ray"
mkdir -p $RAY_TMPDIR

export TORCH_COMPILE_CACHE_DIR="${MY_TEMP_DIR}/torch_compile_cache"
export VLLM_CACHE_DIR="${MY_TEMP_DIR}/vllm_cache"
export TRITON_CACHE_DIR="${MY_TEMP_DIR}/triton_cache"
mkdir -p $TORCH_COMPILE_CACHE_DIR $VLLM_CACHE_DIR $TRITON_CACHE_DIR

# Env Vars
export VLLM_ATTENTION_BACKEND=XFORMERS
export ALFWORLD_DATA=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/.cache/alfworld
# Set PYTHONPATH to include the current directory so modules can be imported
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Conda Setup
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/conda3/anaconda3/setupconda.sh
conda activate rlvmr-alfworld
export PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/conda3/anaconda3/envs/rlvmr-alfworld/bin:/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm-hl/hadoop-mlm/tangjixin/conda3/anaconda3/condabin:$PATH"

# Ray Config
export RAY_DEDUP_LOGS=0
# Ensure we limit CPUs for this test script to avoid overwhelming the node if running on head
export RAY_ADDRESS='local'

echo "=================================================="
echo "Starting Integration Test: test_full_pipeline_debug.py"
echo "=================================================="

# Run the test script
# Assumes the script is run from the project root, e.g., ./tests/integration/run_pipeline_test.sh
# If run from tests/integration, we strictly point to the python file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Script Dir: $SCRIPT_DIR"
echo "Project Root: $PROJECT_ROOT"

# Ensure we are in project root for imports to work nicely if relying on CWD
cd $PROJECT_ROOT

python tests/integration/test_full_pipeline_debug.py
