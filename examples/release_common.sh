#!/usr/bin/env bash
# Shared utilities for release-ready experiment scripts.

set -euo pipefail

RELEASE_EXAMPLES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${RELEASE_EXAMPLES_DIR}/.." && pwd)"

release_parse_engine() {
    ENGINE="${ENGINE:-vllm}"
    if [[ $# -gt 0 && "$1" != *=* && "$1" != +* ]]; then
        ENGINE="$1"
        shift
    fi
    REMAINING_ARGS=("$@")
}

release_activate_conda() {
    if [[ -n "${CONDA_SETUP_SCRIPT:-}" ]]; then
        # Optional site-specific conda bootstrap, for example /opt/conda/etc/profile.d/conda.sh.
        # shellcheck disable=SC1090
        source "${CONDA_SETUP_SCRIPT}"
    fi

    if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
        conda activate "${CONDA_ENV_NAME}"
    fi
}

release_prepare_runtime() {
    local exp_name="$1"

    export RAY_worker_register_timeout_seconds="${RAY_worker_register_timeout_seconds:-600}"
    export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
    export no_proxy="${no_proxy:-${NO_PROXY:-localhost,127.0.0.1,0.0.0.0}}"

    if command -v ray >/dev/null 2>&1; then
        echo "Cleaning up existing Ray processes..."
        ray stop --force 2>/dev/null || true
        sleep 2
    fi

    local default_temp_root="${TMPDIR:-/tmp}/verl-agent/${USER:-user}/${exp_name}"
    export MY_TEMP_DIR="${MY_TEMP_DIR:-${default_temp_root}}"
    export RAY_TMPDIR="${RAY_TMPDIR:-${MY_TEMP_DIR}/ray}"
    export TORCH_COMPILE_CACHE_DIR="${TORCH_COMPILE_CACHE_DIR:-${MY_TEMP_DIR}/torch_compile_cache}"
    export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-${MY_TEMP_DIR}/vllm_cache}"
    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${MY_TEMP_DIR}/triton_cache}"

    mkdir -p "${RAY_TMPDIR}" "${TORCH_COMPILE_CACHE_DIR}" "${VLLM_CACHE_DIR}" "${TRITON_CACHE_DIR}"
}

release_chat_template_args() {
    CHAT_TEMPLATE_ARGS=()
    if [[ -n "${ENABLE_THINKING:-}" ]]; then
        CHAT_TEMPLATE_ARGS+=("+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}")
    fi
}
