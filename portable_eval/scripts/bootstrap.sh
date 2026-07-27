#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python3.10}"

"${python_bin}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"

python --version
java -version
agent-eval --help
