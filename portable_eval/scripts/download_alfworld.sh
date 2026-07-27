#!/usr/bin/env bash
set -euo pipefail

data_dir="${1:-${ALFWORLD_DATA:-$HOME/.cache/alfworld}}"
export ALFWORLD_DATA="${data_dir}"

echo "Downloading ALFWorld data into ${ALFWORLD_DATA}"
alfworld-download
test -d "${ALFWORLD_DATA}/json_2.1.1/valid_unseen"
echo "ALFWorld OOD data is ready."
