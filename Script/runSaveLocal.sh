#!/bin/bash

set -euo pipefail

# Run relative to the directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the environment before running this script:
# conda activate dlc-env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NODE_FILE="NODE.txt"

if [[ ! -f "$NODE_FILE" ]]; then
    echo "Error: Cannot find $SCRIPT_DIR/$NODE_FILE"
    exit 1
fi

# Read every nonempty, non-comment line from NODE.txt.
# This works with the default macOS Bash.
node_names=()

while IFS= read -r node_name || [[ -n "$node_name" ]]; do
    # Remove a possible Windows carriage return.
    node_name="${node_name%$'\r'}"

    # Remove leading and trailing whitespace.
    node_name="$(
        printf '%s' "$node_name" |
        sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
    )"

    # Ignore empty lines and comments.
    if [[ -z "$node_name" || "$node_name" == \#* ]]; then
        continue
    fi

    node_names+=("$node_name")
done < "$NODE_FILE"

if (( ${#node_names[@]} == 0 )); then
    echo "Error: NODE.txt contains no valid node names."
    exit 1
fi

echo "Host: $(hostname)"
echo "Python: $(command -v python)"
echo "Nodes (${#node_names[@]}): ${node_names[*]}"
echo "Started: $(date)"

python -u Save.py \
    --node_names "${node_names[@]}"

echo "Finished: $(date)"