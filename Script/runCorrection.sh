#!/bin/bash

INPUT_FILE="NODE.txt"
PYTHON_SCRIPT="Correction.py"
MAX_PARALLEL=12

pids=()

wait_for_slot() {
    while (( ${#pids[@]} >= MAX_PARALLEL )); do
        for i in "${!pids[@]}"; do
            if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                wait "${pids[$i]}"
                status=$?

                if (( status != 0 )); then
                    echo "[WARN] A job (PID ${pids[$i]}) exited with status $status"
                fi

                unset 'pids[$i]'
            fi
        done

        pids=("${pids[@]}")
        sleep 0.5
    done
}

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "[ERROR] Input file not found: $INPUT_FILE"
    exit 1
fi

mkdir -p logs

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines.
    [[ -z "${line//[[:space:]]/}" ]] && continue

    # Skip lines beginning with #, including whitespace before #.
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    wait_for_slot

    node_name="${line%%, *}"
    node_name="$(echo "$node_name" | xargs)"
    
    if [[ -z "$node_name" ]]; then
        continue
    fi

    wait_for_slot

    log_file="logs/${node_name}.log"

    echo "Launching Session:"
    echo "  node_name = $node_name"
    echo "  log        = $log_file"

    python "$PYTHON_SCRIPT" \
        --node_name "$node_name" \
        > "$log_file" 2>&1 &

    pids+=("$!")

done < "$INPUT_FILE"

echo "All jobs launched. Waiting for completion..."

for pid in "${pids[@]}"; do
    wait "$pid"
    status=$?

    if (( status != 0 )); then
        echo "[WARN] PID $pid exited with status $status"
    fi
done

echo "All nodes finished."