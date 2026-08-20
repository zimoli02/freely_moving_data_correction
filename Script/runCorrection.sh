#!/bin/bash
## delete all SBATCH command if you run this locally (not on cluster)
#SBATCH --job-name=Correction
#SBATCH --partition=day
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dlc-env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" NODE.txt)

# Select the nth nonempty, non-comment line.
line=$(
    awk -v n="$SLURM_ARRAY_TASK_ID" '
        NF && $1 !~ /^#/ {
            count++
            if (count == n) {
                print
                exit
            }
        }
    ' NODE.txt
)

# Remove a possible Windows carriage return.
line="${line%$'\r'}"

if [[ -z "${line//[[:space:]]/}" ]]; then
    echo "Input line ${SLURM_ARRAY_TASK_ID} is empty"
    exit 1
fi

# The input line is the node name.
node_name="$(echo "$line" | xargs)"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Python: $(command -v python)"
echo "node: $node_name"

echo "Started: $(date)"

python -u Correction.py \
    --node_name "$node_name"

echo "Finished: $(date)"