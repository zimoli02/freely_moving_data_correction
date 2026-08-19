#!/bin/bash
## Delete all SBATCH commands if you run this locally.
#SBATCH --job-name=Save
#SBATCH --partition=day
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%A.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module reset
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dlc-env

export OMP_NUM_THREADSS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Read every nonempty, non-comment line from NODE.txt.
# Each line becomes one element in node_names.
mapfile -t node_names < <(
    awk '
        {
            sub(/\r$/, "")                  # Remove Windows carriage return
            gsub(/^[[:space:]]+/, "")       # Remove leading whitespace
            gsub(/[[:space:]]+$/, "")       # Remove trailing whitespace
        }
        NF && $1 !~ /^#/ {
            print
        }
    ' NODE.txt
)

if (( ${#node_names[@]} == 0 )); then
    echo "Error: NODE.txt contains no valid node names."
    exit 1
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(command -v python)"
echo "Nodes (${#node_names[@]}): ${node_names[*]}"
echo "Started: $(date)"

python -u Save.py \
    --node_names "${node_names[@]}"

echo "Finished: $(date)"