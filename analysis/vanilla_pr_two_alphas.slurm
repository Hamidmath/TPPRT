#!/bin/bash
#SBATCH --job-name=vpr_2alpha
#SBATCH --clusters=granite
#SBATCH --account=phillipsj
#SBATCH --partition=granite-guest
#SBATCH --qos=granite-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "$HOME/tppr/sept15_sweep"

echo "=== job $SLURM_JOB_ID on $(hostname), $(date) ==="
module load python/3.11.7
source "$HOME/tppr/venv/bin/activate"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TPPR_INPUTS="$HOME/tppr/sept15_sweep/inputs"
export TPPR_RESULTS="$HOME/tppr/sept15_sweep/results"

time python -u "$HOME/tppr/sept15_sweep/code/vanilla_pr_two_alphas.py"
echo "ended: $(date)"
