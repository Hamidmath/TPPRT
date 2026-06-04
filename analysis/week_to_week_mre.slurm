#!/bin/bash
#SBATCH --job-name=week_to_week_mre
#SBATCH --clusters=granite
#SBATCH --account=phillipsj
#SBATCH --partition=coe-class-grn
#SBATCH --qos=coe-class-grn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
cd "$HOME/tppr/sept15_sweep"
echo "=== job $SLURM_JOB_ID on $(hostname), $(date) ==="
module load python/3.11.7
source "$HOME/tppr/venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TPPR_INPUTS="$HOME/tppr/sept15_sweep/inputs"
export TPPR_RESULTS="$HOME/tppr/sept15_sweep/results"
time python -u "$HOME/tppr/sept15_sweep/code/week_to_week_mre.py"
echo "ended: $(date)"
