#!/bin/bash
# Merge per-code sweep_fe_<CODE>.csv into sweep_fe_summary.csv (after XAI).
#
#SBATCH --job-name=mof_merge_fe
#SBATCH --partition=milanq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/merge_fe_%j.out
#SBATCH --error=slurm_logs/merge_fe_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:?submit with: sbatch --chdir=\$REPO_ROOT slurm/merge_fe.slurm}"
SRC_ROOT="${REPO_ROOT}/src"

# shellcheck source=common.sh
source "${REPO_ROOT}/slurm/common.sh"
mof_source_env_if_needed "${REPO_ROOT}"
mof_resolve_config "${REPO_ROOT}"

mkdir -p "${REPO_ROOT}/slurm_logs"

VENV="${MOF_VENV:-${HOME}/myenv}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"

python "${SRC_ROOT}/mof_analysis.py" --config "${CONFIG}" merge-fe
echo "Finished merge-fe at $(date)"
