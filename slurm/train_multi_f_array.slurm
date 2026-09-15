#!/bin/bash
# Submit from repository root:
#   bash slurm/submit_train.sh
#   sbatch --chdir="$PWD" --array=0 slurm/train_multi_f_array.sbatch
#
# Array size is computed from the YAML config (codes × arch × lr × dropout).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SRC_ROOT="${REPO_ROOT}/src"

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
mof_source_env_if_needed "${REPO_ROOT}"
mof_resolve_config "${REPO_ROOT}"

MODELS_PARENT="${MOF_MODELS_PARENT:-${SRC_ROOT}}"

VENV="${MOF_VENV:-${HOME}/myenv}"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
fi

if [[ -n "${MOF_DATA_ROOT:-}" ]]; then
  DATA_ROOT="${MOF_DATA_ROOT}"
else
  DATA_ROOT="$(python "${SRC_ROOT}/pipeline_config.py" --config "${CONFIG}" print \
    | python -c "import json,sys; print(json.load(sys.stdin)['output_root'])")"
fi

SWEEP_ROOT="${DATA_ROOT}/hyper_sweep"

mkdir -p "${REPO_ROOT}/slurm_logs" "${DATA_ROOT}/logs"
cd "${DATA_ROOT}"

echo "Job ID=${SLURM_JOB_ID} Task=${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname) Date: $(date)"
echo "config=${CONFIG} data-root=${DATA_ROOT}"

module load cuda12.6/toolkit/12.6.3 2>/dev/null || true

export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONPATH="${MODELS_PARENT}:${SRC_ROOT}:${PYTHONPATH:-}"

python "${SRC_ROOT}/train_multi_f_array.py" \
  --config     "${CONFIG}" \
  --data-root  "${DATA_ROOT}" \
  --trial5-exp "${MODELS_PARENT}" \
  --job-root   "${SWEEP_ROOT}" \
  --task-id    "${SLURM_ARRAY_TASK_ID}"

echo "Finished task ${SLURM_ARRAY_TASK_ID} at $(date)"
