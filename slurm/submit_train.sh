#!/bin/bash
# Compute sweep array size from config and submit training jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SRC_ROOT="${REPO_ROOT}/src"

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
mof_source_env_if_needed "${REPO_ROOT}"
mof_resolve_config "${REPO_ROOT}"

export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
LAST_ID="$(python "${SRC_ROOT}/pipeline_config.py" --config "${CONFIG}" sweep-last-id)"

AUTO_BOOTSTRAP="${MOF_AUTO_BOOTSTRAP:-1}"

echo "Submitting training array 0-${LAST_ID} (config=${CONFIG})"
TRAIN_JOB="$(sbatch --parsable \
  --chdir="${REPO_ROOT}" \
  --array="0-${LAST_ID}" \
  "${SCRIPT_DIR}/train_multi_f_array.sbatch")"
echo "Training job: ${TRAIN_JOB}"

if [[ "${AUTO_BOOTSTRAP}" == "1" ]]; then
  BOOT_JOB="$(sbatch --parsable \
    --chdir="${REPO_ROOT}" \
    --dependency="afterok:${TRAIN_JOB}" \
    "${SCRIPT_DIR}/bootstrap.slurm")"
  echo "Bootstrap job (after training): ${BOOT_JOB}"
else
  echo "MOF_AUTO_BOOTSTRAP=0 — skipping chained bootstrap (run bash slurm/submit_bootstrap.sh later)"
fi
