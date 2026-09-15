#!/bin/bash
# Submit XAI jobs — one Slurm task per diagnosis code in config.
# Optionally chains merge_fe job when MOF_AUTO_MERGE_FE=1 (default).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
SRC_ROOT="${REPO_ROOT}/src"

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
mof_source_env_if_needed "${REPO_ROOT}"
mof_resolve_config "${REPO_ROOT}"

export PYTHONPATH="${SRC_ROOT}:${PYTHONPATH:-}"
N_CODES="$(python "${SRC_ROOT}/pipeline_config.py" --config "${CONFIG}" codes | wc -l)"
LAST_ID=$((N_CODES - 1))
AUTO_MERGE_FE="${MOF_AUTO_MERGE_FE:-1}"

echo "Submitting XAI array 0-${LAST_ID} (config=${CONFIG})"
XAI_JOB="$(sbatch --parsable \
  --chdir="${REPO_ROOT}" \
  --array="0-${LAST_ID}" \
  "${SCRIPT_DIR}/run_xai.sbatch")"
echo "XAI job: ${XAI_JOB}"

if [[ "${AUTO_MERGE_FE}" == "1" ]]; then
  MERGE_JOB="$(sbatch --parsable \
    --chdir="${REPO_ROOT}" \
    --dependency="afterok:${XAI_JOB}" \
    "${SCRIPT_DIR}/merge_fe.slurm")"
  echo "Merge f_E job (after XAI): ${MERGE_JOB}"
else
  echo "MOF_AUTO_MERGE_FE=0 — run PYTHONPATH=src python src/mof_analysis.py merge-fe after XAI"
fi
