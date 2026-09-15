#!/bin/bash
# Submit bootstrap job (stage 4). Usually auto-chained by submit_train.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
mof_source_env_if_needed "${REPO_ROOT}"
mof_resolve_config "${REPO_ROOT}"

echo "Submitting bootstrap job (config=${CONFIG})"
sbatch --chdir="${REPO_ROOT}" "${SCRIPT_DIR}/bootstrap.slurm"
