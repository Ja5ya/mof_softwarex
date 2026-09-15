#!/bin/bash
#SBATCH --job-name=mof_train
#SBATCH --partition=a100q
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_logs/train_%A_%a.out
#SBATCH --error=slurm_logs/train_%A_%a.err
#SBATCH --array=0-119

# Array range and chdir are set by slurm/submit_train.sh.
# Do not resolve paths from BASH_SOURCE here — Slurm copies this script to a
# spool directory, so a sibling .slurm file is not available at job runtime.

bash "${SLURM_SUBMIT_DIR}/slurm/train_multi_f_array.slurm"
