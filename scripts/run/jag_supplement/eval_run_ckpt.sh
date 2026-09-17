#!/bin/bash
#SBATCH --job-name=eval_run_ckpt
#SBATCH --output=scripts/run/jag_supplement/logs/ckpt_eval/%x_%j.out
#SBATCH --error=scripts/run/jag_supplement/logs/ckpt_eval/%x_%j.err
#SBATCH --partition=a01,h01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

# Test-set evaluation of one checkpoint using the *exact* Hydra config the run
# was trained with (<run_dir>/.hydra/config.yaml), so no architecture overrides
# have to be reconstructed by hand.
#
#   sbatch eval_run_ckpt.sh <RUN_DIR> <CKPT_FILE> [EXTRA_OVERRIDES...]
#   e.g. eval_run_ckpt.sh logs/foo_2026-08-31_11-17-03 last.ckpt

set -euo pipefail

RUN_DIR="${1:?run dir}"
CKPT_FILE="${2:?ckpt file name}"
shift 2

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_ROOT}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
RUN_BASE="$(basename "${RUN_DIR}")"
RUN_NAME="${RUN_BASE%_20[0-9][0-9]-*}"
TAG="${CKPT_FILE%.ckpt}"
CKPT="${RUN_DIR}/checkpoints/${CKPT_FILE}"
[[ -f "${CKPT}" ]] || { echo "missing ${CKPT}" >&2; exit 2; }
[[ -f "${RUN_DIR}/.hydra/config.yaml" ]] || { echo "missing ${RUN_DIR}/.hydra/config.yaml" >&2; exit 2; }

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
export HF_HUB_OFFLINE=1
source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood

echo "RUN_DIR=${RUN_DIR}"
echo "CKPT=${CKPT}"

python src/eval.py \
  --config-path "${RUN_DIR}/.hydra" --config-name config \
  "ckpt_path=${CKPT}" \
  "experiment_name=jag26_ckpteval_${RUN_NAME}_${TAG}" \
  "+run_dir_evaluated=${RUN_BASE}" \
  "$@"
