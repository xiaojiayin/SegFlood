#!/bin/bash
#SBATCH --job-name=infer_s1s2water
#SBATCH --output=scripts/infer/infer_s1s2water_%j.out
#SBATCH --error=scripts/infer/infer_s1s2water_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

echo "Job: ${SLURM_JOB_ID:-N/A}"
echo "Start: $(date)"

###############################################################################
# Runtime configuration (override via env vars if needed)
# - PROJECT_ROOT: repo root (default: inferred from this script location)
# - CONDA_ENV: conda env name to activate (optional; otherwise activate before sbatch)
# - HF_ENDPOINT / HF_TOKEN: Hugging Face settings (optional)
# - HF_HUB_CACHE / TORCH_HOME: caches (optional; defaults under $PROJECT_ROOT/checkpoints)
###############################################################################

# Repo root (infer from this script path)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

LOG_ROOT="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_ROOT}" || true

: "${HF_ENDPOINT:=}"
: "${HF_TOKEN:=}"
: "${HF_HUB_CACHE:=${PROJECT_ROOT}/checkpoints/.cache}"
: "${TORCH_HOME:=${PROJECT_ROOT}/checkpoints}"
export HF_ENDPOINT HF_TOKEN HF_HUB_CACHE TORCH_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" || true

# Optional conda activation
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}" || true
  fi
fi

echo "Python: $(python --version 2>&1)"

CKPT_PATH="${1:-}"
DATA_ROOT="${2:-}"
OUT_DIR="${3:-scripts/infer/output/s1s2water}"
MODAL_TYPE="${4:-dual}"
ADD_DEM="${5:-false}"
ADD_SLOPE="${6:-false}"

if [[ -z "${CKPT_PATH}" || -z "${DATA_ROOT}" ]]; then
  echo "Usage: sbatch $0 <ckpt_path> <data_root> [out_dir] [modal_type] [add_dem] [add_slope]" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

export PYTHONPATH=.

EXTRA_ARGS=()
if [[ "${ADD_DEM}" == "true" ]]; then
  EXTRA_ARGS+=(--add-dem)
fi
if [[ "${ADD_SLOPE}" == "true" ]]; then
  EXTRA_ARGS+=(--add-slope)
fi

python scripts/infer/infer_s1s2water.py \
  --checkpoint "${CKPT_PATH}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --modal-type "${MODAL_TYPE}" \
  --batch-size 128 \
  --num-workers 8 \
  "${EXTRA_ARGS[@]}"

echo "Done. Output: ${OUT_DIR}"


