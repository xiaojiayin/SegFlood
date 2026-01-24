#!/bin/bash
#SBATCH --job-name=infer_kurosiwo
#SBATCH --output=scripts/infer/infer_kurosiwo_%j.out
#SBATCH --error=scripts/infer/infer_kurosiwo_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
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
OUT_DIR="${3:-scripts/infer/output/kurosiwo}"
EVENTS="${4:-all}"
USE_DEM="${5:-false}"

if [[ -z "${CKPT_PATH}" || -z "${DATA_ROOT}" ]]; then
  echo "Usage: sbatch $0 <ckpt_path> <data_root> [out_dir] [events] [use_dem]" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

export PYTHONPATH=.

EXTRA_ARGS=()
if [[ "${USE_DEM}" == "true" ]]; then
  EXTRA_ARGS+=(--use_dem)
  # DEM scaling must match training
  EXTRA_ARGS+=(--dem_scale_mode zscore)
fi

python scripts/infer/infer_kurosiwo.py \
  --checkpoint "${CKPT_PATH}" \
  --root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --events "${EVENTS}" \
  --zscore \
  --batch_size 128 \
  --tile_size 256 \
  --overlap 64 \
  --mosaic_scale linear \
  --use_ratio \
  "${EXTRA_ARGS[@]}"

echo "Done. Output: ${OUT_DIR}"



