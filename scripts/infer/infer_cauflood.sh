#!/bin/bash
#SBATCH --job-name=infer_cauflood
#SBATCH --output=scripts/infer/infer_cauflood_%j.out
#SBATCH --error=scripts/infer/infer_cauflood_%j.err
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

CKPT="${1:-}"
DATA_ROOT="${2:-}"
OUT_DIR="${3:-scripts/infer/output/cauflood}"
BS="${4:-128}"
NW="${5:-8}"

if [[ -z "${CKPT}" || -z "${DATA_ROOT}" ]]; then
  echo "Usage: sbatch $0 <ckpt_path> <dataset_path> [out_dir] [batch_size] [num_workers]" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

python scripts/infer/infer_cauflood.py \
  --checkpoint "$CKPT" \
  --dataset-path "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  --batch-size "$BS" \
  --num-workers "$NW" \
  --save-predictions \
  --save-format auto

echo "Done. Output: $OUT_DIR"


