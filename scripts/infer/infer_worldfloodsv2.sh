#!/bin/bash
#SBATCH --job-name=infer_wf2
#SBATCH --output=scripts/infer/infer_worldfloodsv2_%j.out
#SBATCH --error=scripts/infer/infer_worldfloodsv2_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

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

# Disable TorchDynamo to avoid verbose logs / potential instability
export TORCHDYNAMO_DISABLE=1
export TORCHDYNAMO_VERBOSE=0
# Avoid Torch log parsing issues: do not set TORCH_LOGS; if set externally, clear it.
unset TORCH_LOGS || true

# Args: <ckpt_path> <data_root> <out_dir>
CKPT_PATH="${1:-}"
DATA_ROOT="${2:-}"
OUT_DIR="${3:-scripts/infer/output/worldfloodsv2}"

if [[ -z "${CKPT_PATH}" || -z "${DATA_ROOT}" ]]; then
  echo "Usage: sbatch $0 <ckpt_path> <dataset_path> [out_dir]" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "Run inference: WorldFloodsv2"

# Note: AMP is disabled by default. If you want to enable AMP, append --amp if supported.
python scripts/infer/infer_worldfloodsv2.py \
  --checkpoint "${CKPT_PATH}" \
  --dataset-path "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --num-workers 8

echo "End: $(date)"
echo "Done. Output: ${OUT_DIR}/predictions"


