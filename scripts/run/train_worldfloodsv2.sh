#!/bin/bash
#SBATCH --job-name=train_wf2
#SBATCH --output=scripts/run/%x_%j.out
#SBATCH --error=scripts/run/%x_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

# Usage:
#   sbatch scripts/run/train_worldfloodsv2.sh [CHANNELS_KEY] [EXPERIMENT_NAME]
# Notes:
#   - CHANNELS_KEY: must match the dataset config (e.g., bgri, all, ...)

CHANNELS_KEY=${1:-bgri}
EXP=${2:-dinov3_worldfloodsv2}

###############################################################################
# Runtime configuration (override via env vars if needed)
# - PROJECT_ROOT: repo root (default: inferred from this script location)
# - DATA_ROOT_BASE: dataset base dir (default: $PROJECT_ROOT/data)
# - DATA_ROOT_WF2: WorldFloodsv2 root (default: $DATA_ROOT_BASE/WorldFloodsv2)
# - CONDA_ENV: conda env name to activate (optional; otherwise activate before sbatch)
# - HF_ENDPOINT / HF_TOKEN: Hugging Face settings (optional)
# - HF_HUB_CACHE / TORCH_HOME: caches (optional; defaults under $PROJECT_ROOT/checkpoints)
###############################################################################

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT

DATA_ROOT_BASE="${DATA_ROOT_BASE:-${PROJECT_ROOT}/data}"
DATA_ROOT_WF2="${DATA_ROOT_WF2:-${DATA_ROOT_BASE}/WorldFloodsv2}"

SUF="ch-${CHANNELS_KEY}"
EXP_NAME="${EXP}_${SUF}"

: "${HF_ENDPOINT:=}"
: "${HF_TOKEN:=}"
: "${HF_HUB_CACHE:=${PROJECT_ROOT}/checkpoints/.cache}"
: "${TORCH_HOME:=${PROJECT_ROOT}/checkpoints}"
export HF_ENDPOINT HF_TOKEN HF_HUB_CACHE TORCH_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PROJECT_ROOT}/logs" || true

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}" || true
  fi
fi

cd "${PROJECT_ROOT}"

echo "================ WorldFloodsv2 run ================"
echo " Channels: ${CHANNELS_KEY}"
echo " Exp:      ${EXP}"
echo " Exp name: ${EXP_NAME}"
echo "==================================================="

OVERRIDES=(
  "experiment=${EXP}"
  "data=worldfloodsv2"
  "data.root=${DATA_ROOT_WF2}"
  "data.channels=${CHANNELS_KEY}"
  "experiment_name=${EXP_NAME}"
)

set +e
python src/train.py "${OVERRIDES[@]}"
code=$?
set -e

if [ $code -ne 0 ]; then
  echo "[RUN] FAILED (code=$code)" >&2
  exit $code
else
  echo "[RUN] OK"
fi


