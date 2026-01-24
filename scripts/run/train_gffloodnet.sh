#!/bin/bash
#SBATCH --job-name=train_gffloodnet
#SBATCH --output=scripts/run/%x_%j.out
#SBATCH --error=scripts/run/%x_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

# Usage:
#   sbatch scripts/run/train_gffloodnet.sh <EXPERIMENT_NAME> [EXTRA_OVERRIDES_STR]
# Notes:
#   - EXPERIMENT_NAME: e.g. resnet50_resnet50_gffloodnet
#   - EXTRA_OVERRIDES_STR: optional Hydra overrides (space-separated)

EXP=${1:-resnet50_resnet50_gffloodnet}
EXTRA_OVERRIDES_STR=${2:-""}

###############################################################################
# Open-source friendly settings (no secrets / no machine paths)
# - PROJECT_ROOT: repo root (default: inferred from this script location)
# - DATA_ROOT_BASE: base directory that contains datasets (default: $PROJECT_ROOT/data)
# - CONDA_ENV: conda env name to activate (optional)
# - HF_ENDPOINT / HF_TOKEN: optional Hugging Face settings (optional)
###############################################################################

# Repo root (infer from this script path)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT

DATA_ROOT_BASE="${DATA_ROOT_BASE:-${PROJECT_ROOT}/data}"
DATA_ROOT_GF="${DATA_ROOT_GF:-${DATA_ROOT_BASE}/GF-FloodNet}"

EXP_NAME="${EXP}"

: "${HF_ENDPOINT:=}"
: "${HF_TOKEN:=}"
: "${HF_HUB_CACHE:=${PROJECT_ROOT}/checkpoints/.cache}"
: "${TORCH_HOME:=${PROJECT_ROOT}/checkpoints}"
export HF_ENDPOINT HF_TOKEN HF_HUB_CACHE TORCH_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"

mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PROJECT_ROOT}/logs"

# Optional conda activation (recommended to activate in your sbatch wrapper/environment)
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}" || true
  fi
fi

echo "================ GF-FloodNet run ================"
echo " EXP/YAML:       ${EXP}"
echo " EXP_NAME:       ${EXP_NAME}"
echo " data.root:      ${DATA_ROOT_GF}"
echo " EXTRA_OVERRIDES: ${EXTRA_OVERRIDES_STR}"
echo "================================================"

# Hydra overrides
OVERRIDES=(
  "experiment=${EXP}"
  "data=gf_floodnet"
  "data.root=${DATA_ROOT_GF}"
  "experiment_name=${EXP_NAME}"
)

# Append user overrides (space-separated)
if [[ -n "${EXTRA_OVERRIDES_STR}" ]]; then
  read -r -a extra_arr <<< "${EXTRA_OVERRIDES_STR}"
  for item in "${extra_arr[@]}"; do
    OVERRIDES+=("${item}")
  done
fi

cd "${PROJECT_ROOT}"

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



