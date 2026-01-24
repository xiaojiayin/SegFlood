#!/bin/bash
#SBATCH --job-name=train_s1s2
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
#   sbatch scripts/run/train_s1s2water.sh <EXPERIMENT_NAME> [ADD_DEM] [ADD_SLOPE] [EXTRA_OVERRIDES_STR]
# Notes:
#   - ADD_DEM / ADD_SLOPE: true|false

EXP=${1:-dinov3_dinov3_s1s2water}
ADD_DEM=${2:-false}
ADD_SLOPE=${3:-false}
# Extra Hydra overrides (optional, arg #4), e.g. disable alignment or align-bias:
#   "model.alignment_enabled=false model.fusion.xattn_align_bias=false experiment_name=dinov3_dinov3_s1s2water_align-off_bias-off"
EXTRA_OVERRIDES_STR=${4:-""}

###############################################################################
# Runtime configuration (override via env vars if needed)
# - PROJECT_ROOT: repo root (default: inferred from this script location)
# - DATA_ROOT_BASE: dataset base dir (default: $PROJECT_ROOT/data)
# - DATA_ROOT_S1S2: S1S2-Water root (default: $DATA_ROOT_BASE/S1S2-Water)
# - CONDA_ENV: conda env name to activate (optional; otherwise activate before sbatch)
# - HF_ENDPOINT / HF_TOKEN: Hugging Face settings (optional)
# - HF_HUB_CACHE / TORCH_HOME: caches (optional; defaults under $PROJECT_ROOT/checkpoints)
###############################################################################

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT

DATA_ROOT_BASE="${DATA_ROOT_BASE:-${PROJECT_ROOT}/data}"
DATA_ROOT_S1S2="${DATA_ROOT_S1S2:-${DATA_ROOT_BASE}/S1S2-Water}"

# Run name suffix (for TensorBoard grouping)
SUF="dem-${ADD_DEM}_slope-${ADD_SLOPE}"
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

echo "================ S1S2-Water run ================"
echo " DEM/SLOPE: dem=${ADD_DEM}, slope=${ADD_SLOPE}"
echo " EXP/YAML:  ${EXP}"
echo " EXP_NAME:  ${EXP_NAME}"
echo "==============================================="

# Hydra overrides
# Compute channel counts based on DEM/SLOPE (optical starts with 4 channels; SAR fixed at 2)
opt_ch=4
sar_ch=2
if [[ "${ADD_DEM}" == "true" ]]; then
  opt_ch=$((opt_ch + 1))
fi
if [[ "${ADD_SLOPE}" == "true" ]]; then
  opt_ch=$((opt_ch + 1))
fi

OVERRIDES=(
  "experiment=${EXP}"
  "data=s1s2_water"
  "data.root=${DATA_ROOT_S1S2}"
  "data.add_dem=${ADD_DEM}"
  "data.add_slope=${ADD_SLOPE}"
  "data.optical_channels=${opt_ch}"
  "data.sar_channels=${sar_ch}"
  "experiment_name=${EXP_NAME}"
)

# Append user overrides (space-separated)
if [[ -n "${EXTRA_OVERRIDES_STR}" ]]; then
  read -r -a extra_arr <<< "${EXTRA_OVERRIDES_STR}"
  for item in "${extra_arr[@]}"; do
    OVERRIDES+=("${item}")
  done
fi

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


