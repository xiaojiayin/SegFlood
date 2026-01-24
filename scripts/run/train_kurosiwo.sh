#!/bin/bash
#SBATCH --job-name=train_kurosiwo
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
#   sbatch scripts/run/train_kurosiwo.sh <EXPERIMENT_NAME> [USE_DEM] [DEM_SCALE_MODE] [SCALE_INPUT] [USE_RATIO] [CLEAR_DB_STATS]
# Notes:
#   - USE_RATIO adds vh/vv as an extra channel (must match training config).

EXP=${1:-dinov3_kurosiwo}
USE_DEM=${2:-false}
DEM_SCALE_MODE=${3:-zscore}
SCALE_INPUT=${4:-db}
USE_RATIO=${5:-false}
CLEAR_DB_STATS=${6:-false}
# Extra Hydra overrides (optional, arg #7), e.g. change backbone names:
#   "model.encoder.optical_model_name=vit_small_patch16_dinov3 model.encoder.sar_model_name=vit_small_patch16_dinov3"
EXTRA_OVERRIDES_STR=${7:-""}

###############################################################################
# Runtime configuration (override via env vars if needed)
# - PROJECT_ROOT: repo root (default: inferred from this script location)
# - DATA_ROOT_BASE: dataset base dir (default: $PROJECT_ROOT/data)
# - DATA_ROOT_KURO: KuroSiwoGRD root (default: $DATA_ROOT_BASE/KuroSiwoGRD)
# - CONDA_ENV: conda env name to activate (optional; otherwise activate before sbatch)
# - HF_ENDPOINT / HF_TOKEN: Hugging Face settings (optional)
# - HF_HUB_CACHE / TORCH_HOME: caches (optional; defaults under $PROJECT_ROOT/checkpoints)
###############################################################################

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT

DATA_ROOT_BASE="${DATA_ROOT_BASE:-${PROJECT_ROOT}/data}"
DATA_ROOT_KURO="${DATA_ROOT_KURO:-${DATA_ROOT_BASE}/KuroSiwoGRD}"

# Run name suffix (for TensorBoard grouping)
SUF="dem-${USE_DEM}_demscale-${DEM_SCALE_MODE}_scale-${SCALE_INPUT}_ratio-${USE_RATIO}_cleardb-${CLEAR_DB_STATS}"
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

echo "================ KuroSiwo run ================"
echo " USE_DEM:        ${USE_DEM}"
echo " DEM_SCALE_MODE: ${DEM_SCALE_MODE}  (zscore | none)"
echo " SCALE_INPUT:    ${SCALE_INPUT}  (normalize | log | db | min-max | custom)"
echo " USE_RATIO:      ${USE_RATIO}"
echo " CLEAR_DB_STATS: ${CLEAR_DB_STATS}"
echo " EXP/YAML:       ${EXP}"
echo " EXP_NAME:       ${EXP_NAME}"
echo " data.root:      ${DATA_ROOT_KURO}"
echo "================================================"

# Hydra overrides
# Dynamically compute SAR channel count:
#   base: VV+VH => 2 channels
#   +1 if USE_RATIO=true (vh/vv)
#   +1 if USE_DEM=true   (DEM)
base_ch=2
if [[ "${USE_RATIO}" == "true" ]]; then
  base_ch=$((base_ch + 1))
fi
sar_ch=${base_ch}
if [[ "${USE_DEM}" == "true" ]]; then
  sar_ch=$((sar_ch + 1))
fi

# SAR channel names (Dataset.channels)
if [[ "${USE_RATIO}" == "true" ]]; then
  CHANNELS="[vv,vh,vh/vv]"
else
  CHANNELS="[vv,vh]"
fi

OVERRIDES=(
  "experiment=${EXP}"
  "data=kurosiwo"
  "data.root=${DATA_ROOT_KURO}"
  "data.scale_input=${SCALE_INPUT}"
  "data.dem=${USE_DEM}"
  "data.dem_scale_mode=${DEM_SCALE_MODE}"
  "data.channels=${CHANNELS}"
  "data.sar_channels=${sar_ch}"
  "experiment_name=${EXP_NAME}"
)

# If "pure dB" is requested, clear mean/std to avoid applying z-score in dB domain.
if [[ "${SCALE_INPUT}" == "db" && "${CLEAR_DB_STATS}" == "true" ]]; then
  OVERRIDES+=("data.data_mean=null")
  OVERRIDES+=("data.data_std=null")
fi

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


