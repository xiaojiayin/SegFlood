#!/bin/bash
#SBATCH --job-name=train_exp
#SBATCH --output=scripts/run/dinov3_dinov3_s1s2water_%j.out
#SBATCH --error=scripts/run/dinov3_dinov3_s1s2water_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

# Usage:
#   sbatch scripts/run/train_experiment.sh <experiment_name>
# Dataset key is inferred from experiment_name:
#   cauflood | gffloodnet | s1s2water | kurosiwo | worldfloodsv2

EXPERIMENT_NAME=${1:-resnet50_resnet50_gffloodnet}

if [ -z "${EXPERIMENT_NAME}" ]; then
  echo "Usage: $0 <experiment_name>" >&2
  exit 1
fi

###############################################################################
# Runtime configuration (override via env vars if needed)
# - PROJECT_ROOT: repo root (default: inferred from this script location)
# - DATA_ROOT_BASE: dataset base dir (default: $PROJECT_ROOT/data)
# - CONDA_ENV: conda env name to activate (optional; otherwise activate before sbatch)
# - HF_ENDPOINT / HF_TOKEN: Hugging Face settings (optional)
# - HF_HUB_CACHE / TORCH_HOME: caches (optional; defaults under $PROJECT_ROOT/checkpoints)
###############################################################################

# Repo root (infer from this script path)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

# Dataset roots
DATA_ROOT_BASE="${DATA_ROOT_BASE:-${PROJECT_ROOT}/data}"
DATA_ROOT_CAU="${DATA_ROOT_CAU:-${DATA_ROOT_BASE}/CAU-Flood}"
DATA_ROOT_GF="${DATA_ROOT_GF:-${DATA_ROOT_BASE}/GF-FloodNet}"
DATA_ROOT_S1S2="${DATA_ROOT_S1S2:-${DATA_ROOT_BASE}/S1S2-Water}"
DATA_ROOT_KURO="${DATA_ROOT_KURO:-${DATA_ROOT_BASE}/KuroSiwoGRD}"
DATA_ROOT_WF2="${DATA_ROOT_WF2:-${DATA_ROOT_BASE}/WorldFloodsv2}"

# Infer dataset key and default root from experiment name
DATASET_KEY=""
DATA_ROOT=""
case "${EXPERIMENT_NAME}" in
  *cauflood*)    DATASET_KEY="cau_flood";   DATA_ROOT="${DATA_ROOT_CAU}" ;;
  *gffloodnet*)  DATASET_KEY="gf_floodnet"; DATA_ROOT="${DATA_ROOT_GF}"  ;;
  *s1s2water*)   DATASET_KEY="s1s2_water";  DATA_ROOT="${DATA_ROOT_S1S2}";;
  *kurosiwo*)    DATASET_KEY="kurosiwo";    DATA_ROOT="${DATA_ROOT_KURO}";;
  *worldfloodsv2*) DATASET_KEY="worldfloodsv2"; DATA_ROOT="${DATA_ROOT_WF2}";;
  *)
    echo "[ERROR] Cannot infer dataset key from experiment name. Include one of:" >&2
    echo "        cauflood|gffloodnet|s1s2water|kurosiwo|worldfloodsv2" >&2
    exit 1
    ;;
esac

# Logs
LOG_ROOT="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

: "${HF_ENDPOINT:=}"
: "${HF_TOKEN:=}"
: "${HF_HUB_CACHE:=${PROJECT_ROOT}/checkpoints/.cache}"
: "${TORCH_HOME:=${PROJECT_ROOT}/checkpoints}"
export HF_ENDPOINT HF_TOKEN HF_HUB_CACHE TORCH_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" || true

# Optional conda activation (recommended to activate in your sbatch wrapper/environment)
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}" || true
  fi
fi

echo "================ Train plan ================"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Dataset:    ${DATASET_KEY}"
echo "Data root:  ${DATA_ROOT}"
echo "==========================================="

# Hydra overrides
OVERRIDES=(
  "experiment=${EXPERIMENT_NAME}"
  "data=${DATASET_KEY}"
  "data.root=${DATA_ROOT}"
  "experiment_name=${EXPERIMENT_NAME}"
)

# Keep this wrapper minimal; configure most options in YAML.

set +e
python src/train.py "${OVERRIDES[@]}"
exit_code=$?
set -e

if [ $exit_code -ne 0 ]; then
  echo "[RUN] FAILED (code=$exit_code)" >&2
  exit $exit_code
else
  echo "[RUN] OK"
fi


