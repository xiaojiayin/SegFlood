#!/bin/bash
#SBATCH --job-name=effb4mobilenetv2_wf2
#SBATCH --output=scripts/run/effb4mobilenetv2_wf2_%j.out
#SBATCH --error=scripts/run/effb4mobilenetv2_wf2_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

# Usage:
#   sbatch scripts/run/train_efficientnetb4_mobilenetv2_worldfloodsv2.sh [channels_key] [lr]
# Args:
#   channels_key: rgb|bgr|bgri|riswir|bgriswir|bgriswirs|l89s2|sub_20|all (default: bgri)
#   lr: learning rate (default: 5e-4)

CHANNELS_KEY=${1:-bgri}
LR=${2:-5e-4}

BASE_EXPERIMENT="efficientnetb4_mobilenetv2_worldfloodsv2"

###############################################################################
# Runtime configuration (override via env vars if needed)
# - PROJECT_ROOT: repo root (default: inferred from this script location)
# - DATA_ROOT: WorldFloodsv2 dataset root (required)
# - CONDA_ENV: conda env name to activate (optional; otherwise activate before sbatch)
# - HF_ENDPOINT / HF_TOKEN: Hugging Face settings (optional)
# - HF_HUB_CACHE / TORCH_HOME: caches (optional; defaults under $PROJECT_ROOT/checkpoints)
###############################################################################

# Repo root (infer from this script path)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

DATA_ROOT="${DATA_ROOT:-}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage: sbatch $0 [channels_key] [lr]" >&2
  echo "  and set DATA_ROOT=/path/to/WorldFloodsv2" >&2
  exit 1
fi

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

echo "================ Run plan ================"
echo "Modalities: optical"
echo "Channels: ${CHANNELS_KEY}"
echo "learning_rate: ${LR}"
echo "fusion_type: (single-modal; not used)"
echo "========================================="

run_base="efficientnetb4_mobilenetv2_optical_${CHANNELS_KEY}_wf2"

# Compute optical channel count from channels_key
case "$CHANNELS_KEY" in
  rgb) opt_ch=3 ;;
  bgr) opt_ch=3 ;;
  bgri) opt_ch=4 ;;
  riswir) opt_ch=3 ;;
  bgriswir) opt_ch=5 ;;
  bgriswirs) opt_ch=6 ;;
  l89s2) opt_ch=8 ;;
  sub_20) opt_ch=10 ;;
  all) opt_ch=13 ;;
  *) echo "[ERROR] Unsupported channels_key: $CHANNELS_KEY" >&2; exit 2 ;;
esac
sar_ch=0

echo -e "\n>>> [RUN] mode=optical start: $(date)"
echo "Channels: optical_channels=${opt_ch}, sar_channels=${sar_ch}"

set +e
python src/train.py \
  experiment=${BASE_EXPERIMENT} \
  data.root=${DATA_ROOT} \
  data.split=train \
  data.channels=${CHANNELS_KEY} \
  data.filter_windows.apply=true \
  data.filter_windows.version=v1 \
  data.filter_windows.threshold_clouds=0.25 \
  data.batch_size=128 \
  data.num_workers=8 \
  model.learning_rate=${LR} \
  +data.optical_channels=${opt_ch} \
  +data.sar_channels=${sar_ch}
exit_code=$?
set -e

if [ $exit_code -ne 0 ]; then
  echo "[RUN] mode=optical FAILED (code=$exit_code)" >&2
  exit $exit_code
else
  echo "[RUN] mode=optical OK"
fi

echo "Training finished: $(date)"


