#!/bin/bash
#SBATCH --job-name=eval_robustness
#SBATCH --output=scripts/run/jag_supplement/logs/robustness_eval/%x_%j.out
#SBATCH --error=scripts/run/jag_supplement/logs/robustness_eval/%x_%j.err
#SBATCH --partition=a01,h01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# Usage:
#   sbatch eval_robustness.sh <EXPERIMENT> <CKPT> <RUN_NAME> <MODAL_MODE> <DEGRADATION> <STRENGTH> "<MODEL_OVERRIDES>"
#   MODAL_MODE:  dual | optical_only | sar_only
#   DEGRADATION: none | sar_noise | optical_cloud
#   MODEL_OVERRIDES must reproduce the architecture of the checkpoint
#   (component gate, alignment projection heads, bias, ...), otherwise the
#   state dict will not load.

set -euo pipefail

EXP="${1:?experiment}"
CKPT="${2:?ckpt}"
RUN_NAME="${3:?run name}"
MODE="${4:-dual}"
DEG="${5:-none}"
STRENGTH="${6:-0.5}"
MODEL_OVR="${7:-}"

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
case "${EXP}" in
  *cauflood*)   DATASET_KEY="cau_flood";   DATA_ROOT="${PROJECT_ROOT}/data/CAU-Flood" ;;
  *gffloodnet*) DATASET_KEY="gf_floodnet"; DATA_ROOT="${PROJECT_ROOT}/data/GF-FloodNet" ;;
  *s1s2water*)  DATASET_KEY="s1s2_water";  DATA_ROOT="${PROJECT_ROOT}/data/S1S2-Water" ;;
  *) echo "unsupported experiment ${EXP}" >&2; exit 1 ;;
esac

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
export HF_HUB_OFFLINE=1
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}"

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood
cd "${PROJECT_ROOT}"

echo "EXP=${EXP} CKPT=${CKPT} MODE=${MODE} DEG=${DEG} STRENGTH=${STRENGTH}"
echo "MODEL_OVR=${MODEL_OVR}"

# shellcheck disable=SC2086
python src/eval.py \
  "experiment=${EXP}" \
  "data=${DATASET_KEY}" \
  "data.root=${DATA_ROOT}" \
  "data.eval_modal_mode=${MODE}" \
  "ckpt_path=${CKPT}" \
  "experiment_name=${RUN_NAME}" \
  "logger=tensorboard" \
  "trainer=gpu" \
  "trainer.precision=bf16-mixed" \
  "+model.test_degradation=${DEG}" \
  "+model.test_degradation_strength=${STRENGTH}" \
  ${MODEL_OVR}
