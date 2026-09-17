#!/usr/bin/env bash
#SBATCH --job-name=eval_ckpt
#SBATCH --output=scripts/run/%x_%j.out
#SBATCH --error=scripts/run/%x_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

# Usage:
#   sbatch scripts/run/eval_checkpoint.sh <EXPERIMENT> <CHECKPOINT> [RUN_NAME]

EXP="${1:?experiment config is required}"
CKPT="${2:?checkpoint path is required}"
RUN_NAME="${3:-${EXP}_reeval}"
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
case "${EXP}" in
  *cauflood*)      DATASET_KEY="cau_flood";    DATA_ROOT="${PROJECT_ROOT}/data/CAU-Flood" ;;
  *gffloodnet*)    DATASET_KEY="gf_floodnet";  DATA_ROOT="${PROJECT_ROOT}/data/GF-FloodNet" ;;
  *s1s2water*)     DATASET_KEY="s1s2_water";   DATA_ROOT="${PROJECT_ROOT}/data/S1S2-Water" ;;
  *kurosiwo*)      DATASET_KEY="kurosiwo";     DATA_ROOT="${PROJECT_ROOT}/data/KuroSiwoGRD" ;;
  *worldfloodsv2*) DATASET_KEY="worldfloodsv2"; DATA_ROOT="${PROJECT_ROOT}/data/WorldFloodsv2" ;;
  *)
    echo "Cannot infer dataset from experiment: ${EXP}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 2
fi

export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
export HF_HUB_OFFLINE=1
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PROJECT_ROOT}/logs"

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood
cd "${PROJECT_ROOT}"

python src/eval.py \
  "experiment=${EXP}" \
  "data=${DATASET_KEY}" \
  "data.root=${DATA_ROOT}" \
  "ckpt_path=${CKPT}" \
  "experiment_name=${RUN_NAME}" \
  "logger=tensorboard" \
  "trainer=gpu" \
  "trainer.precision=bf16-mixed"
