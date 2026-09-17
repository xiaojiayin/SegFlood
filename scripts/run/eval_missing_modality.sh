#!/bin/bash
#SBATCH --job-name=eval_missing_modality
#SBATCH --output=scripts/run/%x_%j.out
#SBATCH --error=scripts/run/%x_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

# 用法：
#   sbatch scripts/run/eval_missing_modality.sh <EXPERIMENT_NAME> <CKPT_PATH> <MODE>
#
# MODE:
#   dual          正常双模态测试
#   optical_only  多模态训练权重，测试时只保留 optical 分支
#   sar_only      多模态训练权重，测试时只保留 SAR 分支
#
# 示例：
#   sbatch scripts/run/eval_missing_modality.sh resnet50_resnet50_gffloodnet /path/to/best.ckpt optical_only

EXP="${1:?需要实验配置名，例如 resnet50_resnet50_gffloodnet}"
CKPT="${2:?需要 checkpoint 路径}"
MODE="${3:-dual}"

case "${MODE}" in
  dual|optical_only|sar_only) ;;
  *)
    echo "MODE 必须是 dual | optical_only | sar_only，当前: ${MODE}" >&2
    exit 1
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
DATA_ROOT_CAU="${PROJECT_ROOT}/data/CAU-Flood"
DATA_ROOT_GF="${PROJECT_ROOT}/data/GF-FloodNet"
DATA_ROOT_S1S2="${PROJECT_ROOT}/data/S1S2-Water"

case "${EXP}" in
  *cauflood*)    DATASET_KEY="cau_flood";  DATA_ROOT="${DATA_ROOT_CAU}" ;;
  *gffloodnet*)  DATASET_KEY="gf_floodnet"; DATA_ROOT="${DATA_ROOT_GF}" ;;
  *s1s2water*)   DATASET_KEY="s1s2_water"; DATA_ROOT="${DATA_ROOT_S1S2}" ;;
  *)
    echo "缺模态实验仅支持 cauflood | gffloodnet | s1s2water 双模态配置，当前: ${EXP}" >&2
    exit 1
    ;;
esac

export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PROJECT_ROOT}/logs"
export HF_HUB_OFFLINE=1

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood

RUN_NAME="${EXP}_missing-${MODE}"

echo "================ Missing-modality eval ================"
echo " EXP:        ${EXP}"
echo " CKPT:       ${CKPT}"
echo " DATASET:    ${DATASET_KEY}"
echo " DATA_ROOT:  ${DATA_ROOT}"
echo " MODE:       ${MODE}"
echo " RUN_NAME:   ${RUN_NAME}"
echo "======================================================="

cd "${PROJECT_ROOT}"

python src/eval.py \
  "experiment=${EXP}" \
  "data=${DATASET_KEY}" \
  "data.root=${DATA_ROOT}" \
  "data.eval_modal_mode=${MODE}" \
  "ckpt_path=${CKPT}" \
  "experiment_name=${RUN_NAME}" \
  "logger=tensorboard" \
  "trainer=gpu" \
  "trainer.precision=bf16-mixed"
