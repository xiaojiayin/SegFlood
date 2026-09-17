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

# 用法：
#   sbatch scripts/run/train_worldfloodsv2.sh [CHANNELS_KEY] [EXPERIMENT_NAME]
# 说明：
#   CHANNELS_KEY（写入 data.channels）：可选值与数据配置一致，例如
#     rgb | bgr | bgri | riswir | bgriswir | bgriswirs | l89s2 | sub_20 | all
#   EXPERIMENT_NAME：直接指定实验配置名，例如
#     dinov3_worldfloodsv2 | sam2_worldfloodsv2 | efficientnetb4_worldfloodsv2 | resnet50_worldfloodsv2 | mobilenetv3_worldfloodsv2

CHANNELS_KEY=${1:-bgri}
EXP=${2:-dinov3_worldfloodsv2}

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
DATA_ROOT_WF2="${PROJECT_ROOT}/data/WorldFloodsv2"

SUF="ch-${CHANNELS_KEY}"
EXP_NAME="${EXP}_${SUF}"

export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PROJECT_ROOT}/logs"
export HF_HUB_OFFLINE=1

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood

echo "================ WorldFloodsv2 运行 ================"
echo " 通道:    ${CHANNELS_KEY}"
echo " 实验:    ${EXP}"
echo " EXP:     ${EXP}"
echo " EXP_NAME:${EXP_NAME}"
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
  echo "[RUN] ❌ 失败 (code=$code)" >&2
  exit $code
else
  echo "[RUN] ✅ 成功结束"
fi


