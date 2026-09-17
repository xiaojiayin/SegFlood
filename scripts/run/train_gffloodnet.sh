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

# 用法：
#   sbatch scripts/run/train_gffloodnet.sh <EXPERIMENT_NAME> [EXTRA_OVERRIDES_STR]
# 说明：
#   EXPERIMENT_NAME：使用已有实验配置名，例如
#     resnet50_resnet50_gffloodnet
#   EXTRA_OVERRIDES_STR：可选的 Hydra 覆盖字符串，用于关闭 alignment / 修改 fusion 等

EXP=${1:-resnet50_resnet50_gffloodnet}
EXTRA_OVERRIDES_STR=${2:-""}

# 数据根
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
DATA_ROOT_GF="${PROJECT_ROOT}/data/GF-FloodNet"

EXP_NAME="${EXP}"

# 环境
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PROJECT_ROOT}/logs"
export HF_HUB_OFFLINE=1

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood

echo "================ GF-FloodNet 实验 ================"
echo " EXP/YAML:       ${EXP}"
echo " EXP_NAME:       ${EXP_NAME}"
echo " data.root:      ${DATA_ROOT_GF}"
echo " EXTRA_OVERRIDES: ${EXTRA_OVERRIDES_STR}"
echo "================================================"

# Hydra 覆盖
OVERRIDES=(
  "experiment=${EXP}"
  "data=gf_floodnet"
  "data.root=${DATA_ROOT_GF}"
  "experiment_name=${EXP_NAME}"
)

# 追加用户指定的额外 Hydra 覆盖（用于关闭对齐损失、对齐偏置、改变 fusion 策略等）
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
  echo "[RUN] ❌ 失败 (code=$code)" >&2
  exit $code
else
  echo "[RUN] ✅ 成功结束"
fi



