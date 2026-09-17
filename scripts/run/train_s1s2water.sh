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

# 用法：
#   sbatch scripts/run/train_s1s2water.sh <EXPERIMENT_NAME> [ADD_DEM] [ADD_SLOPE] [EXTRA_OVERRIDES_STR]
# 说明：
#   EXPERIMENT_NAME：使用已有实验配置名，例如
#     dinov3_dinov3_s1s2water | sam2_sam2_s1s2water | efficientnetb4_mobilenetv3_s1s2water | resnet50_resnet50_s1s2water
#   ADD_DEM, ADD_SLOPE ∈ {true,false}，默认 true

EXP=${1:-dinov3_dinov3_s1s2water}
ADD_DEM=${2:-false}
ADD_SLOPE=${3:-false}
# 额外 Hydra 覆盖项（可选，第4个参数），例如关闭对齐或关闭对齐偏置：
#   "model.alignment_enabled=false model.fusion.xattn_align_bias=false experiment_name=dinov3_dinov3_s1s2water_align-off_bias-off"
EXTRA_OVERRIDES_STR=${4:-""}

# 数据根
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
DATA_ROOT_S1S2="${PROJECT_ROOT}/data/S1S2-Water"

# 记录名后缀（便于 TensorBoard 分组与日志对齐）
SUF="dem-${ADD_DEM}_slope-${ADD_SLOPE}"
EXP_NAME="${EXP}_${SUF}"

# 环境
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" "${PROJECT_ROOT}/logs"
export HF_HUB_OFFLINE=1

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood

echo "================ S1S2-Water 消融 ================"
echo " DEM/SLOPE: dem=${ADD_DEM}, slope=${ADD_SLOPE}"
echo " EXP/YAML:  ${EXP}"
echo " EXP_NAME:  ${EXP_NAME}"
echo " shallow_enabled: true  xattn_align_bias: true alignment_enabled: true"
echo "==============================================="

# Hydra 覆盖
# 依据 DEM/SLOPE 动态计算通道数（光学在基础4通道上追加，SAR 固定2通道）
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

# 追加用户指定的额外 Hydra 覆盖（用于关闭对齐损失、对齐偏置等）
if [[ -n "${EXTRA_OVERRIDES_STR}" ]]; then
  # 按空格拆分为数组，每一项都是一个独立的 override
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
  echo "[RUN] ❌ 失败 (code=$code)" >&2
  exit $code
else
  echo "[RUN] ✅ 成功结束"
fi


