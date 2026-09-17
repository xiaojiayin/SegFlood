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

# 用法：
#   sbatch scripts/run/train_kurosiwo.sh <EXPERIMENT_NAME> [USE_DEM] [DEM_SCALE_MODE] [SCALE_INPUT] [USE_RATIO] [CLEAR_DB_STATS]
# 说明：
#   EXPERIMENT_NAME：使用已有实验配置名，例如
#     dinov3_kurosiwo | resnet50_kurosiwo | efficientnetb4_kurosiwo | mobilenetv3_kurosiwo | sam2_kurosiwo
#   USE_DEM        ∈ {true,false}，是否启用 DEM 早期融合（默认 false）
#   DEM_SCALE_MODE ∈ {zscore,none}，DEM 归一化方式（默认 zscore）
#   SCALE_INPUT    ：KuroSiwo 中 SAR 两个通道的标准化方式
#                    可选：normalize, min-max, custom, log（默认 db 将在 configs/data/kurosiwo.yaml 中配置）
#   USE_RATIO      ∈ {true,false}，是否在 SAR 中追加 vh/vv 比值通道（channels=["vv","vh","vh/vv"]，默认 false）
#   CLEAR_DB_STATS ∈ {true,false}，当 SCALE_INPUT=db 且为 true 时，将 data_mean/data_std 置为 null（即使用“纯 dB”输入）

EXP=${1:-dinov3_kurosiwo}
USE_DEM=${2:-false}
DEM_SCALE_MODE=${3:-zscore}
SCALE_INPUT=${4:-db}
USE_RATIO=${5:-false}
CLEAR_DB_STATS=${6:-false}
# 额外的 Hydra 覆盖项（可选，第7个参数），例如修改主干模型名称：
#   "model.encoder.optical_model_name=vit_small_patch16_dinov3 model.encoder.sar_model_name=vit_small_patch16_dinov3"
EXTRA_OVERRIDES_STR=${7:-""}

# 数据根（指向实际存放 KuroSiwoGRD 的目录，需包含 pickle 子目录）
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
DATA_ROOT_KURO="${PROJECT_ROOT}/data/KuroSiwoGRD"

# 记录名后缀（便于 TensorBoard 分组与日志对齐）
SUF="dem-${USE_DEM}_demscale-${DEM_SCALE_MODE}_scale-${SCALE_INPUT}_ratio-${USE_RATIO}_cleardb-${CLEAR_DB_STATS}"
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

echo "================ KuroSiwo 实验 ================"
echo " USE_DEM:        ${USE_DEM}"
echo " DEM_SCALE_MODE: ${DEM_SCALE_MODE}  (zscore | none)"
echo " SCALE_INPUT:    ${SCALE_INPUT}  (normalize | log | db | min-max | custom)"
echo " USE_RATIO:      ${USE_RATIO}  (是否追加 vh/vv 通道)"
echo " CLEAR_DB_STATS: ${CLEAR_DB_STATS}  (仅当 SCALE_INPUT=db 时生效，为 true 表示使用纯 dB)"
echo " EXP/YAML:       ${EXP}"
echo " EXP_NAME:       ${EXP_NAME}"
echo " data.root:      ${DATA_ROOT_KURO}"
echo "================================================"

# Hydra 覆盖
# 依据 DEM 和 vh/vv 比值动态计算 SAR 通道数：
#   基础:  VV+VH => 2 通道
#   若 USE_RATIO=true 则追加 vh/vv => +1 通道
#   若 USE_DEM=true  则追加 DEM   => +1 通道
base_ch=2
if [[ "${USE_RATIO}" == "true" ]]; then
  base_ch=$((base_ch + 1))
fi
sar_ch=${base_ch}
if [[ "${USE_DEM}" == "true" ]]; then
  sar_ch=$((sar_ch + 1))
fi

# SAR 通道名（Dataset 的 channels 参数）
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

# 当需要“纯 dB”输入时，清空均值/方差，避免在 dB 域再次做 Z-Score
if [[ "${SCALE_INPUT}" == "db" && "${CLEAR_DB_STATS}" == "true" ]]; then
  OVERRIDES+=("data.data_mean=null")
  OVERRIDES+=("data.data_std=null")
fi

# 追加用户指定的额外 Hydra 覆盖（用于更换主干等）
if [[ -n "${EXTRA_OVERRIDES_STR}" ]]; then
  # 按空格拆分为数组，每一项都是一个独立的 override
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


