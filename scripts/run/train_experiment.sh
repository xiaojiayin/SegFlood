#!/bin/bash
#SBATCH --job-name=train_exp
#SBATCH --output=scripts/run/dinov3_dinov3_s1s2water_%j.out
#SBATCH --error=scripts/run/dinov3_dinov3_s1s2water_%j.err
#SBATCH --partition=a01,h01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail

# 用法：
#   sbatch scripts/run/train_experiment.sh <experiment_name> [data_root_override] [extra_overrides]
#
# 数据集关键词（将从 experiment_name 自动解析）：
#   cauflood | gffloodnet | s1s2water | kurosiwo | worldfloodsv2
#
# 示例：
#   sbatch scripts/run/train_experiment.sh resnet50_resnet50_cauflood
#   sbatch scripts/run/train_experiment.sh dinov3_kurosiwo
#   sbatch scripts/run/train_experiment.sh efficientnetb4_worldfloodsv2

EXPERIMENT_NAME=${1:-dinov3_dinov3_s1s2water}
DATA_ROOT_OVERRIDE=${2:-""}
EXTRA_OVERRIDES_STR=${3:-""}

if [ -z "${EXPERIMENT_NAME}" ]; then
  echo "用法: $0 <experiment_name> [data_root_override] [extra_overrides]" >&2
  exit 1
fi

# 根路径映射（按数据集名）
# 如需修改请在此处统一调整
DATA_ROOT_CAU="${PROJECT_ROOT}/data/CAU-Flood"
DATA_ROOT_GF="${PROJECT_ROOT}/data/GF-FloodNet"
DATA_ROOT_S1S2="${PROJECT_ROOT}/data/S1S2-Water"
DATA_ROOT_KURO="${PROJECT_ROOT}/data/KuroSiwoGRD"
DATA_ROOT_WF2="${PROJECT_ROOT}/data/WorldFloodsv2"

# 根据实验名推断数据集 key 和默认 root
DATASET_KEY=""
DATA_ROOT=""
case "${EXPERIMENT_NAME}" in
  *cauflood*)    DATASET_KEY="cau_flood";   DATA_ROOT="${DATA_ROOT_CAU}" ;;
  *gffloodnet*)  DATASET_KEY="gf_floodnet"; DATA_ROOT="${DATA_ROOT_GF}"  ;;
  *s1s2water*)   DATASET_KEY="s1s2_water";  DATA_ROOT="${DATA_ROOT_S1S2}";;
  *kurosiwo*)    DATASET_KEY="kurosiwo";    DATA_ROOT="${DATA_ROOT_KURO}";;
  *worldfloodsv2*) DATASET_KEY="worldfloodsv2"; DATA_ROOT="${DATA_ROOT_WF2}";;
  *)
    echo "无法从实验名推断数据集，请包含以下关键词之一: cauflood|gffloodnet|s1s2water|kurosiwo|worldfloodsv2" >&2
    exit 1
    ;;
esac

if [[ -n "${DATA_ROOT_OVERRIDE}" ]]; then
  DATA_ROOT="${DATA_ROOT_OVERRIDE}"
fi

# 环境设置
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
LOG_ROOT="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

# HF 缓存（需要可开启）
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}"
export HF_HUB_OFFLINE=1

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood

# 打印计划
echo "================ 训练计划 ================"
echo "  实验: ${EXPERIMENT_NAME}"
echo "  数据集: ${DATASET_KEY}"
echo "  data.root: ${DATA_ROOT}"
echo "  EXTRA_OVERRIDES: ${EXTRA_OVERRIDES_STR}"
echo "========================================="

# 组合 Hydra 覆盖
OVERRIDES=(
  "experiment=${EXPERIMENT_NAME}"
  "data=${DATASET_KEY}"
  "data.root=${DATA_ROOT}"
  "experiment_name=${EXPERIMENT_NAME}"
)

if [[ -n "${EXTRA_OVERRIDES_STR}" ]]; then
  read -r -a extra_arr <<< "${EXTRA_OVERRIDES_STR}"
  for item in "${extra_arr[@]}"; do
    OVERRIDES+=("${item}")
  done
fi

set +e
python src/train.py "${OVERRIDES[@]}"
exit_code=$?
set -e

if [ $exit_code -ne 0 ]; then
  echo "[RUN] ❌ 失败 (code=$exit_code)" >&2
  exit $exit_code
else
  echo "[RUN] ✅ 成功结束"
fi


