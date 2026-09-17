#!/bin/bash
#SBATCH --job-name=infer_s1s2water
#SBATCH --output=scripts/infer/infer_s1s2water_%j.out
#SBATCH --error=scripts/infer/infer_s1s2water_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail

# 参考 infer_and_eval.sh 环境设置
echo "🔧 设置环境..."
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"  # set your Hugging Face token in the environment if gated weights are needed
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"

echo "作业ID: ${SLURM_JOB_ID:-N/A}"
echo "开始时间: $(date)"

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood
echo "Python: $(python --version)"


# 用法：
#   bash scripts/infer/infer_s1s2water.sh \
#     ${PROJECT_ROOT}/logs/dinov3_dinov3_s1s2water_dem-false_slope-false_2025-11-24_14-35-40/checkpoints/water_iou_0.9742.ckpt \
#     ${PROJECT_ROOT}/data/S1S2-Water \
#     scripts/infer/output/s1s2water \
#     dual \
#     false \
#     false

CKPT_PATH="${1:-${PROJECT_ROOT}/logs/dinov3_dinov3_s1s2water_dem-false_slope-false_2025-11-24_14-35-40/checkpoints/water_iou_0.9742.ckpt}"
DATA_ROOT="${2:-data/S1S2-Water}"
OUT_DIR="${3:-scripts/infer/output/s1s2water}"
MODAL_TYPE="${4:-dual}"
ADD_DEM="${5:-false}"
ADD_SLOPE="${6:-false}"

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood

cd ${PROJECT_ROOT}
mkdir -p "${OUT_DIR}"

export PYTHONPATH=.

EXTRA_ARGS=()
if [[ "${ADD_DEM}" == "true" ]]; then
  EXTRA_ARGS+=(--add-dem)
fi
if [[ "${ADD_SLOPE}" == "true" ]]; then
  EXTRA_ARGS+=(--add-slope)
fi

# 说明：默认禁用 AMP（混合精度）。原因：在该数据集上启用 AMP 会导致指标下降约 30 个百分点。
# 如需实验性启用，可在下方 python 命令末尾手动追加 --amp。
python scripts/infer/infer_s1s2water.py \
  --checkpoint "${CKPT_PATH}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --modal-type "${MODAL_TYPE}" \
  --batch-size 128 \
  --num-workers 8 \
  "${EXTRA_ARGS[@]}"

echo "✅ 完成：输出目录 ${OUT_DIR}（predictions/ 与 mosaics/）"


