#!/bin/bash
#SBATCH --job-name=infer_kurosiwo
#SBATCH --output=scripts/infer/infer_kurosiwo_%j.out
#SBATCH --error=scripts/infer/infer_kurosiwo_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail

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

cd ${PROJECT_ROOT}

# 用法：
#   bash scripts/infer/infer_kurosiwo.sh \
#     ${PROJECT_ROOT}/logs/.../checkpoints/water_iou_0.6720.ckpt \
#     /WORK/DATA/KuroSiwoGRD \
#     scripts/infer/output/kurosiwo \
#     all \
#     false

CKPT_PATH="${1:-${PROJECT_ROOT}/logs/sam2_kurosiwo_dem-false_demscale-zscore_scale-db_ratio-true_cleardb-false_2025-11-25_14-53-29/checkpoints/water_iou_0.6604.ckpt}"
DATA_ROOT="${2:-data/KuroSiwoGRD}"
OUT_DIR="${3:-scripts/infer/output/kurosiwo}"
EVENTS="${4:-all}"
USE_DEM="${5:-false}"

if [[ -z "${CKPT_PATH}" || -z "${DATA_ROOT}" ]]; then
  echo "用法: bash scripts/infer/infer_kurosiwo.sh <ckpt_path> <data_root> [out_dir] [events] [use_dem]"
  exit 1
fi

mkdir -p "${OUT_DIR}"

export PYTHONPATH=.

EXTRA_ARGS=()
if [[ "${USE_DEM}" == "true" ]]; then
  EXTRA_ARGS+=(--use_dem)
  # 推理端 DEM 归一化 2 选1：与训练保持一致
  EXTRA_ARGS+=(--dem_scale_mode zscore)
fi

python scripts/infer/infer_kurosiwo.py \
  --checkpoint "${CKPT_PATH}" \
  --root "${DATA_ROOT}" \
  --out_dir "${OUT_DIR}" \
  --events "${EVENTS}" \
  --zscore \
  --batch_size 128 \
  --tile_size 256 \
  --overlap 64 \
  --mosaic_scale linear \
  --use_ratio \
  "${EXTRA_ARGS[@]}"

echo "✅ 完成：KuroSiwo 事件级推理与拼接输出目录: ${OUT_DIR}"



