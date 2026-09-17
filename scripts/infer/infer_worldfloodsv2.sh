#!/bin/bash
#SBATCH --job-name=infer_wf2
#SBATCH --output=scripts/infer/infer_worldfloodsv2_%j.out
#SBATCH --error=scripts/infer/infer_worldfloodsv2_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

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

# 关闭 TorchDynamo 编译，避免符号形状日志/潜在不稳定
export TORCHDYNAMO_DISABLE=1
export TORCHDYNAMO_VERBOSE=0
# 避免 Torch 日志解析异常：不设置 TORCH_LOGS；若外部已设置则显式清除
unset TORCH_LOGS || true

# 可传参: <ckpt_path> <data_root> <out_dir>
CKPT_PATH="${1:-${PROJECT_ROOT}/logs/dinov3_worldfloodsv2_2025-11-26_21-14-00/checkpoints/water_iou_0.9076.ckpt}"
DATA_ROOT="${2:-${PROJECT_ROOT}/data/WorldFloodsv2}"
OUT_DIR="${3:-scripts/infer/output/worldfloodsv2}"

mkdir -p "${OUT_DIR}"

echo "🚀 推理 WorldFloodsv2"

# 说明：默认禁用 AMP（混合精度）。原因：在该数据集上启用 AMP 会导致指标下降约 30 个百分点。
# 如需实验性启用，可在下方 python 命令末尾手动追加 --amp。
python scripts/infer/infer_worldfloodsv2.py \
  --checkpoint "${CKPT_PATH}" \
  --dataset-path "${DATA_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --num-workers 8

echo "结束时间: $(date)"
echo "✅ 完成：预测 GeoTIFF 按事件归档于 ${OUT_DIR}/predictions"


