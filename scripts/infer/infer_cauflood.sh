#!/bin/bash
#SBATCH --job-name=infer_cauflood
#SBATCH --output=scripts/infer/infer_cauflood_%j.out
#SBATCH --error=scripts/infer/infer_cauflood_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
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

# 进入项目根目录
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || PROJECT_ROOT="$PWD"
fi
cd "$PROJECT_ROOT" 2>/dev/null || true

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}" || true
conda activate segflood || true

CKPT="${1:-${PROJECT_ROOT}/logs/resnet50_resnet50_cauflood_2025-11-12_13-41-34/checkpoints/water_iou_0.8248.ckpt}"
DATA_ROOT="${2:-data/CAU-Flood}"
OUT_DIR="${3:-scripts/infer/output/cauflood}"
BS="${4:-128}"
NW="${5:-8}"

# 输出根
mkdir -p "$OUT_DIR"

# 说明：默认禁用 AMP（混合精度）。原因：在该数据集上启用 AMP 会导致指标下降约 30 个百分点。
# 如需实验性启用，可在下方 python 命令末尾手动追加 --amp。
python scripts/infer/infer_cauflood.py \
  --checkpoint "$CKPT" \
  --dataset-path "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  --batch-size "$BS" \
  --num-workers "$NW" \
  --save-predictions \
  --save-format auto

echo "✅ 完成：输出于 $OUT_DIR"


