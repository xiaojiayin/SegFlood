#!/bin/bash
#SBATCH --job-name=infer_eval_array
#SBATCH --output=scripts/infer/infer_and_eval_%A_%a.out
#SBATCH --error=scripts/infer/infer_and_eval_%A_%a.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-4

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail

echo "作业ID: ${SLURM_JOB_ID:-N/A}, 数组任务: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "开始时间: $(date)"

echo "🔧 设置环境..."
export HF_ENDPOINT="https://hf-mirror.com"

export HF_TOKEN="${HF_TOKEN:-}"  # set your Hugging Face token in the environment if gated weights are needed
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood
echo "Python: $(python --version)"

echo "🚀 GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits || true

# 全局说明：数组脚本默认禁用 AMP（混合精度）。原因：在多数据集上开启 AMP 会导致指标下降约 30 个百分点。
# 如需实验性启用，可在对应数据集分支的 python 命令末尾手动追加 --amp。

# 进入项目根目录（优先 SLURM_SUBMIT_DIR；否则相对脚本路径；失败则保持当前目录）
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}" ]]; then
  PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
  PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || PROJECT_ROOT="$PWD"
fi
cd "$PROJECT_ROOT" 2>/dev/null || true

# 模型与数据集（5个并行数组任务；每个任务独占1张GPU）
CKPTS=(
  "${PROJECT_ROOT}/logs/resnet50_resnet50_gffloodnet_2025-11-26_17-24-35/checkpoints/water_iou_0.9654.ckpt"
  "${PROJECT_ROOT}/logs/resnet50_resnet50_cauflood_2025-11-12_13-41-34/checkpoints/water_iou_0.8248.ckpt"
  "${PROJECT_ROOT}/logs/sam2_kurosiwo_dem-false_demscale-zscore_scale-db_ratio-true_cleardb-false_2025-11-25_14-53-29/checkpoints/water_iou_0.6604.ckpt"
  "${PROJECT_ROOT}/logs/dinov3_dinov3_s1s2water_dem-false_slope-false_2025-11-24_14-35-40/checkpoints/water_iou_0.9742.ckpt"
  "${PROJECT_ROOT}/logs/dinov3_worldfloodsv2_2025-11-26_21-14-00/checkpoints/water_iou_0.9076.ckpt"
)
MODEL_NAMES=(
  "resnet50_resnet50_gffloodnet_best"
  "resnet50_resnet50_cauflood_best"
  "sam2_kurosiwo_dem-false_demscale-zscore_scale-db_ratio-true_cleardb-false"
  "dinov3_dinov3_s1s2water_dem-false_slope-false"
  "dinov3_worldfloodsv2"
)
DATASET_NAMES=(
  "GF-FloodNet"
  "CAU-Flood"
  "KuroSiwo"
  "S1S2-Water"
  "WorldFloodsv2"
)
DATASET_PATHS=(
  "data/GF-FloodNet"
  "data/CAU-Flood"
  "data/KuroSiwoGRD"
  "data/S1S2-Water"
  "data/WorldFloodsv2"
)
# 数据集特定 batch/worker
BATCH_SIZES=(
  128  # GF-FloodNet（小图）
  128  # CAU-Flood（小图）
  128  # KuroSiwo（滑窗批处理）
  128  # S1S2-Water（tile推理）
  128  # WorldFloodsv2（predict阶段内部仍按1处理，保持一致）
)
NUM_WORKERS=(
  8
  8
  8
  8
  8
)

# 选择可写的输出根目录（优先：用户环境变量，其次提交目录，其次项目内，最后HOME）
pick_output_base() {
  local cand
  for cand in \
    "${OUT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}/scripts/infer/output" \
    "$PROJECT_ROOT/scripts/infer/output" \
    "$HOME/SegFlood_outputs"
  do
    if [[ -n "$cand" ]]; then
      mkdir -p "$cand" 2>/dev/null && echo "$cand" && return 0
    fi
  done
  # 若全部失败，使用当前目录（可能失败，但至少不中断脚本展开）
  echo "$PWD"
}
OUTPUT_BASE="$(pick_output_base)"
echo "🔖 OUTPUT_BASE: $OUTPUT_BASE"

# 选择数组索引
IDX="${SLURM_ARRAY_TASK_ID:-0}"
CKPT="${CKPTS[$IDX]}"
MODEL_NAME="${MODEL_NAMES[$IDX]}"
DATASET_NAME="${DATASET_NAMES[$IDX]}"
DATASET_PATH="${DATASET_PATHS[$IDX]}"
BS="${BATCH_SIZES[$IDX]}"
NW="${NUM_WORKERS[$IDX]}"

DATASET_SLUGS=(
  "gffloodnet"
  "cauflood"
  "kurosiwo"
  "s1s2water"
  "worldfloodsv2"
)
DATASET_SLUG="${DATASET_SLUGS[$IDX]}"
OUT_DIR="$OUTPUT_BASE/$DATASET_SLUG"
  mkdir -p "$OUT_DIR"

  echo "\n============================================================"
  echo "🚀 推理评估: $MODEL_NAME"
  echo "📁 Checkpoint: $CKPT"
  echo "🗂️ Dataset: $DATASET_NAME -> $DATASET_PATH"
  echo "📤 Output: $OUT_DIR"
  echo "🧰 BatchSize=$BS, NumWorkers=$NW"
  echo "============================================================"

# 调用对应数据集独立入口
case "$DATASET_NAME" in
  "GF-FloodNet")
    python scripts/infer/infer_gffloodnet.py \
      --checkpoint "$CKPT" \
      --dataset-path "$DATASET_PATH" \
      --output-dir "$OUT_DIR" \
      --batch-size "$BS" --num-workers "$NW" --save-predictions --save-format auto
    ;;
  "CAU-Flood")
    python scripts/infer/infer_cauflood.py \
      --checkpoint "$CKPT" \
      --dataset-path "$DATASET_PATH" \
      --output-dir "$OUT_DIR" \
      --batch-size "$BS" --num-workers "$NW" --save-predictions --save-format auto
    ;;
  "KuroSiwo")
    python scripts/infer/infer_kurosiwo.py \
      --checkpoint "$CKPT" \
      --root "$DATASET_PATH" \
      --out_dir "$OUT_DIR" \
      --events all \
      --batch_size "$BS" \
      --tile_size 256 \
      --overlap 64 \
      --mosaic_scale linear \
      --zscore \
      --use_ratio
    ;;
  "S1S2-Water")
    python scripts/infer/infer_s1s2water.py \
      --checkpoint "$CKPT" \
      --data-root "$DATASET_PATH" \
      --out-dir "$OUT_DIR" \
      --modal-type dual \
      --batch-size "$BS" \
      --num-workers "$NW"
    ;;
  "WorldFloodsv2")
    python scripts/infer/infer_worldfloodsv2.py \
    --checkpoint "$CKPT" \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUT_DIR" \
      --num-workers "$NW"
    ;;
  *)
    echo "[ERROR] Unsupported dataset in this batch script: $DATASET_NAME"
    exit 1
    ;;
esac

echo "结束时间: $(date)"
echo "✅ 推理评估完成"


