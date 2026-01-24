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

set -euo pipefail

echo "Job: ${SLURM_JOB_ID:-N/A} (array=${SLURM_ARRAY_TASK_ID:-N/A})"
echo "Start: $(date)"

###############################################################################
# Runtime configuration (override via env vars if needed)
# - PROJECT_ROOT: repo root (default: inferred from this script location)
# - CONDA_ENV: conda env name to activate (optional; otherwise activate before sbatch)
# - HF_ENDPOINT / HF_TOKEN: Hugging Face settings (optional)
# - HF_HUB_CACHE / TORCH_HOME: caches (optional; defaults under $PROJECT_ROOT/checkpoints)
###############################################################################

# Repo root (infer from this script path)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

: "${HF_ENDPOINT:=}"
: "${HF_TOKEN:=}"
: "${HF_HUB_CACHE:=${PROJECT_ROOT}/checkpoints/.cache}"
: "${TORCH_HOME:=${PROJECT_ROOT}/checkpoints}"
export HF_ENDPOINT HF_TOKEN HF_HUB_CACHE TORCH_HOME
export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
mkdir -p "${HF_HUB_CACHE}" "${TORCH_HOME}" || true

# Optional conda activation
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda activate "${CONDA_ENV}" || true
  fi
fi

echo "Python: $(python --version 2>&1)"

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits || true

CKPTS=(
  "${PROJECT_ROOT}/logs/<your_gffloodnet_run>/checkpoints/last.ckpt"
  "/path/to/<your_cauflood_run>/checkpoints/best.ckpt"
  "/path/to/<your_kurosiwo_run>/checkpoints/best.ckpt"
  "/path/to/<your_s1s2water_run>/checkpoints/best.ckpt"
  "/path/to/<your_worldfloodsv2_run>/checkpoints/best.ckpt"
)

# Global note: AMP is disabled by default in these scripts. If you want to enable AMP,
# append --amp to the corresponding python command below (if supported).
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
# Data is stored under the old repo path on this server.
DATA_ROOT_BASE="${DATA_ROOT_BASE:-${PROJECT_ROOT}/data}"
DATASET_PATHS=(
  "${DATA_ROOT_BASE}/GF-FloodNet"
  "${DATA_ROOT_BASE}/CAU-Flood"
  "${DATA_ROOT_BASE}/KuroSiwoGRD"
  "${DATA_ROOT_BASE}/S1S2-Water"
  "${DATA_ROOT_BASE}/WorldFloodsv2"
)
# Per-dataset batch/worker
BATCH_SIZES=(
  128  # GF-FloodNet (small tiles)
  128  # CAU-Flood (small tiles)
  128  # KuroSiwo (sliding-window batches)
  128  # S1S2-Water (tile inference)
  128  # WorldFloodsv2
)
NUM_WORKERS=(
  8
  8
  8
  8
  8
)

# Output base (match your current layout)
pick_output_base() {
  local cand
  for cand in \
    "${OUT_ROOT:-}" \
    "${SLURM_SUBMIT_DIR:-}/scripts/infer/output" \
    "$PROJECT_ROOT/scripts/infer/output" \
    "$HOME/Segflood_outputs"
  do
    if [[ -n "$cand" ]]; then
      mkdir -p "$cand" 2>/dev/null && echo "$cand" && return 0
    fi
  done
  # Fallback: current directory
  echo "$PWD"
}
OUTPUT_BASE="$(pick_output_base)"
echo "OUTPUT_BASE: $OUTPUT_BASE"

# Array index
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

  echo ""
  echo "============================================================"
  echo "Run:       $MODEL_NAME"
  echo "Checkpoint:$CKPT"
  echo "Dataset:   $DATASET_NAME -> $DATASET_PATH"
  echo "Output:    $OUT_DIR"
  echo "Workers:   $NW   BatchSize: $BS"
  echo "============================================================"

# Dispatch to per-dataset entry
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

echo "End: $(date)"
echo "Done."


