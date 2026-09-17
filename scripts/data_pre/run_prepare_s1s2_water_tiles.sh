#!/bin/bash
#SBATCH --job-name=tiles_s1s2
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --cpus-per-task=32
#SBATCH -o tiles_s1s2_%j.log
#SBATCH -e tiles_s1s2_%j.err
#SBATCH --no-requeue

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail

ROOT="${PROJECT_ROOT}/data/S1S2-Water"
SPLIT_JSON="${ROOT}/splits/s1s2_water_split_v1.json"
TILE_SIZE=256

# 仅生成一套 dual (S1+S2+Elev+Slope) 切片，直接覆盖 ROOT 下 train/val/test 目录；
# 保留 100% 有效像素的 tile，生成后不再另存 valid/ 目录（因为全部已有效）。
# 单模态实验时再在 Dataset 层面选择通道子集即可。

python scripts/data_pre/prepare_s1s2_water_tiles.py \
  --root "$ROOT" \
  --split-json "$SPLIT_JSON" \
  --out-root "$ROOT" \
  --sensor dual \
  --include-elevation \
  --include-slope \
  --exclude-nodata \
  --valid-threshold 1.0 \
  --tile-size ${TILE_SIZE} \
  --overwrite

echo "[DONE] Dual-modality tiles (100% valid) generated in $ROOT" 