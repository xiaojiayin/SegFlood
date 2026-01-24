#!/bin/bash
#SBATCH --job-name=tiles_s1s2
#SBATCH -N 1
#SBATCH -p a01
#SBATCH --cpus-per-task=32
#SBATCH -o tiles_s1s2_%j.log
#SBATCH -e tiles_s1s2_%j.err
#SBATCH --no-requeue

set -euo pipefail

ROOT="${ROOT:-./data/S1S2-Water}"
SPLIT_JSON="${SPLIT_JSON:-${ROOT}/splits/s1s2_water_split_v1.json}"
TILE_SIZE=256

# Generate a dual-modality (S1+S2+Elev+Slope) tile set under ${ROOT}/{train,val,test}.
# We keep only tiles with 100% valid pixels (valid-threshold=1.0) and do not write a separate valid/ folder.
# For single-modality experiments, select a channel subset in the Dataset/DataModule.

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