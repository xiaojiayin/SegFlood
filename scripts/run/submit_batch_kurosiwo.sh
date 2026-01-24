#!/bin/bash

# Batch-submit KuroSiwo experiments (based on sam2_kurosiwo)
#
# Usage:
#   bash scripts/run/submit_batch_kurosiwo.sh
#
# Notes:
#   - Calls scripts/run/train_kurosiwo.sh
#   - EXP is fixed to sam2_kurosiwo
#   - Submits these variants:
#       1) linear domain + z-score (SAR only)
#       2) dB domain + z-score + vh/vv
#       3) dB domain + z-score (SAR only)
#       4) dB domain + DEM(z-score)
#       5) dB domain + DEM(no normalization)
#       6) pure dB (no z-score; SAR only)

set -euo pipefail

EXP="sam2_kurosiwo"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_kurosiwo_${STAMP}.txt"
mkdir -p scripts/run
echo "# KuroSiwo submissions @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local use_dem="$1"
  local dem_scale="$2"
  local scale_input="$3"
  local use_ratio="$4"
  local clear_db="$5"

  local cfg_name="${EXP}_dem-${use_dem}_demscale-${dem_scale}_scale-${scale_input}_ratio-${use_ratio}_cleardb-${clear_db}"
  echo "[submit] ${cfg_name}"

  jid=$(sbatch \
    --job-name "${cfg_name}" \
    scripts/run/train_kurosiwo.sh \
      "${EXP}" \
      "${use_dem}" \
      "${dem_scale}" \
      "${scale_input}" \
      "${use_ratio}" \
      "${clear_db}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}" >> "${MANIFEST}"
}

# 1) linear domain + z-score (SAR only)
submit_one "false" "zscore" "normalize" "false" "false"

# 2) dB domain + z-score + vh/vv
submit_one "false" "zscore" "db" "true" "false"

# 3) dB domain + z-score (SAR only)
submit_one "false" "zscore" "db" "false" "false"

# 4) dB domain + DEM(z-score)
submit_one "true" "zscore" "db" "false" "false"

# 5) dB domain + DEM(no normalization)
submit_one "true" "none" "db" "false" "false"

# 6) pure dB (no z-score; SAR only)
submit_one "false" "zscore" "db" "false" "true"

echo "Manifest: ${MANIFEST}"
echo "Done. Submitted 6 jobs."


