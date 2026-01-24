#!/bin/bash

# Batch-submit KuroSiwo experiments across different backbones.
# Fixed settings:
#   - scale_input=db, dem=false, use_ratio=false
#
# Usage:
#   bash scripts/run/submit_batch_kurosiwo_backbones.sh

set -euo pipefail

EXP_DINO="dinov3_kurosiwo"
EXP_SAM2="sam2_kurosiwo"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_kurosiwo_backbones_${STAMP}.txt"
mkdir -p scripts/run
echo "# KuroSiwo backbone submissions @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local exp="$1"              # experiment name (dinov3_kurosiwo / sam2_kurosiwo)
  local backbone="$2"         # backbone model name (for job-name)
  local extra_overrides="$3"  # passed to train_kurosiwo.sh arg #7 (EXTRA_OVERRIDES_STR)

  local cfg_name="${exp}_${backbone}_dem-false_scale-db"
  echo "[submit] ${cfg_name}"

  # Fixed: dem=false, dem_scale=zscore, scale_input=db, use_ratio=false, clear_db=false
  jid=$(sbatch \
    --job-name "${cfg_name}" \
    scripts/run/train_kurosiwo.sh \
      "${exp}" \
      "false" \
      "zscore" \
      "db" \
      "false" \
      "false" \
      "${extra_overrides}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}" >> "${MANIFEST}"
}

########################
# 1) DINOv3 backbones   #
########################

# vit_small_patch16_dinov3
submit_one "${EXP_DINO}" "vit_small_patch16_dinov3" \
  "model.encoder.optical_model_name=vit_small_patch16_dinov3 model.encoder.sar_model_name=vit_small_patch16_dinov3"

# vit_small_plus_patch16_dinov3
submit_one "${EXP_DINO}" "vit_small_plus_patch16_dinov3" \
  "model.encoder.optical_model_name=vit_small_plus_patch16_dinov3 model.encoder.sar_model_name=vit_small_plus_patch16_dinov3"

# vit_base_patch16_dinov3
submit_one "${EXP_DINO}" "vit_base_patch16_dinov3" \
  "model.encoder.optical_model_name=vit_base_patch16_dinov3 model.encoder.sar_model_name=vit_base_patch16_dinov3"

# vit_base_patch16_dinov3_qkvb
submit_one "${EXP_DINO}" "vit_base_patch16_dinov3_qkvb" \
  "model.encoder.optical_model_name=vit_base_patch16_dinov3_qkvb model.encoder.sar_model_name=vit_base_patch16_dinov3_qkvb"

# vit_large_patch16_dinov3
submit_one "${EXP_DINO}" "vit_large_patch16_dinov3" \
  "model.encoder.optical_model_name=vit_large_patch16_dinov3 model.encoder.sar_model_name=vit_large_patch16_dinov3"

# vit_large_patch16_dinov3.sat493m
submit_one "${EXP_DINO}" "vit_large_patch16_dinov3.sat493m" \
  "model.encoder.optical_model_name=vit_large_patch16_dinov3.sat493m model.encoder.sar_model_name=vit_large_patch16_dinov3.sat493m"

##########################
# 2) SAM2-Hiera backbones #
##########################

# sam2_hiera_tiny
submit_one "${EXP_SAM2}" "sam2_hiera_tiny" \
  "model.encoder.optical_model_name=sam2_hiera_tiny model.encoder.sar_model_name=sam2_hiera_tiny"

# sam2_hiera_small
submit_one "${EXP_SAM2}" "sam2_hiera_small" \
  "model.encoder.optical_model_name=sam2_hiera_small model.encoder.sar_model_name=sam2_hiera_small"

# sam2_hiera_base_plus
submit_one "${EXP_SAM2}" "sam2_hiera_base_plus" \
  "model.encoder.optical_model_name=sam2_hiera_base_plus model.encoder.sar_model_name=sam2_hiera_base_plus"

# sam2_hiera_large
submit_one "${EXP_SAM2}" "sam2_hiera_large" \
  "model.encoder.optical_model_name=sam2_hiera_large model.encoder.sar_model_name=sam2_hiera_large"

echo "Manifest: ${MANIFEST}"
echo "Done. Submitted 10 jobs."

