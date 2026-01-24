#!/bin/bash

# GF-FloodNet (ResNet50 dual-stream) alignment ablation submission script
#
# Base config: configs/experiment/resnet50_resnet50_gffloodnet.yaml
# It enables:
#   - cross-attention fusion (fusion_type=xattn)
#   - alignment regularization (SigLIP-style PatchNCE)
#
# This script submits 4 variants:
#   1) full_alignloss-on_bias-on
#      - xattn + align bias + alignment loss (alignment_enabled=true)
#   2) no-align-loss_bias-on
#      - disable alignment loss; keep align bias
#   3) alignloss-on_no-align-bias
#      - keep alignment loss; disable align bias
#   4) no-align-loss_no-align-bias
#      - disable both (dual-stream baseline without explicit alignment)
#
# Usage (login node):
#   bash scripts/run/submit_ablation_resnet50_resnet50_gffloodnet.sh

set -euo pipefail

EXP="resnet50_resnet50_gffloodnet"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_resnet50_resnet50_gffloodnet_ablation_${STAMP}.txt"
mkdir -p scripts/run
echo "# GF-FloodNet ResNet50 alignment ablation @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local variant="$1"          # text label
  local extra_overrides="$2"  # passed to train_gffloodnet.sh arg #2 (EXTRA_OVERRIDES_STR)

  local cfg_name="${EXP}_${variant}"
  echo "[submit] ${cfg_name}"

  # Inject variant into experiment_name for easier log/ckpt grouping
  local exp_name_override="experiment_name=${cfg_name}"
  local all_overrides="${extra_overrides} ${exp_name_override}"

  jid=$(sbatch \
    --job-name "${cfg_name}" \
    scripts/run/train_gffloodnet.sh \
      "${EXP}" \
      "${all_overrides}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}" >> "${MANIFEST}"
}

############################################
# 1. full_alignloss-on_bias-on (full model) #
############################################
# Keep xattn + align bias + alignment_enabled=true
submit_one "full_alignloss-on_bias-on" \
  "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=true model.alignment_enabled=true"

#######################################
# 2. no-align-loss_bias-on             #
#######################################
# Disable alignment loss; keep align bias
submit_one "no-align-loss_bias-on" \
  "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=true model.alignment_enabled=false model.alignment_target_weight=0.0"

##########################################
# 3. alignloss-on_no-align-bias           #
##########################################
# Keep alignment loss; disable align bias
submit_one "alignloss-on_no-align-bias" \
  "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=false model.alignment_enabled=true"

#############################################
# 4. no-align-loss_no-align-bias             #
#############################################
# Disable both -> baseline without explicit alignment
submit_one "no-align-loss_no-align-bias" \
  "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"

echo "Manifest: ${MANIFEST}"
echo "Done. Submitted 4 jobs."



