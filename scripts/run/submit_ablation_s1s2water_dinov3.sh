#!/bin/bash

# S1S2-Water (DINOv3 dual-stream) alignment ablation submission script
#
# Base config: configs/experiment/dinov3_dinov3_s1s2water.yaml
# Variants:
#   1) alignment loss + align bias (full)
#   2) disable alignment loss
#   3) disable align bias
#   4) disable both
#
# Fixed settings:
#   - add_dem=false, add_slope=false
#
# Usage (login node):
#   bash scripts/run/submit_ablation_s1s2water_dinov3.sh

set -euo pipefail

EXP="dinov3_dinov3_s1s2water"
ADD_DEM="false"
ADD_SLOPE="false"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_s1s2water_dinov3_ablation_${STAMP}.txt"
mkdir -p scripts/run
echo "# S1S2-Water DINOv3 alignment ablation @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local variant="$1"          # text label
  local extra_overrides="$2"  # passed to train_s1s2water.sh arg #4 (EXTRA_OVERRIDES_STR)

  local cfg_name="${EXP}_${variant}_dem-${ADD_DEM}_slope-${ADD_SLOPE}"
  echo "[submit] ${cfg_name}"

  # Inject variant into experiment_name for easier log grouping
  local exp_name_override="experiment_name=${cfg_name}"
  local all_overrides="${extra_overrides} ${exp_name_override}"

  jid=$(sbatch \
    --job-name "${cfg_name}" \
    scripts/run/train_s1s2water.sh \
      "${EXP}" \
      "${ADD_DEM}" \
      "${ADD_SLOPE}" \
      "${all_overrides}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}" >> "${MANIFEST}"
}

#############################
# 1. Full model              #
#############################
submit_one "full_alignloss-on_bias-on" \
  "model.fusion.xattn_align_bias=true model.alignment_enabled=true"

###################################
# 2. Disable alignment loss         #
###################################
submit_one "no-align-loss_bias-on" \
  "model.alignment_enabled=false model.alignment_target_weight=0.0 model.fusion.xattn_align_bias=true"

################################
# 3. Disable align bias          #
################################
submit_one "alignloss-on_no-align-bias" \
  "model.fusion.xattn_align_bias=false model.alignment_enabled=true"

########################################
# 4. Disable both                       #
########################################
submit_one "no-align-loss_no-align-bias" \
  "model.alignment_enabled=false model.alignment_target_weight=0.0 model.fusion.xattn_align_bias=false"

echo "Manifest: ${MANIFEST}"
echo "Done. Submitted 4 jobs."



