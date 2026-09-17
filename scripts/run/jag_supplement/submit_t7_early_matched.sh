#!/usr/bin/env bash

# Table 7: capacity-matched early-fusion rows for the non-ResNet backbones.
#
# The 2025-11 *_early_fusion_s1s2water runs kept fusion_type=concat with a
# single stream, which routes features through a 1x1 projection to 2c and a
# 2c-wide UNet decoder (ResNet-50 Early: 199 M of 245 M params in the decoder).
# The MA-XAttn rows decode at c. These runs use the same definition as the
# Table 6 "Input concatenation" baseline (single encoder, identity fusion,
# c-wide decoder, aux off) and the *same optical backbone as the MA row*
# (SAM2 hiera_base_plus, DINOv3 ViT-B), so each Early/MA pair differs only in
# encoder count and fusion topology.
#
#   DRY_RUN=1 bash scripts/run/jag_supplement/submit_t7_early_matched.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="scripts/run/jag_supplement/logs/t7_early_matched"
MANIFEST="scripts/run/jag_supplement/submissions_t7_early_matched_${STAMP}.tsv"
mkdir -p "${LOG_ROOT}"
printf "variant\tjob_id\texperiment\toverrides\n" > "${MANIFEST}"

COMMON="seed=42 test=true trainer.min_epochs=50 data.add_dem=false data.add_slope=false +data.output_image_key=true model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0 model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity"

submit_train() {
  local variant="$1" experiment="$2" overrides="$3"
  local name="jag26_t7early_${variant}_s42"
  local all="experiment_name=${name} ${COMMON} ${overrides}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${name}: ${experiment} ${all}"; return
  fi
  local jid
  jid="$(sbatch --exclude "${EXCLUDE_NODES}" --job-name "${name}" \
    --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
    scripts/run/train_experiment.sh "${experiment}" "" "${all}" | awk '{print $4}')"
  printf "%s\t%s\t%s\t%s\n" "${variant}" "${jid}" "${experiment}" "${all}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

submit_train mobilenet efficientnetb4_mobilenetv3_s1s2water \
  "model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.optical_pretrained=false"
submit_train efficientnet efficientnetb4_mobilenetv3_s1s2water \
  "model.encoder.optical_model_name=efficientnet_b4"
submit_train sam2 sam2_sam2_s1s2water ""
submit_train dinov3 dinov3_dinov3_s1s2water ""
echo "Manifest: ${MANIFEST}"
