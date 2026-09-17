#!/usr/bin/env bash

# Table 2 (GF-FloodNet backbone x fusion) completion under one protocol:
#   (a) SAM2 Concat / MA-XAttn retrained (seed42, min_epochs=50, component gate,
#       no agreement, test=true) to replace the legacy rows;
#   (b) test evaluation of the existing MobileNetV3 / EfficientNet-B4
#       min_epochs=50 checkpoints (Jobs 504713-716 were test=false).
#
#   DRY_RUN=1 bash scripts/run/jag_supplement/submit_gf_table2_fill.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="scripts/run/jag_supplement/logs/gf_table2_fill"
MANIFEST="scripts/run/jag_supplement/submissions_gf_table2_fill_${STAMP}.tsv"
mkdir -p "${LOG_ROOT}"
printf "variant\tjob_id\tkind\tdetail\n" > "${MANIFEST}"

HOMO="model.encoder.optical_channels=4 model.encoder.sar_channels=1 model.encoder.optical_pretrained=true model.encoder.sar_pretrained=false"
MA="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
CONCAT="model.fusion.fusion_type=concat model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"

sb() {
  local name="$1"; shift
  sbatch --exclude "${EXCLUDE_NODES}" --job-name "${name}" \
    --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" "$@" | awk '{print $4}'
}

submit_train() {
  local variant="$1" experiment="$2" overrides="$3"
  local name="jag26_gft2_${variant}_s42"
  local all="experiment_name=${name} seed=42 test=true trainer.min_epochs=50 ${overrides}"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] ${name}: ${experiment} ${all}"; return; fi
  local jid; jid="$(sb "${name}" scripts/run/train_experiment.sh "${experiment}" "" "${all}")"
  printf "%s\t%s\ttrain\t%s\n" "${variant}" "${jid}" "${all}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

submit_test() {
  local variant="$1" experiment="$2" ckpt="$3" overrides="$4"
  local name="jag26_gft2test_${variant}_s42"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] ${name}: ${ckpt}"; return; fi
  local jid; jid="$(sb "${name}" scripts/run/jag_supplement/eval_robustness.sh "${experiment}" "${PROJECT_ROOT}/${ckpt}" "${name}" dual none 0.0 "${overrides}")"
  printf "%s\t%s\ttest\t%s\n" "${variant}" "${jid}" "${ckpt}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

# (a) SAM2 retrain (hiera_base_plus optical + hiera_small SAR, as in the submitted manuscript)
submit_train sam2_dualconcat sam2_sam2_gffloodnet "${HOMO} ${CONCAT}"
submit_train sam2_ma         sam2_sam2_gffloodnet "${HOMO} ${MA}"

# (b) test the existing lightweight min_epochs=50 checkpoints
MB="model.encoder.optical_pretrained=false model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.sar_model_name=mobilenetv3_large_100"
EF="model.encoder.optical_model_name=efficientnet_b4 model.encoder.sar_model_name=efficientnet_b4"
B=logs/jag26_alignfix_gf_homogeneous_minep
submit_test mobilenet_dualconcat    efficientnetb4_mobilenetv3_gffloodnet "${B}_gf_mobilenet_dualconcat_s42_2026-09-02_13-49-07/checkpoints/water_iou_0.9520.ckpt"    "${HOMO} ${MB} ${CONCAT}"
submit_test mobilenet_ma            efficientnetb4_mobilenetv3_gffloodnet "${B}_gf_mobilenet_ma_s42_2026-09-02_13-49-07/checkpoints/water_iou_0.9473.ckpt"            "${HOMO} ${MB} ${MA}"
submit_test efficientnet_dualconcat efficientnetb4_mobilenetv3_gffloodnet "${B}_gf_efficientnet_dualconcat_s42_2026-09-02_13-49-07/checkpoints/water_iou_0.9667.ckpt" "${HOMO} ${EF} ${CONCAT}"
submit_test efficientnet_ma         efficientnetb4_mobilenetv3_gffloodnet "${B}_gf_efficientnet_ma_s42_2026-09-02_13-49-07/checkpoints/water_iou_0.9615.ckpt"         "${HOMO} ${EF} ${MA}"
echo "Manifest: ${MANIFEST}"
