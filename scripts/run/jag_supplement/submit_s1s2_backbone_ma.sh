#!/usr/bin/env bash

# S1S2-Water: MA-XAttn (component gate, no agreement) per backbone family,
# so that Table 7 reports Early vs MA pairs like Tables 2 and 5.
#
#   E1  MobileNetV3 + MobileNetV3      (random init, as in the GF homogeneous runs)
#   E2  EfficientNet-B4 + EfficientNet-B4
#   E3  SAM2 Hiera-B+ + Hiera-S         (replaces the legacy Full row)
#   E4  DINOv3 ViT-B + ViT-S            (after the out_indices truncation fix)
#   E5  test of the three ResNet-50 component-gate seeds already trained
#
# Usage:
#   bash scripts/run/jag_supplement/submit_s1s2_backbone_ma.sh          # E1-E5
#   ONLY=train bash ...                                                # E1-E4
#   ONLY=test  bash ...                                                # E5
#   DRY_RUN=1 bash ...

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-all}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
PARTITION="${PARTITION:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="scripts/run/jag_supplement/logs/s1s2_backbone_ma"
MANIFEST="scripts/run/jag_supplement/submissions_s1s2_backbone_ma_${STAMP}.tsv"
mkdir -p "${LOG_ROOT}"
printf "variant\tjob_id\texperiment\toverrides\n" > "${MANIFEST}"

COMMON="seed=42 test=true trainer.min_epochs=50 data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2"
MA="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"

sbatch_extra=()
[[ -n "${EXCLUDE_NODES}" ]] && sbatch_extra+=(--exclude "${EXCLUDE_NODES}")
[[ -n "${PARTITION}" ]] && sbatch_extra+=(--partition "${PARTITION}")

submit_train() {
  local variant="$1" experiment="$2" overrides="$3"
  local name="jag26_s1s2bb_${variant}_s42"
  local all="experiment_name=${name} ${COMMON} ${overrides}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${name}: ${experiment} ${all}"
    printf "%s\tDRY_RUN\t%s\t%s\n" "${variant}" "${experiment}" "${all}" >> "${MANIFEST}"
    return
  fi
  local jid
  jid="$(sbatch "${sbatch_extra[@]}" --job-name "${name}" \
    --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
    scripts/run/train_experiment.sh "${experiment}" "" "${all}" | awk '{print $4}')"
  printf "%s\t%s\t%s\t%s\n" "${variant}" "${jid}" "${experiment}" "${all}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

submit_test() {
  local seed="$1" ckpt="$2"
  local name="jag26_s1s2bb_resnet50_ma_test_s${seed}"
  local ovr="data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2 ${MA}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${name}: ${ckpt}"
    printf "%s\tDRY_RUN\tresnet50_resnet50_s1s2water\t%s\n" "${name}" "${ckpt}" >> "${MANIFEST}"
    return
  fi
  local jid
  jid="$(sbatch "${sbatch_extra[@]}" --job-name "${name}" \
    --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
    scripts/run/jag_supplement/eval_robustness.sh resnet50_resnet50_s1s2water "${ckpt}" "${name}" dual none 0.0 "${ovr}" \
    | awk '{print $4}')"
  printf "%s\t%s\tresnet50_resnet50_s1s2water\t%s\n" "${name}" "${jid}" "${ckpt}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

if [[ "${ONLY}" == "all" || "${ONLY}" == "train" ]]; then
  submit_train mobilenet_ma efficientnetb4_mobilenetv3_s1s2water \
    "model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.sar_model_name=mobilenetv3_large_100 model.encoder.optical_pretrained=false model.encoder.sar_pretrained=false ${MA}"
  submit_train efficientnet_ma efficientnetb4_mobilenetv3_s1s2water \
    "model.encoder.optical_model_name=efficientnet_b4 model.encoder.sar_model_name=efficientnet_b4 model.encoder.sar_pretrained=false ${MA}"
  submit_train sam2_ma sam2_sam2_s1s2water "${MA}"
  submit_train dinov3_ma dinov3_dinov3_s1s2water "${MA}"
fi

if [[ "${ONLY}" == "all" || "${ONLY}" == "test" ]]; then
  submit_test 42   "${PROJECT_ROOT}/logs/jag26_alignfix_components_s1s2_gated_s42_2026-08-31_11-17-03/checkpoints/water_iou_0.9776.ckpt"
  submit_test 123  "${PROJECT_ROOT}/logs/jag26_alignfix_components_s1s2_gated_s123_2026-08-31_11-17-05/checkpoints/water_iou_0.9756.ckpt"
  submit_test 2026 "${PROJECT_ROOT}/logs/jag26_alignfix_components_s1s2_gated_s2026_2026-08-31_11-16-59/checkpoints/water_iou_0.9760.ckpt"
fi

echo "Manifest: ${MANIFEST}"
