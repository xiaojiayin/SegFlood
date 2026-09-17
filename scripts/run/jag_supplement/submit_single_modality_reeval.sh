#!/usr/bin/env bash

# Re-evaluate selected existing checkpoints so Kuro Siwo and WorldFloods v2
# have complete, traceable test tables in the supplementary material.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${PROJECT_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="scripts/run/jag_supplement/logs/single_modality"
MANIFEST="scripts/run/jag_supplement/submissions_single_modality_${STAMP}.tsv"
mkdir -p "${LOG_DIR}"
printf "dataset\texperiment\tcheckpoint\tjob_id\n" > "${MANIFEST}"

submit_eval() {
  local dataset="$1"
  local exp="$2"
  local ckpt="$3"
  local name="jag26_eval_${exp}"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[missing checkpoint] ${ckpt}" >&2
    printf "%s\t%s\t%s\t%s\n" "${dataset}" "${exp}" "${ckpt}" "MISSING" >> "${MANIFEST}"
    return
  fi

  local jid
  jid="$(
    sbatch \
      --job-name "${name}" \
      --output "${LOG_DIR}/${name}_%j.out" \
      --error "${LOG_DIR}/${name}_%j.err" \
      scripts/run/eval_checkpoint.sh "${exp}" "${ckpt}" "${name}" \
      | awk '{print $4}'
  )"
  printf "%s\t%s\t%s\t%s\n" "${dataset}" "${exp}" "${ckpt}" "${jid}" >> "${MANIFEST}"
  echo "[submitted] ${exp}: ${jid}"
}

submit_eval "KuroSiwo" "mobilenetv3_kurosiwo" \
  "${PROJECT_ROOT}/logs/mobilenetv3_kurosiwo_2025-11-25_23-29-35/checkpoints/water_iou_0.6954.ckpt"
submit_eval "KuroSiwo" "efficientnetb4_kurosiwo" \
  "${PROJECT_ROOT}/logs/efficientnetb4_kurosiwo_2025-11-25_23-29-35/checkpoints/water_iou_0.6919.ckpt"
submit_eval "KuroSiwo" "resnet50_kurosiwo" \
  "${PROJECT_ROOT}/logs/resnet50_kurosiwo_2025-11-25_23-29-34/checkpoints/water_iou_0.7122.ckpt"
submit_eval "KuroSiwo" "dinov3_kurosiwo" \
  "${PROJECT_ROOT}/logs/dinov3_kurosiwo_2025-11-25_23-29-25/checkpoints/water_iou_0.7155.ckpt"
submit_eval "KuroSiwo" "sam2_kurosiwo" \
  "${PROJECT_ROOT}/logs/sam2_kurosiwo_2025-11-25_23-29-34/checkpoints/water_iou_0.7060.ckpt"

submit_eval "WorldFloodsv2" "mobilenetv3_worldfloodsv2" \
  "${PROJECT_ROOT}/logs/mobilenetv3_worldfloodsv2_2025-11-26_01-04-13/checkpoints/water_iou_0.9031.ckpt"
submit_eval "WorldFloodsv2" "efficientnetb4_worldfloodsv2" \
  "${PROJECT_ROOT}/logs/efficientnetb4_worldfloodsv2_2025-11-05_19-54-55/checkpoints/water_iou_0.9013.ckpt"
submit_eval "WorldFloodsv2" "dinov3_worldfloodsv2" \
  "${PROJECT_ROOT}/logs/dinov3_worldfloodsv2_2025-11-26_21-13-59/checkpoints/water_iou_0.9088.ckpt"

echo "Manifest: ${MANIFEST}"
