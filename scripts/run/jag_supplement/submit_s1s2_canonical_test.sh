#!/usr/bin/env bash

# Test-set evaluation of the S1S2 canonical fusion baselines (5 methods x 3
# seeds) from their best-validation checkpoints, so that Table 6 can report
# test metrics alongside the MA-XAttn component-gate test runs (505486-505488).
#
#   DRY_RUN=1 bash scripts/run/jag_supplement/submit_s1s2_canonical_test.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="scripts/run/jag_supplement/logs/s1s2_canonical_test"
MANIFEST="scripts/run/jag_supplement/submissions_s1s2_canonical_test_${STAMP}.tsv"
mkdir -p "${LOG_ROOT}"
printf "variant\tseed\tjob_id\tckpt\n" > "${MANIFEST}"

BASE="data.add_dem=false data.add_slope=false model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
DUAL="model.encoder.optical_channels=4 model.encoder.sar_channels=2"

declare -A OVR=(
  [inputconcat]="${BASE} +data.output_image_key=true model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity"
  [dualadd]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_add"
  [dualconcat]="${BASE} ${DUAL} model.fusion.fusion_type=concat"
  [stdcross]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_cross"
  [gated]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_gated"
)

for variant in inputconcat dualadd dualconcat stdcross gated; do
  for seed in 42 123 2026; do
    dir="$(ls -d logs/jag26_alignfix_canonical_baselines_s1s2_canonical_${variant}_s${seed}_* | tail -1)"
    ckpt="$(ls "${dir}"/checkpoints/water_iou_*.ckpt | tail -1)"
    name="jag26_s1s2test_${variant}_s${seed}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] ${name}: ${ckpt}"
      printf "%s\t%s\tDRY_RUN\t%s\n" "${variant}" "${seed}" "${ckpt}" >> "${MANIFEST}"
      continue
    fi
    jid="$(sbatch --exclude "${EXCLUDE_NODES}" --job-name "${name}" \
      --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
      scripts/run/jag_supplement/eval_robustness.sh resnet50_resnet50_s1s2water \
      "${PROJECT_ROOT}/${ckpt}" "${name}" dual none 0.0 "${OVR[${variant}]}" | awk '{print $4}')"
    printf "%s\t%s\t%s\t%s\n" "${variant}" "${seed}" "${jid}" "${ckpt}" >> "${MANIFEST}"
    echo "[submitted] ${name}: ${jid}"
  done
done
echo "Manifest: ${MANIFEST}"
