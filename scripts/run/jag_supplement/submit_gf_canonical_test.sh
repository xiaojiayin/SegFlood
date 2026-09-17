#!/usr/bin/env bash
# Test-set evaluation of the GF-FloodNet six-method canonical baselines for seeds 123 and 2026
# (seed 42 was tested on 2026-09-02). Best-validation checkpoints; inference only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
STAMP="$(date +%Y%m%d_%H%M%S)"; MANIFEST="scripts/run/jag_supplement/submissions_gf_canonical_test_${STAMP}.tsv"
printf "method\tseed\tjob_id\tckpt\n" > "${MANIFEST}"
BASE="model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
DUAL="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
declare -A OVR=(
  [dualadd]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_add"
  [dualconcat]="${BASE} ${DUAL} model.fusion.fusion_type=concat"
  [stdcross]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_cross"
  [gated]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_gated"
  [inputconcat]="${BASE} +data.output_image_key=true model.encoder.optical_channels=5 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity"
)
SEEDS="${SEEDS:-123 2026}"
for m in dualadd dualconcat stdcross gated inputconcat; do for s in ${SEEDS}; do
  ck="$(ls logs/jag26_alignfix_canonical_baselines_gf_gf_canonical_${m}_s${s}_*/checkpoints/water_iou_*.ckpt | sort | tail -1)"
  name="jag26_gftest_${m}_s${s}"
  jid="$(sbatch --exclude g59,g10 --time=00:40:00 --job-name "${name}" scripts/run/jag_supplement/eval_robustness.sh resnet50_resnet50_gffloodnet "${PWD}/${ck}" "${name}" dual none 0.0 "${OVR[$m]}" | awk '{print $4}')"
  printf "%s\t%s\t%s\t%s\n" "${m}" "${s}" "${jid}" "${ck}" >> "${MANIFEST}"; echo "[submitted] ${name}: ${jid}"
done; done
echo "Manifest: ${MANIFEST}"
