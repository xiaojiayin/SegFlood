#!/usr/bin/env bash
# S1S2 degraded-condition evaluation of the four dual-stream Table-5 baselines (seed 42 best-val ckpt):
# missing optical (sar_only) / missing SAR (optical_only) / 30% optical cloud. Input concat cannot drop a modality (single encoder) -> cloud only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
BASE="data.add_dem=false data.add_slope=false model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
DUAL="model.encoder.optical_channels=4 model.encoder.sar_channels=2"
declare -A OVR=(
  [dualadd]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_add"
  [dualconcat]="${BASE} ${DUAL} model.fusion.fusion_type=concat"
  [stdcross]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_cross"
  [gated]="${BASE} ${DUAL} model.fusion.fusion_type=canonical_gated"
  [inputconcat]="${BASE} +data.output_image_key=true model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity"
)
declare -A COND=([sar_only]="sar_only none 0.5" [optical_only]="optical_only none 0.5" [optical_cloud]="dual optical_cloud 0.3")
for m in dualadd dualconcat stdcross gated inputconcat; do
  ck="$(ls logs/jag26_alignfix_canonical_baselines_s1s2_canonical_${m}_s42_*/checkpoints/water_iou_*.ckpt | sort | tail -1)"
  for c in sar_only optical_only optical_cloud; do
    [[ "${m}" == "inputconcat" && "${c}" != "optical_cloud" ]] && continue
    read -r mode deg st <<< "${COND[$c]}"
    name="jag26_rob_s1s2_${m}_${c}"
    sbatch --exclude g59,g10 --job-name "${name}" scripts/run/jag_supplement/eval_robustness.sh resnet50_resnet50_s1s2water "${PWD}/${ck}" "${name}" "${mode}" "${deg}" "${st}" "${OVR[$m]}" | awk -v n="${name}" '{print "[submitted] " n ": " $4}'
  done
done
