#!/usr/bin/env bash

# Residual optical/SAR misregistration test. Inference only.
# Shift the SAR input by 0/1/2/4/8 pixels (labels and optical fixed) and
# evaluate matched fusion baselines and MA-XAttn variants on GF-FloodNet test.
# Usage: DRY_RUN=1 bash scripts/run/jag_supplement/submit_shift_eval_gf.sh

set -euo pipefail
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p scripts/run/jag_supplement/logs/robustness_eval
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="scripts/run/jag_supplement/submissions_shift_eval_gf_${STAMP}.tsv"
printf "model\tshift\tjob_id\tckpt\n" > "${MANIFEST}"

EXP="resnet50_resnet50_gffloodnet"
DUAL="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
CANON="model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
GATE="${DUAL} model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated"
NOALIGN="model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
FULL="model.fusion.xattn_align_bias=true +model.fusion.xattn_align_bias_mode=local_cross +model.fusion.xattn_align_bias_radius=1 model.fusion.xattn_align_bias_scale=1.0 +model.alignment_cap_mode=soft +model.alignment_warmup_epochs=5 +model.alignment_independent_projection=true +model.alignment_projection_dim=128 model.alignment_mode=semantic_local +model.alignment_exclude_boundaries=true model.alignment_patch_radius=1 model.alignment_target_weight=0.02 model.alignment_temperature=0.12 model.alignment_enabled=true"

declare -A CKPT OVR
CKPT[noalign]="logs/jag26_alignfix_components_gf_gated_s42_2026-08-31_11-17-04/checkpoints/water_iou_0.9656.ckpt"; OVR[noalign]="${GATE} ${NOALIGN}"
CKPT[full]="logs/jag26_alignfix_alignment_bias_final_gf_semantic_plus_bias_s42_2026-09-01_18-45-15/checkpoints/water_iou_0.9639.ckpt"; OVR[full]="${GATE} ${FULL}"
CKPT[inputconcat]="$(ls logs/jag26_alignfix_canonical_baselines_gf_gf_canonical_inputconcat_s42_*/checkpoints/water_iou_*.ckpt | tail -1)"; OVR[inputconcat]="${CANON} +data.output_image_key=true model.encoder.optical_channels=5 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity +model.test_degradation_image_sar_start=4"
CKPT[dualconcat]="$(ls logs/jag26_alignfix_canonical_baselines_gf_gf_canonical_dualconcat_s42_*/checkpoints/water_iou_*.ckpt | tail -1)"; OVR[dualconcat]="${CANON} ${DUAL} model.fusion.fusion_type=concat"
CKPT[stdcross]="$(ls logs/jag26_alignfix_canonical_baselines_gf_gf_canonical_stdcross_s42_*/checkpoints/water_iou_*.ckpt | tail -1)"; OVR[stdcross]="${CANON} ${DUAL} model.fusion.fusion_type=canonical_cross"
CKPT[gated]="$(ls logs/jag26_alignfix_canonical_baselines_gf_gf_canonical_gated_s42_*/checkpoints/water_iou_*.ckpt | tail -1)"; OVR[gated]="${CANON} ${DUAL} model.fusion.fusion_type=canonical_gated"

for m in ${MODELS:-inputconcat dualconcat stdcross gated noalign full}; do
  [[ -f "${CKPT[$m]}" ]] || { echo "missing ckpt for ${m}: ${CKPT[$m]}" >&2; exit 1; }
  for k in ${SHIFTS:-1 2 4 8}; do
    name="jag26_shift_${m}_px${k}"
    if [[ "${DRY_RUN}" == "1" ]]; then
      echo "[dry-run] ${name}: ${CKPT[$m]}"
      continue
    fi
    jid="$(sbatch --exclude "${EXCLUDE_NODES}" --job-name "${name}" \
      scripts/run/jag_supplement/eval_robustness.sh "${EXP}" "${CKPT[$m]}" "${name}" dual sar_shift "${k}" "${OVR[$m]}" \
      | awk '{print $4}')"
    printf "%s\t%s\t%s\t%s\n" "${m}" "${k}" "${jid}" "${CKPT[$m]}" >> "${MANIFEST}"
    echo "[submitted] ${name}: ${jid}"
  done
done
echo "Manifest: ${MANIFEST}"
