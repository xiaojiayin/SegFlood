#!/usr/bin/env bash
# Evaluate existing S1S2-Water checkpoints under occlusion
# forms NOT seen in training: area 10 / 50 / 70 / 100 % (zero fill; 100 % = whole optical tile zero-filled
# and passed through the fusion path, i.e. the alternative missing-modality protocol). Inference only.
# In-range constant fills (0.6 / 0.05) are submitted by submit_ood2_inrange_fill_evals.sh.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
L=${PROJECT_ROOT}/logs
STAMP="$(date +%Y%m%d_%H%M%S)"; MANIFEST="scripts/run/jag_supplement/submissions_ood_occlusion_${STAMP}.tsv"
printf "model\tcondition\tjob_id\tckpt\n" > "${MANIFEST}"
S1S2="model.encoder.optical_channels=4 model.encoder.sar_channels=2 data.add_dem=false data.add_slope=false"
BASEOFF="model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
MA="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
NOALIGN="model.alignment_enabled=false model.alignment_target_weight=0.0"
SYMALIGN="model.alignment_enabled=true model.alignment_mode=semantic_local +model.alignment_in_image_only=true +model.alignment_cap_mode=none +model.alignment_independent_projection=false model.alignment_layers=[-1,-2] model.alignment_target_weight=0.0005 +model.alignment_semantic_stop_gradient=true"
INCAT="+data.output_image_key=true model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity +model.test_degradation_image_sar_start=4"
declare -A CKPT OVR
CKPT[inputconcat]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_inputconcat_s42_2026-09-01_16-02-54/checkpoints/water_iou_0.9703.ckpt"; OVR[inputconcat]="data.add_dem=false data.add_slope=false ${BASEOFF} ${INCAT}"
CKPT[add]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_dualadd_s42_2026-09-01_16-02-54/checkpoints/water_iou_0.9696.ckpt"; OVR[add]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_add"
CKPT[concat]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_dualconcat_s42_2026-09-01_16-02-52/checkpoints/water_iou_0.9702.ckpt"; OVR[concat]="${S1S2} ${BASEOFF} model.fusion.fusion_type=concat"
CKPT[bixattn]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_stdcross_s42_2026-09-01_16-02-52/checkpoints/water_iou_0.9694.ckpt"; OVR[bixattn]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_cross"
CKPT[gate]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_gated_s42_2026-09-01_16-02-52/checkpoints/water_iou_0.9675.ckpt"; OVR[gate]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_gated"
CKPT[mastd]="$L/jag26_alignfix_components_s1s2_gated_s42_2026-08-31_11-17-03/checkpoints/water_iou_0.9776.ckpt"; OVR[mastd]="${S1S2} ${MA} ${NOALIGN}"
CKPT[oa_s42]="$L/jag26_smagnet_s1s2_sym_align_minep50_s42_2026-09-08_11-25-20/checkpoints/last.ckpt"; OVR[oa_s42]="${S1S2} ${MA} ${SYMALIGN}"
CKPT[oa_s123]="$L/jag26_smagnet_s1s2_sym_align_minep50_s123_2026-09-08_20-01-41/checkpoints/last.ckpt"; OVR[oa_s123]="${S1S2} ${MA} ${SYMALIGN}"
CKPT[oa_s2026]="$L/jag26_smagnet_s1s2_sym_align_minep50_s2026_2026-09-08_20-01-41/checkpoints/last.ckpt"; OVR[oa_s2026]="${S1S2} ${MA} ${SYMALIGN}"
for s in 42 123 2026; do d="$(ls -d $L/jag26_iso_sym_noalign_s${s}_* | tail -1)"; CKPT[noalign_s${s}]="$d/checkpoints/last.ckpt"; OVR[noalign_s${s}]="${S1S2} ${MA} ${NOALIGN}"; done
# condition -> "strength fill"
declare -A COND=([occ10]="0.1 0" [occ50]="0.5 0" [occ70]="0.7 0" [occ100]="1.0 0")
for m in inputconcat add concat bixattn gate mastd oa_s42 oa_s123 oa_s2026 noalign_s42 noalign_s123 noalign_s2026; do
  [[ -f "${CKPT[$m]}" ]] || { echo "missing ckpt ${m}: ${CKPT[$m]}" >&2; exit 1; }
  for c in occ10 occ50 occ70 occ100; do
    read -r st fill <<< "${COND[$c]}"
    name="jag27_ood_${m}_${c}"
    ovr="${OVR[$m]} +model.test_degradation_seed=1 +model.test_degradation_fill=${fill}"
    jid="$(sbatch --exclude g59,g10 --time=01:30:00 --job-name "${name}" scripts/run/jag_supplement/eval_robustness.sh resnet50_resnet50_s1s2water "${CKPT[$m]}" "${name}" dual optical_cloud "${st}" "${ovr}" | awk '{print $4}')"
    printf "%s\t%s\t%s\t%s\n" "${m}" "${c}" "${jid}" "${CKPT[$m]}" >> "${MANIFEST}"
    echo "[submitted] ${name}: ${jid}"
  done
done
echo "Manifest: ${MANIFEST}"
