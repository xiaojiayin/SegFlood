#!/usr/bin/env bash
# Table 8 three-seed completion:
# evaluate the seed-123 and seed-2026 checkpoints of the standard-protocol models
# (already trained for Table 5 / Table S1 / Table S3 / Table 2) under the four degraded
# conditions of Table 8, so that every row of Table 8 can be reported as mean +- SD.
# No new training. Degradation draws use test_degradation_seed=1 (same as draw d1 of S12).
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-123 2026}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="scripts/run/jag_supplement/submissions_table8_seeds_${STAMP}.tsv"
mkdir -p scripts/run/jag_supplement/logs/robustness_eval
printf "dataset\tmodel\tseed\tcondition\tjob_id\tckpt\n" > "${MANIFEST}"
L=${PROJECT_ROOT}/logs

S1S2="model.encoder.optical_channels=4 model.encoder.sar_channels=2 data.add_dem=false data.add_slope=false"
GF="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
BASEOFF="model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
MA="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
MAGF="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
NOALIGN="model.alignment_enabled=false model.alignment_target_weight=0.0"
INCAT="+data.output_image_key=true model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity +model.test_degradation_image_sar_start=4"

# model -> experiment | run-dir prefix (seed appended as _s<seed>_) | overrides | dual-only? (input concat has no single-modality path)
declare -A EXP DIR OVR SINGLE
EXP[s1s2_inputconcat]=resnet50_resnet50_s1s2water; DIR[s1s2_inputconcat]="jag26_alignfix_canonical_baselines_s1s2_canonical_inputconcat"; OVR[s1s2_inputconcat]="data.add_dem=false data.add_slope=false ${BASEOFF} ${INCAT}"; SINGLE[s1s2_inputconcat]=0
EXP[s1s2_dualadd]=resnet50_resnet50_s1s2water;     DIR[s1s2_dualadd]="jag26_alignfix_canonical_baselines_s1s2_canonical_dualadd";         OVR[s1s2_dualadd]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_add";   SINGLE[s1s2_dualadd]=1
EXP[s1s2_dualconcat]=resnet50_resnet50_s1s2water;  DIR[s1s2_dualconcat]="jag26_alignfix_canonical_baselines_s1s2_canonical_dualconcat";   OVR[s1s2_dualconcat]="${S1S2} ${BASEOFF} model.fusion.fusion_type=concat";       SINGLE[s1s2_dualconcat]=1
EXP[s1s2_stdcross]=resnet50_resnet50_s1s2water;    DIR[s1s2_stdcross]="jag26_alignfix_canonical_baselines_s1s2_canonical_stdcross";       OVR[s1s2_stdcross]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_cross"; SINGLE[s1s2_stdcross]=1
EXP[s1s2_gated]=resnet50_resnet50_s1s2water;       DIR[s1s2_gated]="jag26_alignfix_canonical_baselines_s1s2_canonical_gated";             OVR[s1s2_gated]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_gated";   SINGLE[s1s2_gated]=1
EXP[s1s2_mastd]=resnet50_resnet50_s1s2water;       DIR[s1s2_mastd]="jag26_alignfix_components_s1s2_gated";                                OVR[s1s2_mastd]="${S1S2} ${MA} ${NOALIGN}";                                       SINGLE[s1s2_mastd]=1
EXP[gf_dualconcat]=resnet50_resnet50_gffloodnet;   DIR[gf_dualconcat]="jag26_gft2_resnet50_dualconcat";                                   OVR[gf_dualconcat]="${GF} model.fusion.fusion_type=concat ${BASEOFF}";           SINGLE[gf_dualconcat]=1
EXP[gf_mastd]=resnet50_resnet50_gffloodnet;        DIR[gf_mastd]="jag26_gft2_resnet50_ma";                                                OVR[gf_mastd]="${GF} ${MAGF} ${NOALIGN}";                                         SINGLE[gf_mastd]=1

# condition -> mode degradation strength
declare -A COND=([clean]="dual none 0" [optical_cloud]="dual optical_cloud 0.3" [sar_noise]="dual sar_noise 0.5" [missing_optical]="sar_only none 0" [missing_sar]="optical_only none 0")

for m in s1s2_inputconcat s1s2_dualadd s1s2_dualconcat s1s2_stdcross s1s2_gated s1s2_mastd gf_dualconcat gf_mastd; do
  for s in ${SEEDS}; do
    d="$(ls -d "${L}/${DIR[$m]}_s${s}_"* 2>/dev/null | tail -1)"
    [[ -n "${d}" ]] || { echo "missing run dir for ${m} seed ${s}" >&2; exit 1; }
    ck="$(ls "${d}"/checkpoints/water_iou_*.ckpt 2>/dev/null | tail -1)"
    [[ -f "${ck}" ]] || { echo "missing ckpt in ${d}" >&2; exit 1; }
    for c in optical_cloud sar_noise missing_optical missing_sar; do
      if [[ "${SINGLE[$m]}" == "0" && ( "${c}" == "missing_optical" || "${c}" == "missing_sar" ) ]]; then continue; fi
      read -r mode deg st <<< "${COND[$c]}"
      name="jag26_t8seed_${m}_s${s}_${c}"
      ovr="${OVR[$m]} +model.test_degradation_seed=1"
      if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] ${name}: ${ck} | ${mode} ${deg} ${st}"; continue; fi
      jid="$(sbatch --exclude g59,g10 --time=01:30:00 --job-name "${name}" scripts/run/jag_supplement/eval_robustness.sh "${EXP[$m]}" "${ck}" "${name}" "${mode}" "${deg}" "${st}" "${ovr}" | awk '{print $4}')"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${m%%_*}" "${m}" "${s}" "${c}" "${jid}" "${ck}" >> "${MANIFEST}"
      echo "[submitted] ${name}: ${jid}"
    done
  done
done
echo "Manifest: ${MANIFEST}"
