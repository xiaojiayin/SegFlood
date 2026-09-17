#!/usr/bin/env bash
# Repeat the two stochastic degraded-input conditions of main-text Table 10 (30% optical
# occlusion, SAR noise 0.5 x SD) with fixed, model-shared degradation seeds (1 and 2), for
# every Table-10 model. The archived Table-10 values were produced with an unseeded RNG
# (equivalent to one further independent draw). Inference only -- no training.
# Usage: DRY_RUN=1 bash scripts/run/jag_supplement/submit_degradation_seed_repeats.sh
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-1 2}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="scripts/run/jag_supplement/submissions_degradation_seed_repeats_${STAMP}.tsv"
printf "dataset\tmodel\tcondition\tdeg_seed\tjob_id\tckpt\n" > "${MANIFEST}"
L=${PROJECT_ROOT}/logs

S1S2="model.encoder.optical_channels=4 model.encoder.sar_channels=2 data.add_dem=false data.add_slope=false"
GF="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
BASEOFF="model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
MA="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
MAGF="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
NOALIGN="model.alignment_enabled=false model.alignment_target_weight=0.0"
SYMALIGN="model.alignment_enabled=true model.alignment_mode=semantic_local +model.alignment_in_image_only=true +model.alignment_cap_mode=none +model.alignment_independent_projection=false model.alignment_layers=[-1,-2] model.alignment_target_weight=0.0005 +model.alignment_semantic_stop_gradient=true"
INCAT="+data.output_image_key=true model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity +model.test_degradation_image_sar_start=4"

declare -A EXP CKPT OVR
# ---- S1S2-Water (Table 10a) ----
EXP[s1s2_inputconcat]=resnet50_resnet50_s1s2water; CKPT[s1s2_inputconcat]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_inputconcat_s42_2026-09-01_16-02-54/checkpoints/water_iou_0.9703.ckpt"; OVR[s1s2_inputconcat]="data.add_dem=false data.add_slope=false ${BASEOFF} ${INCAT}"
EXP[s1s2_dualadd]=resnet50_resnet50_s1s2water;     CKPT[s1s2_dualadd]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_dualadd_s42_2026-09-01_16-02-54/checkpoints/water_iou_0.9696.ckpt";     OVR[s1s2_dualadd]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_add"
EXP[s1s2_dualconcat]=resnet50_resnet50_s1s2water;  CKPT[s1s2_dualconcat]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_dualconcat_s42_2026-09-01_16-02-52/checkpoints/water_iou_0.9702.ckpt";  OVR[s1s2_dualconcat]="${S1S2} ${BASEOFF} model.fusion.fusion_type=concat"
EXP[s1s2_stdcross]=resnet50_resnet50_s1s2water;    CKPT[s1s2_stdcross]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_stdcross_s42_2026-09-01_16-02-52/checkpoints/water_iou_0.9694.ckpt";    OVR[s1s2_stdcross]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_cross"
EXP[s1s2_gated]=resnet50_resnet50_s1s2water;       CKPT[s1s2_gated]="$L/jag26_alignfix_canonical_baselines_s1s2_canonical_gated_s42_2026-09-01_16-02-52/checkpoints/water_iou_0.9675.ckpt";       OVR[s1s2_gated]="${S1S2} ${BASEOFF} model.fusion.fusion_type=canonical_gated"
EXP[s1s2_mastd]=resnet50_resnet50_s1s2water;       CKPT[s1s2_mastd]="$L/jag26_alignfix_components_s1s2_gated_s42_2026-08-31_11-17-03/checkpoints/water_iou_0.9776.ckpt";                          OVR[s1s2_mastd]="${S1S2} ${MA} ${NOALIGN}"
EXP[s1s2_ma_s42]=resnet50_resnet50_s1s2water;      CKPT[s1s2_ma_s42]="$L/jag26_smagnet_s1s2_sym_align_minep50_s42_2026-09-08_11-25-20/checkpoints/last.ckpt";                                     OVR[s1s2_ma_s42]="${S1S2} ${MA} ${SYMALIGN}"
EXP[s1s2_ma_s123]=resnet50_resnet50_s1s2water;     CKPT[s1s2_ma_s123]="$L/jag26_smagnet_s1s2_sym_align_minep50_s123_2026-09-08_20-01-41/checkpoints/last.ckpt";                                   OVR[s1s2_ma_s123]="${S1S2} ${MA} ${SYMALIGN}"
EXP[s1s2_ma_s2026]=resnet50_resnet50_s1s2water;    CKPT[s1s2_ma_s2026]="$L/jag26_smagnet_s1s2_sym_align_minep50_s2026_2026-09-08_20-01-41/checkpoints/last.ckpt";                                 OVR[s1s2_ma_s2026]="${S1S2} ${MA} ${SYMALIGN}"
# ---- GF-FloodNet (Table 10b) ----
EXP[gf_dualconcat]=resnet50_resnet50_gffloodnet;   CKPT[gf_dualconcat]="$L/jag26_gft2_resnet50_dualconcat_s42_2026-09-07_11-48-45/checkpoints/water_iou_0.9668.ckpt"; OVR[gf_dualconcat]="${GF} model.fusion.fusion_type=concat ${BASEOFF}"
EXP[gf_mastd]=resnet50_resnet50_gffloodnet;        CKPT[gf_mastd]="$L/jag26_gft2_resnet50_ma_s42_2026-09-07_11-46-26/checkpoints/water_iou_0.9656.ckpt";             OVR[gf_mastd]="${GF} ${MAGF} ${NOALIGN}"
EXP[gf_ma_s42]=resnet50_resnet50_gffloodnet;       CKPT[gf_ma_s42]="$L/jag26_smagnet_gf_sym_align_minep50_s42_2026-09-08_20-14-28/checkpoints/last.ckpt";            OVR[gf_ma_s42]="${GF} ${MA} ${SYMALIGN}"

declare -A COND=([optical_cloud]="dual optical_cloud 0.3" [sar_noise]="dual sar_noise 0.5")
for m in s1s2_inputconcat s1s2_dualadd s1s2_dualconcat s1s2_stdcross s1s2_gated s1s2_mastd s1s2_ma_s42 s1s2_ma_s123 s1s2_ma_s2026 gf_dualconcat gf_mastd gf_ma_s42; do
  [[ -f "${CKPT[$m]}" ]] || { echo "missing ckpt for ${m}: ${CKPT[$m]}" >&2; exit 1; }
  for c in optical_cloud sar_noise; do
    read -r mode deg st <<< "${COND[$c]}"
    for ds in ${SEEDS}; do
      name="jag26_robseed_${m}_${c}_d${ds}"
      ovr="${OVR[$m]} +model.test_degradation_seed=${ds}"
      if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] ${name}: ${mode} ${deg} ${st} | ${ovr}"; continue; fi
      jid="$(sbatch --exclude g59,g10 --time=01:30:00 --job-name "${name}" scripts/run/jag_supplement/eval_robustness.sh "${EXP[$m]}" "${CKPT[$m]}" "${name}" "${mode}" "${deg}" "${st}" "${ovr}" | awk '{print $4}')"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${m%%_*}" "${m}" "${c}" "${ds}" "${jid}" "${CKPT[$m]}" >> "${MANIFEST}"
      echo "[submitted] ${name}: ${jid}"
    done
  done
done
echo "Manifest: ${MANIFEST}"
