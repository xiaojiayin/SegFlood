#!/usr/bin/env bash

# Test-set evaluation of every Table 9 checkpoint (best-validation Water IoU)
# so that Table 9 reports test metrics like Table 6, and per-image confusion
# CSVs are available for paired bootstrap.
#
#   Group A (3 seeds): spatial / channel / fixed sum / reverse order / component gate
#   Group B (seed42; loss-only has 3 seeds): loss-only / bias-only / loss+bias
#
#   DRY_RUN=1 bash scripts/run/jag_supplement/submit_table9_test.sh
#   DATASETS="s1s2" bash scripts/run/jag_supplement/submit_table9_test.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
DATASETS="${DATASETS:-s1s2 gf}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="scripts/run/jag_supplement/logs/table9_test"
MANIFEST="scripts/run/jag_supplement/submissions_table9_test_${STAMP}.tsv"
mkdir -p "${LOG_ROOT}"
printf "dataset\tvariant\tseed\tjob_id\tckpt\n" > "${MANIFEST}"

MA_BASE="model.fusion.fusion_type=xattn +model.fusion.xattn_attention_type=ma"
NOALIGN="model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
GATE="+model.fusion.xattn_component_fusion=gated"
INDEPENDENT="+model.alignment_cap_mode=soft +model.alignment_warmup_epochs=5 +model.alignment_independent_projection=true +model.alignment_projection_dim=128"
SEM="${INDEPENDENT} model.alignment_mode=semantic_local +model.alignment_exclude_boundaries=true model.alignment_patch_radius=1 model.alignment_target_weight=0.02 model.alignment_temperature=0.12 model.alignment_enabled=true"
BIAS="model.fusion.xattn_align_bias=true +model.fusion.xattn_align_bias_mode=local_cross +model.fusion.xattn_align_bias_radius=1 model.fusion.xattn_align_bias_scale=1.0"

declare -A OVR DIRPAT SEEDS
OVR[spatial]="${MA_BASE} ${NOALIGN} +model.fusion.xattn_components=spatial +model.fusion.xattn_affinity_order=opt_sar";               DIRPAT[spatial]="components_DS_spatial";  SEEDS[spatial]="42 123 2026"
OVR[channel]="${MA_BASE} ${NOALIGN} +model.fusion.xattn_components=channel +model.fusion.xattn_affinity_order=opt_sar";               DIRPAT[channel]="components_DS_channel";  SEEDS[channel]="42 123 2026"
OVR[sum]="${MA_BASE} ${NOALIGN} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=opt_sar";           DIRPAT[sum]="components_DS_sum";          SEEDS[sum]="42 123 2026"
OVR[reverse]="${MA_BASE} ${NOALIGN} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=sar_opt";       DIRPAT[reverse]="components_DS_reverse";  SEEDS[reverse]="42 123 2026"
OVR[gated]="${MA_BASE} ${NOALIGN} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=opt_sar ${GATE}"; DIRPAT[gated]="components_DS_gated";      SEEDS[gated]="42 123 2026"
OVR[loss_only]="${MA_BASE} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=opt_sar ${GATE} model.fusion.xattn_align_bias=false ${SEM} +model.alignment_stop_gradient=true"; DIRPAT[loss_only]="final_selected_DS_semantic_selected"; SEEDS[loss_only]="42 123 2026"
OVR[bias_only]="${MA_BASE} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=opt_sar ${GATE} ${BIAS} model.alignment_enabled=false model.alignment_target_weight=0.0"; DIRPAT[bias_only]="alignment_bias_final_DS_bias_only"; SEEDS[bias_only]="42"
OVR[full]="${MA_BASE} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=opt_sar ${GATE} ${BIAS} ${SEM}"; DIRPAT[full]="alignment_bias_final_DS_semantic_plus_bias"; SEEDS[full]="42"

for ds in ${DATASETS}; do
  if [[ "${ds}" == "s1s2" ]]; then
    EXP="resnet50_resnet50_s1s2water"; DATA="data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2"
  else
    EXP="resnet50_resnet50_gffloodnet"; DATA="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
  fi
  for v in spatial channel sum reverse gated loss_only bias_only full; do
    pat="${DIRPAT[$v]/DS/${ds}}"
    for seed in ${SEEDS[$v]}; do
      dir="$(ls -d logs/jag26_alignfix_${pat}_s${seed}_* | tail -1)"
      ckpt="$(ls "${dir}"/checkpoints/water_iou_*.ckpt | sort | tail -1)"
      name="jag26_t9test_${ds}_${v}_s${seed}"
      if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry-run] ${name}: ${ckpt}"; continue
      fi
      jid="$(sbatch --exclude "${EXCLUDE_NODES}" --job-name "${name}" \
        --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
        scripts/run/jag_supplement/eval_robustness.sh "${EXP}" "${PROJECT_ROOT}/${ckpt}" "${name}" dual none 0.0 "${DATA} ${OVR[$v]}" | awk '{print $4}')"
      printf "%s\t%s\t%s\t%s\t%s\n" "${ds}" "${v}" "${seed}" "${jid}" "${ckpt}" >> "${MANIFEST}"
      echo "[submitted] ${name}: ${jid}"
    done
  done
done
echo "Manifest: ${MANIFEST}"
