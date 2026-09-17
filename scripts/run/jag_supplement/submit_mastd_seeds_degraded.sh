#!/usr/bin/env bash
# Degraded-input evaluation of the *standard-protocol* MA-XAttn model for all three seeds, so that
# the "MA-XAttn, standard protocol" row of main-text Table 10 can be reported as mean +- SD.
# S1S2-Water: components_s1s2_gated_s{123,2026} (seed 42 is covered by submit_degradation_seed_repeats.sh
# and the archived jag26_rob_s1s2_m0_* runs). GF-FloodNet: components_gf_gated_s{42,123,2026}
# (the three-seed ablation batch of Table 9). Inference only -- no training.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
DRY_RUN="${DRY_RUN:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST="scripts/run/jag_supplement/submissions_mastd_seeds_degraded_${STAMP}.tsv"
printf "dataset\tseed\tcondition\tdeg_seed\tjob_id\tckpt\n" > "${MANIFEST}"
L=${PROJECT_ROOT}/logs
S1S2="model.encoder.optical_channels=4 model.encoder.sar_channels=2 data.add_dem=false data.add_slope=false"
GF="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
MA="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
NOALIGN="model.alignment_enabled=false model.alignment_target_weight=0.0"
ck() { ls "$L"/$1_*/checkpoints/water_iou_*.ckpt | sort | tail -1; }

submit() { # ds exp ckpt seed ovr
  local ds="$1" exp="$2" ckpt="$3" seed="$4" ovr="$5"
  [[ -f "${ckpt}" ]] || { echo "missing ckpt ${ckpt}" >&2; exit 1; }
  # condition -> "mode deg strength degseeds"
  local -A C=([optical_cloud]="dual optical_cloud 0.3 1 2" [sar_noise]="dual sar_noise 0.5 1 2" [sar_only]="sar_only none 0.5 -" [optical_only]="optical_only none 0.5 -")
  for c in optical_cloud sar_noise sar_only optical_only; do
    read -r mode deg st dseeds <<< "${C[$c]}"
    for d in ${dseeds}; do
      local name="jag26_mastd_${ds}_s${seed}_${c}"; local o="${ovr}"
      if [[ "${d}" != "-" ]]; then name="${name}_d${d}"; o="${o} +model.test_degradation_seed=${d}"; fi
      if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] ${name}: ${mode} ${deg} ${st} | ${o}"; continue; fi
      local jid; jid="$(sbatch --exclude g59,g10 --time=01:30:00 --job-name "${name}" scripts/run/jag_supplement/eval_robustness.sh "${exp}" "${ckpt}" "${name}" "${mode}" "${deg}" "${st}" "${o}" | awk '{print $4}')"
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${ds}" "${seed}" "${c}" "${d}" "${jid}" "${ckpt}" >> "${MANIFEST}"
      echo "[submitted] ${name}: ${jid}"
    done
  done
}
for s in 123 2026; do submit s1s2 resnet50_resnet50_s1s2water "$(ck jag26_alignfix_components_s1s2_gated_s${s})" "${s}" "${S1S2} ${MA} ${NOALIGN}"; done
for s in 42 123 2026; do submit gf resnet50_resnet50_gffloodnet "$(ck jag26_alignfix_components_gf_gated_s${s})" "${s}" "${GF} ${MA} ${NOALIGN}"; done
echo "Manifest: ${MANIFEST}"
