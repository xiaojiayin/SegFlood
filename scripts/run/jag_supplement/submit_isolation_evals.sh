#!/usr/bin/env bash
# Degraded-input evaluation of the S1S2-Water isolation runs (last.ckpt, 50-epoch schedule),
# chained with afterok on the still-running training jobs so that results appear without
# manual intervention. Conditions and degradation seed (1) match Table 8 / Supplement S13.
#   sym_noalign     : final symmetric occlusion-aware recipe WITHOUT the agreement term (MA-XAttn)
#   std_minep50     : standard protocol trained for 50 epochs, last checkpoint (already finished)
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
L=${PROJECT_ROOT}/logs
mkdir -p scripts/run/jag_supplement/logs/robustness_eval
STAMP="$(date +%Y%m%d_%H%M%S)"; MANIFEST="scripts/run/jag_supplement/submissions_isolation_evals_${STAMP}.tsv"
printf "variant\tseed\tcondition\tjob_id\tdepends_on\n" > "${MANIFEST}"

S1S2="model.encoder.optical_channels=4 model.encoder.sar_channels=2 data.add_dem=false data.add_slope=false"
NOALIGN="model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
MA="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
CONCAT="model.fusion.fusion_type=concat"

# variant -> eval overrides ; variant -> training job id per seed (empty = already finished)
declare -A OVR DEP
OVR[sym_noalign]="${S1S2} ${MA} model.alignment_enabled=false model.alignment_target_weight=0.0"
OVR[std_minep50]="${S1S2} ${MA} model.alignment_enabled=false model.alignment_target_weight=0.0"
DEP[sym_noalign_42]=534002; DEP[sym_noalign_123]=534004; DEP[sym_noalign_2026]=533989
DEP[std_minep50_42]=""; DEP[std_minep50_123]=""; DEP[std_minep50_2026]=""

declare -A COND=([clean]="dual none 0" [optical_cloud]="dual optical_cloud 0.3" [sar_noise]="dual sar_noise 0.5" [missing_optical]="sar_only none 0" [missing_sar]="optical_only none 0")

for v in ${VARIANTS}; do
  for s in 42 123 2026; do
    dep="${DEP[${v}_${s}]}"; depflag=""; [[ -n "${dep}" ]] && depflag="--dependency=afterok:${dep}"
    prefix="jag26_iso_${v}_s${s}"
    for c in clean optical_cloud sar_noise missing_optical missing_sar; do
      read -r mode deg st <<< "${COND[$c]}"
      ename="jag26_isoeval_${v}_s${s}_${c}"
      jid="$(sbatch --exclude g59,g10 --partition a01,h01 --gres=gpu:1 --cpus-per-task=8 --time=02:00:00 ${depflag} --job-name "${ename}" \
        --output "scripts/run/jag_supplement/logs/robustness_eval/${ename}_%j.out" \
        --error  "scripts/run/jag_supplement/logs/robustness_eval/${ename}_%j.err" \
        scripts/run/jag_supplement/eval_last_ckpt.sh "${prefix}" resnet50_resnet50_s1s2water "${ename}" "${mode}" "${deg}" "${st}" "${OVR[$v]} +model.test_degradation_seed=1" \
        | awk '{print $4}')"
      printf "%s\t%s\t%s\t%s\t%s\n" "${v}" "${s}" "${c}" "${jid}" "${dep:--}" >> "${MANIFEST}"
      echo "[submitted] ${ename}: ${jid} ${depflag}"
    done
  done
done
echo "Manifest: ${MANIFEST}"
