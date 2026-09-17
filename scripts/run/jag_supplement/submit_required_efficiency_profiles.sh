#!/usr/bin/env bash

# Profile the matched fusion baselines on GF and S1S2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/profile_required"
RESULT_DIR="scripts/run/jag_supplement/profile_results/required"
MANIFEST="${SCRIPT_DIR}/submissions_profile_required_${STAMP}.tsv"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
printf "dataset\tvariant\tjob_id\texperiment\toverrides\n" > "${MANIFEST}"

submit_profile() {
  local dataset="$1"
  local experiment="$2"
  local variant="$3"
  local overrides="$4"
  local name="jag26_prof_${dataset}_${variant}"
  local output="${RESULT_DIR}/${name}.csv"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${dataset}" "${variant}" "DRY_RUN" "${experiment}" "${overrides}" \
      >> "${MANIFEST}"
    echo "[dry-run] ${name}: ${experiment} ${overrides}"
    return
  fi

  local jid
  jid="$(
    sbatch \
      --exclude "${EXCLUDE_NODES}" \
      --job-name "${name}" \
      --output "${LOG_DIR}/${name}_%j.out" \
      --error "${LOG_DIR}/${name}_%j.err" \
      --partition a01 \
      --nodes 1 \
      --ntasks-per-node 1 \
      --cpus-per-task 4 \
      --gres gpu:1 \
      --time 02:00:00 \
      --wrap "bash -lc 'cd \"${PROJECT_ROOT}\" && source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}" && conda activate segflood && python scripts/run/jag_supplement/profile_model.py --experiment \"${experiment}\" --name \"${name}\" --overrides \"${overrides}\" --output \"${output}\"'" \
      | awk '{print $4}'
  )"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${dataset}" "${variant}" "${jid}" "${experiment}" "${overrides}" \
    >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

for dataset in gf s1s2; do
  if [[ "${dataset}" == "gf" ]]; then
    dual_experiment="resnet50_resnet50_gffloodnet"
    early_experiment="resnet50_early_fusion_gffloodnet"
    channels="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
  else
    dual_experiment="resnet50_resnet50_s1s2water"
    early_experiment="resnet50_early_fusion_s1s2water"
    channels="data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2"
  fi

  submit_profile "${dataset}" "${early_experiment}" "inputconcat" ""
  submit_profile "${dataset}" "${dual_experiment}" "dualadd" \
    "${channels} model.fusion.fusion_type=add model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_profile "${dataset}" "${dual_experiment}" "dualconcat" \
    "${channels} model.fusion.fusion_type=concat model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_profile "${dataset}" "${dual_experiment}" "stdcross" \
    "${channels} model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial +model.fusion.xattn_attention_type=standard model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_profile "${dataset}" "${dual_experiment}" "ma_component_gate" \
    "${channels} model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
done

echo "Manifest: ${MANIFEST}"
