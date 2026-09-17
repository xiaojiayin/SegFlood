#!/usr/bin/env bash

# Matched GF-FloodNet baselines for the two pre-registered regional holdouts.
# Usage: DRY_RUN=1 bash scripts/run/jag_supplement/submit_gf_holdout_baselines.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="scripts/run/jag_supplement/logs/gf_holdout_baselines"
MANIFEST="scripts/run/jag_supplement/submissions_gf_holdout_baselines_${STAMP}.tsv"
mkdir -p "${LOG_DIR}"
printf "region\tvariant\tjob_id\toverrides\n" > "${MANIFEST}"

submit_one() {
  local region="$1"
  local experiment="$2"
  local variant="$3"
  local overrides="$4"
  local name="jag26_gfhold_${region}_${variant}"
  local all_overrides="seed=42 data.holdout_patterns=[${region}_*] experiment_name=${name} ${overrides}"

  if squeue -h -u "${USER}" -n "${name}" 2>/dev/null | awk 'NF { found=1 } END { exit !found }'; then
    echo "[skip queued] ${name}"
    return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\n" "${region}" "${variant}" "DRY_RUN" "${overrides}" >> "${MANIFEST}"
    echo "[dry-run] ${name}: ${experiment} ${all_overrides}"
    return
  fi

  local jid
  jid="$(
    sbatch \
      --job-name "${name}" \
      --output "${LOG_DIR}/${name}_%j.out" \
      --error "${LOG_DIR}/${name}_%j.err" \
      scripts/run/train_gffloodnet.sh "${experiment}" "${all_overrides}" \
      | awk '{print $4}'
  )"
  printf "%s\t%s\t%s\t%s\n" "${region}" "${variant}" "${jid}" "${overrides}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

for region in Australia Brazil; do
  submit_one "${region}" "resnet50_early_fusion_gffloodnet" "inputconcat" ""
  submit_one "${region}" "resnet50_resnet50_gffloodnet" "dualconcat" \
    "model.fusion.fusion_type=concat model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_one "${region}" "resnet50_resnet50_gffloodnet" "gated" \
    "model.fusion.fusion_type=gated model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_one "${region}" "resnet50_resnet50_gffloodnet" "manoalign" \
    "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_one "${region}" "resnet50_optical_gffloodnet" "optical" ""
  submit_one "${region}" "resnet50_sar_gffloodnet" "sar" ""
done

echo "Manifest: ${MANIFEST}"
