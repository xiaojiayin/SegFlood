#!/usr/bin/env bash

# Profile canonical/default S1S2 fusion baselines without MA-specific tricks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/profile_canonical"
RESULT_DIR="scripts/run/jag_supplement/profile_results/canonical"
MANIFEST="${SCRIPT_DIR}/submissions_profile_canonical_${STAMP}.tsv"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
printf "variant\tjob_id\toverrides\n" > "${MANIFEST}"

submit_profile() {
  local variant="$1"
  local overrides="$2"
  local name="jag26_prof_canonical_${variant}"
  local jid
  jid="$(
    sbatch \
      --exclude "${EXCLUDE_NODES:-g59}" \
      --job-name "${name}" \
      --output "${LOG_DIR}/${name}_%j.out" \
      --error "${LOG_DIR}/${name}_%j.err" \
      --partition a01 \
      --nodes 1 \
      --ntasks-per-node 1 \
      --cpus-per-task 4 \
      --gres gpu:1 \
      --time 02:00:00 \
      --wrap "bash -lc 'cd \"${PROJECT_ROOT}\" && source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}" && conda activate segflood && python scripts/run/jag_supplement/profile_model.py --experiment resnet50_resnet50_s1s2water --name \"${name}\" --overrides \"${overrides}\" --output \"${RESULT_DIR}/${name}.csv\"'" \
      | awk '{print $4}'
  )"
  printf "%s\t%s\t%s\n" "${variant}" "${jid}" "${overrides}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

COMMON="data.add_dem=false data.add_slope=false model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
submit_profile "inputconcat" \
  "${COMMON} +data.output_image_key=true model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity"
submit_profile "dualadd" \
  "${COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=canonical_add"
submit_profile "dualconcat" \
  "${COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=concat"
submit_profile "stdcross" \
  "${COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=canonical_cross"
submit_profile "gated" \
  "${COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=canonical_gated"

echo "Manifest: ${MANIFEST}"
