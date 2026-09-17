#!/bin/bash

# Submit a small set of multi-seed runs for seed-stability checks.
#
# Default seeds are 42, 123, and 2026. Override with:
#   SEEDS="123 2026 3407" bash scripts/run/jag_supplement/submit_multiseed_key.sh
#
# Usage on the server login node:
#   cd ${PROJECT_ROOT}
#   bash scripts/run/jag_supplement/submit_multiseed_key.sh
#
# The selected runs are intentionally small:
#   - GF-FloodNet ResNet50 concat vs full MA-XAttn
#   - GF-FloodNet ResNet50 xattn without explicit alignment
#   - S1S2-Water ResNet50 full MA-XAttn

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

SEEDS="${SEEDS:-42 123 2026}"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="scripts/run/jag_supplement/logs/multiseed"
MANIFEST="scripts/run/jag_supplement/submissions_multiseed_key_${STAMP}.txt"
mkdir -p "${LOG_DIR}"

echo "# supplementary key multi-seed runs @ ${STAMP}" > "${MANIFEST}"
echo "# seeds=${SEEDS}" >> "${MANIFEST}"

submit_seeded() {
  local exp="$1"
  local variant="$2"
  local base_overrides="$3"
  local seed="$4"

  local cfg_name="${exp}_seed-${seed}_${variant}"
  local all_overrides="${base_overrides} seed=${seed} experiment_name=${cfg_name}"

  echo "[submit] ${cfg_name}"
  local jid
  jid=$(sbatch \
    --job-name "${cfg_name}" \
    --output "${LOG_DIR}/${cfg_name}_%j.out" \
    --error "${LOG_DIR}/${cfg_name}_%j.err" \
    scripts/run/train_experiment.sh \
      "${exp}" \
      "" \
      "${all_overrides}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}\t${base_overrides}" >> "${MANIFEST}"
}

for seed in ${SEEDS}; do
  submit_seeded "resnet50_resnet50_gffloodnet" "concat" \
    "model.fusion.fusion_type=concat model.alignment_enabled=false model.alignment_target_weight=0.0" \
    "${seed}"

  submit_seeded "resnet50_resnet50_gffloodnet" "xattn_no_align" \
    "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0" \
    "${seed}"

  submit_seeded "resnet50_resnet50_gffloodnet" "maxattn_full" \
    "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=true model.alignment_enabled=true" \
    "${seed}"

  submit_seeded "resnet50_resnet50_s1s2water" "maxattn_full" \
    "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=true model.alignment_enabled=true" \
    "${seed}"
done

echo "Manifest: ${MANIFEST}"
echo "Done."
