#!/bin/bash
#
# Batch-submit multiple experiments (by YAML path or experiment name).
#
# Usage:
#   1) Use the default list (edit DEFAULT_CFGS):
#        bash scripts/run/submit_batch_experiments.sh
#   2) Provide YAML paths or experiment names:
#        bash scripts/run/submit_batch_experiments.sh \
#          configs/experiment/dinov3_dinov3_cauflood.yaml \
#          configs/experiment/dinov3_dinov3_gffloodnet.yaml
#
# Notes:
#  - Generates per-job out/err under scripts/run/*
#  - Calls scripts/run/train_experiment.sh (dataset root inferred from experiment name)

set -euo pipefail

# Log directory routing
# Base dir
LOG_BASE="scripts/run"
# Per-dataset subdirs
LOG_SUBDIR_CAU="scripts/run/cauflood"
LOG_SUBDIR_GF="scripts/run/gffloodnet"
LOG_SUBDIR_S1S2="scripts/run/s1s2water"
LOG_SUBDIR_KURO="scripts/run/kurosiwo"
LOG_SUBDIR_WF2="scripts/run/worldfloodsv2"

# Default list (examples)
DEFAULT_CFGS=(
configs/experiment/dinov3_dinov3_s1s2water.yaml
configs/experiment/dinov3_kurosiwo.yaml
)

args=("$@")
if [ ${#args[@]} -eq 0 ]; then
  CFGS=("${DEFAULT_CFGS[@]}")
else
  CFGS=("${args[@]}")
fi

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_${STAMP}.txt"
mkdir -p scripts/run
echo "# submissions @ ${STAMP}" >"${MANIFEST}"

submit_one() {
  local item="$1"
  local exp="${item}"
  # Supports: YAML path or experiment name
  if [[ "${item}" == *.yaml ]]; then
    if [ ! -f "${item}" ]; then
      echo "[skip] file not found: ${item}" >&2
      return
    fi
    exp=$(basename "${item}")
    exp="${exp%.yaml}"
  fi

  # Route logs by dataset keyword
  local subdir="${LOG_BASE}"
  case "${exp}" in
    *cauflood*)      subdir="${LOG_SUBDIR_CAU}" ;;
    *gffloodnet*)    subdir="${LOG_SUBDIR_GF}" ;;
    *s1s2water*)     subdir="${LOG_SUBDIR_S1S2}" ;;
    *kurosiwo*)      subdir="${LOG_SUBDIR_KURO}" ;;
    *worldfloodsv2*) subdir="${LOG_SUBDIR_WF2}" ;;
    *)               subdir="${LOG_BASE}" ;;
  esac
  mkdir -p "${subdir}"
  local out="${subdir}/${exp}_%j.out"
  local err="${subdir}/${exp}_%j.err"
  echo "[submit] ${exp} → out=${out} err=${err}"
  # Override SBATCH options via sbatch CLI
  jid=$(sbatch --job-name "${exp}" --output "${out}" --error "${err}" scripts/run/train_experiment.sh "${exp}" | awk '{print $4}')
  echo "${exp}\t${jid}\t${out}\t${err}" >>"${MANIFEST}"
}

for it in "${CFGS[@]}"; do
  submit_one "${it}"
done

echo "Manifest: ${MANIFEST}"
echo "Done. Submitted ${#CFGS[@]} jobs."


