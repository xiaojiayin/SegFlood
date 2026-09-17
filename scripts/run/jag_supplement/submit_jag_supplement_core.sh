#!/bin/bash

# One-command entry for supplementary experiments.
#
# Recommended first run:
#   cd ${PROJECT_ROOT}
#   bash scripts/run/jag_supplement/submit_jag_supplement_core.sh
#
# Full run with key multi-seed stability checks:
#   bash scripts/run/jag_supplement/submit_jag_supplement_core.sh full

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail

MODE="${1:-standard}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "================ supplementary submission ================"
echo "mode: ${MODE}"
echo "project: ${PROJECT_ROOT}"
echo "==========================================================="

bash scripts/run/jag_supplement/submit_fusion_baselines.sh all
bash scripts/run/jag_supplement/submit_profile_fusion_baselines.sh

if [[ "${MODE}" == "full" ]]; then
  bash scripts/run/jag_supplement/submit_multiseed_key.sh
elif [[ "${MODE}" != "standard" ]]; then
  echo "Unknown mode: ${MODE}. Use standard|full" >&2
  exit 1
fi

echo "All requested supplementary jobs have been submitted."
