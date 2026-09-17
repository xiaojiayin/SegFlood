#!/bin/bash

# 一次性提交同一个多模态 checkpoint 的三种测试：
#   dual / optical_only / sar_only
#
# 用法：
#   bash scripts/run/submit_missing_modality.sh <EXPERIMENT_NAME> <CKPT_PATH>

set -euo pipefail

EXP="${1:?需要实验配置名，例如 resnet50_resnet50_gffloodnet}"
CKPT="${2:?需要 checkpoint 路径}"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_missing_modality_${EXP}_${STAMP}.txt"
LOG_DIR="scripts/run/logs/missing_modality/${EXP}"
mkdir -p scripts/run "${LOG_DIR}"
echo "# Missing-modality eval for ${EXP} @ ${STAMP}" > "${MANIFEST}"
echo "# ckpt=${CKPT}" >> "${MANIFEST}"

for MODE in dual optical_only sar_only; do
  JOB_NAME="${EXP}_missing-${MODE}"
  echo "[submit] ${JOB_NAME}"
  jid=$(sbatch \
    --job-name "${JOB_NAME}" \
    --output "${LOG_DIR}/${JOB_NAME}_%j.out" \
    --error "${LOG_DIR}/${JOB_NAME}_%j.err" \
    scripts/run/eval_missing_modality.sh \
      "${EXP}" \
      "${CKPT}" \
      "${MODE}" | awk '{print $4}')
  echo -e "${JOB_NAME}\t${jid}" >> "${MANIFEST}"
done

echo "清单文件: ${MANIFEST}"
