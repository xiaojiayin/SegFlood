#!/bin/bash

# GF-FloodNet cross-event / cross-region holdout 训练提交脚本。
#
# 按 images/*.tif 文件名匹配 holdout pattern：
#   - 训练/验证：不匹配 pattern 的 tiles
#   - 测试：匹配 pattern 的 tiles
#
# 用法：
#   bash scripts/run/submit_gffloodnet_holdout.sh <EXPERIMENT_NAME> <PATTERN1> [PATTERN2 ...]
#
# 示例：
#   bash scripts/run/submit_gffloodnet_holdout.sh resnet50_resnet50_gffloodnet "Brazil_*" "China_*"
#
# 注意：
#   pattern 会交给 Python fnmatch，支持 * 通配符；请用引号包起来，避免 shell 展开。

set -euo pipefail

EXP="${1:-resnet50_resnet50_gffloodnet}"
shift || true

if [ "$#" -eq 0 ]; then
  echo "用法: bash $0 <EXPERIMENT_NAME> <PATTERN1> [PATTERN2 ...]" >&2
  echo "示例: bash $0 resnet50_resnet50_gffloodnet \"Brazil_*\" \"China_*\"" >&2
  exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_gffloodnet_holdout_${EXP}_${STAMP}.txt"
LOG_DIR="scripts/run/logs/gffloodnet_holdout"
mkdir -p scripts/run "${LOG_DIR}"
echo "# GF-FloodNet holdout submissions for ${EXP} @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local pattern="$1"
  local safe_name
  safe_name=$(echo "${pattern}" | tr -cs 'A-Za-z0-9_-' '_' | sed 's/^_//;s/_$//')
  local cfg_name="${EXP}_holdout_${safe_name}"
  local list_arg="[${pattern}]"
  local overrides="data.holdout_patterns=${list_arg} experiment_name=${cfg_name}"

  echo "[submit] ${cfg_name} holdout=${pattern}"
  jid=$(sbatch \
    --job-name "${cfg_name}" \
    --output "${LOG_DIR}/${cfg_name}_%j.out" \
    --error "${LOG_DIR}/${cfg_name}_%j.err" \
    scripts/run/train_gffloodnet.sh \
      "${EXP}" \
      "${overrides}" | awk '{print $4}')
  echo -e "${cfg_name}\t${jid}\t${pattern}" >> "${MANIFEST}"
}

for pattern in "$@"; do
  submit_one "${pattern}"
done

echo "清单文件: ${MANIFEST}"
