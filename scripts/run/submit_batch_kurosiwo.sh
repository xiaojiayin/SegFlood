#!/bin/bash

# 批量提交 KuroSiwo 的核心 5 个实验（基于 sam2_kurosiwo）
#
# 用法：
#   bash scripts/run/submit_batch_kurosiwo.sh
#
# 说明：
#   - 内部调用 scripts/run/train_kurosiwo.sh
#   - EXP 固定为 sam2_kurosiwo（SAM2-Base 实验）
#   - 提交的 5 个配置：
#       1) 线性域 Z-Score（纯 SAR）
#       2) dB 域 + Z-Score + vh/vv
#       3) dB 域 + Z-Score（纯 SAR）
#       4) dB 域 + DEM(Z-Score)
#       5) dB 域 + DEM(不归一化)
#       6) 纯 dB（不做 Z-Score，纯 SAR）

set -euo pipefail

EXP="sam2_kurosiwo"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_kurosiwo_${STAMP}.txt"
mkdir -p scripts/run
echo "# KuroSiwo submissions @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local use_dem="$1"
  local dem_scale="$2"
  local scale_input="$3"
  local use_ratio="$4"
  local clear_db="$5"

  local cfg_name="${EXP}_dem-${use_dem}_demscale-${dem_scale}_scale-${scale_input}_ratio-${use_ratio}_cleardb-${clear_db}"
  echo "[submit] ${cfg_name}"

  jid=$(sbatch \
    --job-name "${cfg_name}" \
    scripts/run/train_kurosiwo.sh \
      "${EXP}" \
      "${use_dem}" \
      "${dem_scale}" \
      "${scale_input}" \
      "${use_ratio}" \
      "${clear_db}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}" >> "${MANIFEST}"
}

# 1) 线性域 Z-Score（纯 SAR）
submit_one "false" "zscore" "normalize" "false" "false"

# 2) dB 域 + Z-Score + vh/vv
submit_one "false" "zscore" "db" "true" "false"

# 3) dB 域 + Z-Score（纯 SAR）
submit_one "false" "zscore" "db" "false" "false"

# 4) dB 域 + DEM(Z-Score)
submit_one "true" "zscore" "db" "false" "false"

# 5) dB 域 + DEM(不归一化)
submit_one "true" "none" "db" "false" "false"

# 6) 纯 dB（不做 Z-Score，纯 SAR）
submit_one "false" "zscore" "db" "false" "true"

echo "清单文件: ${MANIFEST}"
echo "完成：共提交 6 个作业。"


