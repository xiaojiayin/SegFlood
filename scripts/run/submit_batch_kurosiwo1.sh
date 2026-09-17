#!/bin/bash

# 批量提交 KuroSiwo 不同主干模型的实验（固定使用 db + z-score，且不启用 DEM）
# 模型列表：
#   DINOv3:
#     - vit_small_patch16_dinov3
#     - vit_small_plus_patch16_dinov3
#     - vit_base_patch16_dinov3
#     - vit_base_patch16_dinov3_qkvb
#     - vit_large_patch16_dinov3
#     - vit_large_patch16_dinov3.sat493m
#   SAM2-Hiera:
#     - sam2_hiera_tiny
#     - sam2_hiera_small
#     - sam2_hiera_base_plus
#     - sam2_hiera_large
#
# 用法：
#   bash scripts/run/submit_batch_kurosiwo1.sh
#
# 说明：
#   - 内部调用 scripts/run/train_kurosiwo.sh
#   - 归一化方式固定为：db + z-score（data.scale_input=db，data.data_mean/std 使用默认配置）
#   - DEM 固定关闭：dem=false
#   - vh/vv 比值通道关闭：use_ratio=false（保持与主线结果一致）
#   - 每个作业的 job-name / 日志文件名中都会包含具体主干名称，便于区分

set -euo pipefail

EXP_DINO="dinov3_kurosiwo"
EXP_SAM2="sam2_kurosiwo"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_kurosiwo_backbones_${STAMP}.txt"
mkdir -p scripts/run
echo "# KuroSiwo backbone submissions @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local exp="$1"           # 实验 YAML 名称（dinov3_kurosiwo / sam2_kurosiwo）
  local backbone="$2"      # 主干模型名（用于 job-name）
  local extra_overrides="$3"  # 传给 train_kurosiwo.sh 的第7个参数（EXTRA_OVERRIDES_STR）

  local cfg_name="${exp}_${backbone}_dem-false_scale-db"
  echo "[submit] ${cfg_name}"

  # 统一设置：dem=false, dem_scale=zscore(无效), scale_input=db, use_ratio=false, clear_db=false
  jid=$(sbatch \
    --job-name "${cfg_name}" \
    scripts/run/train_kurosiwo.sh \
      "${exp}" \
      "false" \
      "zscore" \
      "db" \
      "false" \
      "false" \
      "${extra_overrides}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}" >> "${MANIFEST}"
}

########################
# 1. DINOv3 主干系列  #
########################

# vit_small_patch16_dinov3
submit_one "${EXP_DINO}" "vit_small_patch16_dinov3" \
  "model.encoder.optical_model_name=vit_small_patch16_dinov3 model.encoder.sar_model_name=vit_small_patch16_dinov3"

# vit_small_plus_patch16_dinov3
submit_one "${EXP_DINO}" "vit_small_plus_patch16_dinov3" \
  "model.encoder.optical_model_name=vit_small_plus_patch16_dinov3 model.encoder.sar_model_name=vit_small_plus_patch16_dinov3"

# vit_base_patch16_dinov3
submit_one "${EXP_DINO}" "vit_base_patch16_dinov3" \
  "model.encoder.optical_model_name=vit_base_patch16_dinov3 model.encoder.sar_model_name=vit_base_patch16_dinov3"

# vit_base_patch16_dinov3_qkvb
submit_one "${EXP_DINO}" "vit_base_patch16_dinov3_qkvb" \
  "model.encoder.optical_model_name=vit_base_patch16_dinov3_qkvb model.encoder.sar_model_name=vit_base_patch16_dinov3_qkvb"

# vit_large_patch16_dinov3
submit_one "${EXP_DINO}" "vit_large_patch16_dinov3" \
  "model.encoder.optical_model_name=vit_large_patch16_dinov3 model.encoder.sar_model_name=vit_large_patch16_dinov3"

# vit_large_patch16_dinov3.sat493m
submit_one "${EXP_DINO}" "vit_large_patch16_dinov3.sat493m" \
  "model.encoder.optical_model_name=vit_large_patch16_dinov3.sat493m model.encoder.sar_model_name=vit_large_patch16_dinov3.sat493m"

########################
# 2. SAM2-Hiera 主干  #
########################

# sam2_hiera_tiny
submit_one "${EXP_SAM2}" "sam2_hiera_tiny" \
  "model.encoder.optical_model_name=sam2_hiera_tiny model.encoder.sar_model_name=sam2_hiera_tiny"

# sam2_hiera_small
submit_one "${EXP_SAM2}" "sam2_hiera_small" \
  "model.encoder.optical_model_name=sam2_hiera_small model.encoder.sar_model_name=sam2_hiera_small"

# sam2_hiera_base_plus
submit_one "${EXP_SAM2}" "sam2_hiera_base_plus" \
  "model.encoder.optical_model_name=sam2_hiera_base_plus model.encoder.sar_model_name=sam2_hiera_base_plus"

# sam2_hiera_large
submit_one "${EXP_SAM2}" "sam2_hiera_large" \
  "model.encoder.optical_model_name=sam2_hiera_large model.encoder.sar_model_name=sam2_hiera_large"

echo "清单文件: ${MANIFEST}"
echo "完成：共提交 10 个主干对比实验。"



