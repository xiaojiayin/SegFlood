#!/bin/bash

# GF-FloodNet（ResNet50 双流）对齐组件消融实验提交脚本
#
# 基准配置：configs/experiment/resnet50_resnet50_gffloodnet.yaml
# 其中启用了：
#   - fusion.xattn 作为跨模态特征融合
#   - alignment_* SigLIP PatchNCE 对齐正则
#
# 这里构建 4 组对比实验：
#   1) full_alignloss-on_bias-on
#      - xattn 融合 + 对齐偏置 + 对齐正则（alignment_enabled=true）
#   2) no-align-loss_bias-on
#      - 仅关闭对齐正则 (alignment_enabled=false, alignment_target_weight=0.0)，保留对齐偏置
#   3) alignloss-on_no-align-bias
#      - 保留对齐正则，但关闭对齐偏置 (xattn_align_bias=false)
#   4) no-align-loss_no-align-bias
#      - 同时关闭对齐正则 + 对齐偏置，退化为“无显式对齐”的双流 baseline
#
# 用法（登录节点）：
#   bash scripts/run/submit_ablation_resnet50_resnet50_gffloodnet.sh

set -euo pipefail

EXP="resnet50_resnet50_gffloodnet"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_resnet50_resnet50_gffloodnet_ablation_${STAMP}.txt"
LOG_DIR="scripts/run/logs/alignment_ablation"
mkdir -p scripts/run "${LOG_DIR}"
echo "# GF-FloodNet ResNet50 对齐组件消融 @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local variant="$1"          # 文本标识
  local extra_overrides="$2"  # 传给 train_gffloodnet.sh 的第2个参数（EXTRA_OVERRIDES_STR）

  local cfg_name="${EXP}_${variant}"
  echo "[submit] ${cfg_name}"

  # 确保 experiment_name 中也写入 variant，便于日志/ckpt 区分
  local exp_name_override="experiment_name=${cfg_name}"
  local all_overrides="${extra_overrides} ${exp_name_override}"

  jid=$(sbatch \
    --job-name "${cfg_name}" \
    --output "${LOG_DIR}/${cfg_name}_%j.out" \
    --error "${LOG_DIR}/${cfg_name}_%j.err" \
    scripts/run/train_gffloodnet.sh \
      "${EXP}" \
      "${all_overrides}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}" >> "${MANIFEST}"
}

############################################
# 1. full_alignloss-on_bias-on （主模型）   #
############################################
# 与更新后的 yaml 一致：xattn + 对齐偏置 + alignment_enabled=true
submit_one "full_alignloss-on_bias-on" \
  "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=true model.alignment_enabled=true"

#######################################
# 2. no-align-loss_bias-on            #
#######################################
# 保留 xattn + 对齐偏置，仅关闭对齐正则：
submit_one "no-align-loss_bias-on" \
  "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=true model.alignment_enabled=false model.alignment_target_weight=0.0"

##########################################
# 3. alignloss-on_no-align-bias          #
##########################################
# 保留对齐正则，仅关闭对齐偏置：
submit_one "alignloss-on_no-align-bias" \
  "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=false model.alignment_enabled=true"

#############################################
# 4. no-align-loss_no-align-bias            #
#############################################
# 同时关闭对齐正则 + 对齐偏置 → 无显式对齐的 baseline。
submit_one "no-align-loss_no-align-bias" \
  "model.fusion.fusion_type=xattn model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"

echo "清单文件: ${MANIFEST}"
echo "完成：共提交 4 个 GF-FloodNet ResNet50 对齐组件消融实验。"



