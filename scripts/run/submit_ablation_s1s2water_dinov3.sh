#!/bin/bash

# S1S2-Water（DINOv3 双流）对齐组件消融实验提交脚本
#
# 以 configs/experiment/dinov3_dinov3_s1s2water.yaml 为基准，分别比较：
#   1) 对齐损失 + 对齐偏置（完整模型）
#   2) 仅关闭对齐损失（alignment_enabled=false）
#   3) 仅关闭对齐偏置（xattn_align_bias=false）
#   4) 同时关闭两者（alignment_enabled=false, xattn_align_bias=false）
#
# 统一设置：
#   - DEM/SLOPE 全部关闭：add_dem=false, add_slope=false
#   - 其它超参数均与 dinov3_dinov3_s1s2water.yaml 保持一致
#
# 用法（登录节点）：
#   bash scripts/run/submit_ablation_s1s2water_dinov3.sh

set -euo pipefail

EXP="dinov3_dinov3_s1s2water"
ADD_DEM="false"
ADD_SLOPE="false"

STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST="scripts/run/submissions_s1s2water_dinov3_ablation_${STAMP}.txt"
mkdir -p scripts/run
echo "# S1S2-Water DINOv3 对齐组件消融 @ ${STAMP}" > "${MANIFEST}"

submit_one() {
  local variant="$1"          # 文本标识：full | no-align-loss | no-align-bias | no-align-both
  local extra_overrides="$2"  # 传给 train_s1s2water.sh 的第4个参数（EXTRA_OVERRIDES_STR）

  local cfg_name="${EXP}_${variant}_dem-${ADD_DEM}_slope-${ADD_SLOPE}"
  echo "[submit] ${cfg_name}"

  # 通过 experiment_name 覆盖，让日志名能看出具体消融设置
  local exp_name_override="experiment_name=${cfg_name}"
  local all_overrides="${extra_overrides} ${exp_name_override}"

  jid=$(sbatch \
    --job-name "${cfg_name}" \
    scripts/run/train_s1s2water.sh \
      "${EXP}" \
      "${ADD_DEM}" \
      "${ADD_SLOPE}" \
      "${all_overrides}" | awk '{print $4}')

  echo -e "${cfg_name}\t${jid}" >> "${MANIFEST}"
}

#############################
# 1. 完整模型（对齐损失 + 偏置） #
#############################
# 与 dinov3_dinov3_s1s2water.yaml 保持一致：
#   model.fusion.xattn_align_bias=true
#   model.alignment_enabled=true
submit_one "full_alignloss-on_bias-on" \
  "model.fusion.xattn_align_bias=true model.alignment_enabled=true"

###################################
# 2. 仅关闭对齐损失（PatchNCE）      #
###################################
# 只关闭 SigLIP PatchNCE 对齐正则，保持 cross-attn 结构不变：
#   alignment_enabled=false, alignment_target_weight=0.0
submit_one "no-align-loss_bias-on" \
  "model.alignment_enabled=false model.alignment_target_weight=0.0 model.fusion.xattn_align_bias=true"

################################
# 3. 仅关闭对齐偏置（xattn）     #
################################
# 保留对齐损失约束，但去掉 cross-attn 内部对齐偏置：
submit_one "alignloss-on_no-align-bias" \
  "model.fusion.xattn_align_bias=false model.alignment_enabled=true"

########################################
# 4. 同时关闭对齐损失 + 对齐偏置        #
########################################
# 退化为「无显式对齐」的双流基线：
submit_one "no-align-loss_no-align-bias" \
  "model.alignment_enabled=false model.alignment_target_weight=0.0 model.fusion.xattn_align_bias=false"

echo "清单文件: ${MANIFEST}"
echo "完成：共提交 4 个 S1S2-Water DINOv3 对齐组件消融实验。"



