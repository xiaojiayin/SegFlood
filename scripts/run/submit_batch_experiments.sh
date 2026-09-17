#!/bin/bash
# 批量提交多个实验（按 YAML 路径或实验名）
# 用法：
#   1) 使用默认列表示例（编辑脚本内 DEFAULT_CFGS 后直接运行）
#        bash scripts/run/submit_batch_experiments.sh
#   2) 手动指定若干 YAML 路径或实验名：
#        bash scripts/run/submit_batch_experiments.sh \
#          configs/experiment/dinov3_dinov3_cauflood.yaml \
#          configs/experiment/dinov3_dinov3_gffloodnet.yaml \
#          configs/experiment/dinov3_dinov3_s1s2water.yaml
# 说明：
#  - 会为每个实验生成独立的 out/err：scripts/run/<exp>_%j.out/err
#  - 内部调用 scripts/run/train_experiment.sh（支持数据根自动映射）

set -euo pipefail

# ===== 日志目录配置（可在此处统一调整） =====
# 基础目录（默认所有输出写到这里）
LOG_BASE="scripts/run"
# 按数据集路由的子目录（可自行修改命名）
LOG_SUBDIR_CAU="scripts/run/cauflood"
LOG_SUBDIR_GF="scripts/run/gffloodnet"
LOG_SUBDIR_S1S2="scripts/run/s1s2water"
LOG_SUBDIR_KURO="scripts/run/kurosiwo"
LOG_SUBDIR_WF2="scripts/run/worldfloodsv2"

# 默认列表（示例）：可直接编辑本数组后运行本脚本
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
  # 支持两种输入：YAML 路径 或 直接实验名
  if [[ "${item}" == *.yaml ]]; then
    if [ ! -f "${item}" ]; then
      echo "[skip] 文件不存在: ${item}" >&2
      return
    fi
    exp=$(basename "${item}")
    exp="${exp%.yaml}"
  fi

  # 根据实验名将日志分类到子目录（可在文件顶部配置 LOG_* 变量）
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
  # 通过命令行参数覆盖脚本内 SBATCH 选项
  jid=$(sbatch --job-name "${exp}" --output "${out}" --error "${err}" scripts/run/train_experiment.sh "${exp}" | awk '{print $4}')
  echo "${exp}\t${jid}\t${out}\t${err}" >>"${MANIFEST}"
}

for it in "${CFGS[@]}"; do
  submit_one "${it}"
done

echo "清单文件: ${MANIFEST}"
echo "完成：共提交 ${#CFGS[@]} 个作业。"


