#!/bin/bash
#SBATCH --job-name=effb4mobilenetv2_wf2
#SBATCH --output=scripts/run/effb4mobilenetv2_wf2_%j.out
#SBATCH --error=scripts/run/effb4mobilenetv2_wf2_%j.err
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail

# 用法：
#   sbatch scripts/run/train_efficientnetb4_mobilenetv2_worldfloodsv2.sh [channels_key] [lr]
# 参数：
#   channels_key: rgb|bgr|bgri|riswir|bgriswir|bgriswirs|l89s2|sub_20|all  默认 bgri
#   lr: 学习率（可选，默认 5e-4）

CHANNELS_KEY=${1:-bgri}
LR=${2:-5e-4}

BASE_EXPERIMENT="efficientnetb4_mobilenetv2_worldfloodsv2"
DATA_ROOT="${PROJECT_ROOT}/data/WorldFloodsv2"
LOG_ROOT="${PROJECT_ROOT}/logs"
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_CACHE="${PROJECT_ROOT}/checkpoints/.cache"
export TORCH_HOME="${PROJECT_ROOT}/checkpoints"
mkdir -p "$HF_HUB_CACHE" "$LOG_ROOT"

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate segflood

echo "================ 运行计划 ================"
echo "Modalities: optical"
echo "Channels: ${CHANNELS_KEY}"
echo "learning_rate: ${LR}"
echo "fusion_type: (single-modal; not used)"
echo "========================================="

run_base="efficientnetb4_mobilenetv2_optical_${CHANNELS_KEY}_wf2"

# 根据通道组合计算光学通道数
case "$CHANNELS_KEY" in
  rgb) opt_ch=3 ;;
  bgr) opt_ch=3 ;;
  bgri) opt_ch=4 ;;
  riswir) opt_ch=3 ;;
  bgriswir) opt_ch=5 ;;
  bgriswirs) opt_ch=6 ;;
  l89s2) opt_ch=8 ;;
  sub_20) opt_ch=10 ;;
  all) opt_ch=13 ;;
  *) echo "不支持的 channels_key: $CHANNELS_KEY" >&2; exit 2 ;;
esac
sar_ch=0

echo -e "\n>>> [RUN] mode=optical 开始时间: $(date)"
echo "通道: optical_channels=${opt_ch}, sar_channels=${sar_ch}"

set +e
python src/train.py \
  experiment=${BASE_EXPERIMENT} \
  data.root=${DATA_ROOT} \
  data.split=train \
  data.channels=${CHANNELS_KEY} \
  data.filter_windows.apply=true \
  data.filter_windows.version=v1 \
  data.filter_windows.threshold_clouds=0.25 \
  data.batch_size=128 \
  data.num_workers=8 \
  model.learning_rate=${LR} \
  +data.optical_channels=${opt_ch} \
  +data.sar_channels=${sar_ch}
exit_code=$?
set -e

if [ $exit_code -ne 0 ]; then
  echo "[RUN] mode=optical ❌ 失败 (code=$exit_code) - 终止" >&2
  exit $exit_code
else
  echo "[RUN] mode=optical ✅ 成功结束"
fi

echo "训练完成: $(date)"


