#!/usr/bin/env bash
# CAU-Flood Table 4 completion under the standard protocol (same period / same overrides as
# Job 499168 = ResNet-50 MA-XAttn, Job 504717 = ResNet-50 early fusion, Job 505655 = DINOv3 MA-XAttn):
#   (a) needed rows: EfficientNet-B4 early + MA, SAM2 early + MA, DINOv3 early   [MA = fixed-sum two-path, as 499168/505655]
#   (b) optional:    MA-XAttn with the weighted (gated) two-path block for all four backbones,
#                    so that Table 4 can alternatively be reported with the paper-default block.
# seed 42, test=true, CAU experiment defaults (100 max epochs, cosine T_max 50, patience 10, aux 0.3).
# Usage: DRY_RUN=1 bash scripts/run/jag_supplement/submit_cau_table4_fill.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
DRY_RUN="${DRY_RUN:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="scripts/run/jag_supplement/logs/cau_table4_fill"; mkdir -p "${LOG_ROOT}"
MANIFEST="scripts/run/jag_supplement/submissions_cau_table4_fill_${STAMP}.tsv"
printf "variant\tjob_id\texperiment\toverrides\n" > "${MANIFEST}"

MA_SUM="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_align_bias=false +model.alignment_enabled=false +model.alignment_target_weight=0.0 model.alignment_mode=disabled"
MA_GATED="${MA_SUM} +model.fusion.xattn_component_fusion=gated"
EFF_HOMO="model.encoder.sar_model_name=efficientnet_b4"

submit() { # variant experiment overrides
  local name="jag26_caut4_$1_s42"
  local all="experiment_name=${name} seed=42 test=true $3"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] ${name}: $2 | ${all}"; return; fi
  local jid; jid="$(sbatch --exclude g59,g10 --time=12:00:00 --job-name "${name}" --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" scripts/run/train_experiment.sh "$2" "" "${all}" | awk '{print $4}')"
  printf "%s\t%s\t%s\t%s\n" "$1" "${jid}" "$2" "${all}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}
# (a) needed
submit effb4_early   efficientnetb4_early_fusion_cauflood ""
submit effb4_ma_sum  efficientnetb4_mobilenetv3_cauflood  "${EFF_HOMO} ${MA_SUM}"
submit sam2_early    sam2_early_fusion_cauflood           ""
submit sam2_ma_sum   sam2_sam2_cauflood                   "${MA_SUM}"
submit dinov3_early  dinov3_early_fusion_cauflood         ""
# (b) optional gated variants
submit effb4_ma_gated  efficientnetb4_mobilenetv3_cauflood "${EFF_HOMO} ${MA_GATED}"
submit sam2_ma_gated   sam2_sam2_cauflood                  "${MA_GATED}"
submit r50_ma_gated    resnet50_resnet50_cauflood          "${MA_GATED}"
submit dinov3_ma_gated dinov3_dinov3_cauflood              "${MA_GATED}"
echo "Manifest: ${MANIFEST}"
