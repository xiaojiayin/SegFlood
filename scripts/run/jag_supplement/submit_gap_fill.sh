#!/usr/bin/env bash

# Remaining gap-fill experiments (2026-09-02:
# the R1/R2 coverage table in JAG_FINAL_FIGURE_TABLE_DECISIONS.md).
#
#   dinov3_fixed     DINOv3 dual-stream runs after the SAR out_indices truncation
#                    fix: S1S2 MA (OOM retry at bs64/accum2),
#                    GF-FloodNet Concat + MA (Table 2), CAU-Flood MA (Table 5).
#                    paired with Job 499168 (agreement disabled).
#   profile_dinov3   Re-profile the fixed DINOv3 dual-stream models for Table 7.
#
# Usage:
#   bash scripts/run/jag_supplement/submit_gap_fill.sh dinov3_fixed
#   DRY_RUN=1 bash scripts/run/jag_supplement/submit_gap_fill.sh all

set -euo pipefail

TARGET="${1:-all}"
DRY_RUN="${DRY_RUN:-0}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
PARTITION="${PARTITION:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

case "${TARGET}" in
  dinov3_fixed|profile_dinov3|all) ;;
  *) echo "Unsupported TARGET: ${TARGET}" >&2; exit 2 ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="scripts/run/jag_supplement/logs/gap_fill"
PROF_DIR="scripts/run/jag_supplement/profile_results/table7"
MANIFEST="scripts/run/jag_supplement/submissions_gap_fill_${TARGET}_${STAMP}.tsv"
mkdir -p "${LOG_ROOT}" "${PROF_DIR}"
printf "group\tname\tjob_id\texperiment\toverrides\n" > "${MANIFEST}"

sbatch_extra=()
[[ -n "${EXCLUDE_NODES}" ]] && sbatch_extra+=(--exclude "${EXCLUDE_NODES}")
[[ -n "${PARTITION}" ]] && sbatch_extra+=(--partition "${PARTITION}")

submit_train() {
  local group="$1" name="$2" experiment="$3" overrides="$4"
  local all="experiment_name=${name} ${overrides}"
  if squeue -h -u "${USER}" -n "${name}" 2>/dev/null | awk 'NF { found=1 } END { exit !found }'; then
    echo "[skip queued] ${name}"; return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${name}: ${experiment} ${all}"
    printf "%s\t%s\tDRY_RUN\t%s\t%s\n" "${group}" "${name}" "${experiment}" "${all}" >> "${MANIFEST}"
    return
  fi
  local jid
  jid="$(sbatch "${sbatch_extra[@]}" --job-name "${name}" \
    --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
    scripts/run/train_experiment.sh "${experiment}" "" "${all}" | awk '{print $4}')"
  printf "%s\t%s\t%s\t%s\t%s\n" "${group}" "${name}" "${jid}" "${experiment}" "${all}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

submit_profile() {
  local name="$1" experiment="$2" overrides="$3"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] ${name}: profile ${experiment} ${overrides}"; return
  fi
  local jid
  jid="$(sbatch --exclude "${EXCLUDE_NODES}" --job-name "${name}" \
      --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
      --partition a01 --nodes 1 --ntasks-per-node 1 --cpus-per-task 4 --gres gpu:1 --time 01:00:00 \
      --wrap "bash -lc 'cd \"${PROJECT_ROOT}\" && source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}" && conda activate segflood && python scripts/run/jag_supplement/profile_model.py --experiment ${experiment} --name \"${name}\" --overrides \"${overrides}\" --output \"${PROF_DIR}/${name}.csv\"'" \
      | awk '{print $4}')"
  printf "profile\t%s\t%s\t%s\t%s\n" "${name}" "${jid}" "${experiment}" "${overrides}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

# Shared fragments (identical to submit_alignment_repair.sh / submit_s1s2_backbone_ma.sh)
MA_GATE="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated"
NOALIGN="model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
SOFT_CAP="+model.alignment_cap_mode=soft +model.alignment_warmup_epochs=5"
INDEPENDENT="${SOFT_CAP} +model.alignment_independent_projection=true +model.alignment_projection_dim=128"
SEMANTIC="${INDEPENDENT} model.alignment_mode=semantic_local +model.alignment_exclude_boundaries=true model.alignment_patch_radius=1 model.alignment_target_weight=0.02 model.alignment_temperature=0.12"
BIAS="model.fusion.xattn_align_bias=true +model.fusion.xattn_align_bias_mode=local_cross +model.fusion.xattn_align_bias_radius=1 model.fusion.xattn_align_bias_scale=1.0"
CANON="model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"

S1S2_COMMON="seed=42 test=true trainer.min_epochs=50 data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2"
GF_COMMON="seed=42 test=true trainer.min_epochs=50 model.encoder.optical_channels=4 model.encoder.sar_channels=1"

if [[ "${TARGET}" == "all" || "${TARGET}" == "dinov3_fixed" ]]; then
  # Job 505531 (bs128) hit CUDA OOM inside the semantic pair matrix; halve the
  # batch and accumulate two steps so the effective batch stays 128.
  # Table 2 (GF) DINOv3 rows were produced with the truncated ViT-S SAR encoder.
  submit_train dinov3_fixed jag26_gfbb_dinov3_dualconcat_s42 dinov3_dinov3_gffloodnet \
    "${GF_COMMON} model.fusion.fusion_type=concat ${CANON}"
  submit_train dinov3_fixed jag26_gfbb_dinov3_ma_s42 dinov3_dinov3_gffloodnet \
    "${GF_COMMON} ${MA_GATE} ${NOALIGN}"
  # Table 5 (CAU) DINOv3 MA row: same protocol as Job 499168 (MA, agreement disabled).
  submit_train dinov3_fixed jag26_caubb_dinov3_ma_s42 dinov3_dinov3_cauflood \
    "seed=42 test=true model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_align_bias=false +model.alignment_enabled=false +model.alignment_target_weight=0.0 model.alignment_mode=disabled"
fi


if [[ "${TARGET}" == "all" || "${TARGET}" == "profile_dinov3" ]]; then
  submit_profile jag26_prof_t7_dinov3_ma_fixed dinov3_dinov3_s1s2water \
    "data.add_dem=false data.add_slope=false ${MA_GATE} ${NOALIGN}"
  submit_profile jag26_prof_t7_dinov3_dualconcat_fixed dinov3_dinov3_s1s2water \
    "data.add_dem=false data.add_slope=false model.fusion.fusion_type=concat ${CANON}"
fi

echo "Manifest: ${MANIFEST}"
