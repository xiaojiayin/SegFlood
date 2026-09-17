#!/usr/bin/env bash

# Leave-one-region-out GF-FloodNet holdouts for the four remaining regions
# (China, India, Pakistan, Russia). Australia and Brazil were run earlier by
# submit_gf_holdout_baselines.sh. Each region is held out entirely as the test
# set; the remaining regions are split 80/20 into train/val (seed 42).
#
# Methods follow the canonical definitions used for the S1S2 fusion table:
#   inputconcat   single-encoder early fusion (5-channel input)
#   dualconcat    stage-wise feature concatenation, no aux head
#   gated         canonical sigmoid modality gate, no aux head
#   ma            MA-XAttn spatial+channel, component gate, no agreement (main model)
#
# Usage: DRY_RUN=1 bash scripts/run/jag_supplement/submit_gf_holdout_regions.sh
#        REGIONS="China Russia" bash scripts/run/jag_supplement/submit_gf_holdout_regions.sh

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
REGIONS="${REGIONS:-China India Pakistan Russia}"
VARIANTS="${VARIANTS:-inputconcat dualconcat gated ma}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59,g40}"
PARTITION="${PARTITION:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="scripts/run/jag_supplement/logs/gf_holdout_regions"
MANIFEST="scripts/run/jag_supplement/submissions_gf_holdout_regions_${STAMP}.tsv"
mkdir -p "${LOG_DIR}"
printf "region\tvariant\tjob_id\toverrides\n" > "${MANIFEST}"

# Identical strings to canonical_baselines_gf / alignment_bias_final in
# submit_alignment_repair.sh so the holdout rows share one definition per method.
NOALIGN="model.alignment_enabled=false model.alignment_target_weight=0.0"
CANON="model.aux_loss_weight=0.0 ${NOALIGN}"
DUAL="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
MA_CORE="${DUAL} model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated"
SEMANTIC_LOSS="+model.alignment_cap_mode=soft +model.alignment_warmup_epochs=5 +model.alignment_independent_projection=true +model.alignment_projection_dim=128 model.alignment_mode=semantic_local +model.alignment_exclude_boundaries=true model.alignment_patch_radius=1 model.alignment_target_weight=0.02 model.alignment_temperature=0.12 model.alignment_enabled=true"
LOCAL_BIAS="model.fusion.xattn_align_bias=true +model.fusion.xattn_align_bias_mode=local_cross +model.fusion.xattn_align_bias_radius=1 model.fusion.xattn_align_bias_scale=1.0"

submit_one() {
  local region="$1"
  local experiment="$2"
  local variant="$3"
  local overrides="$4"
  local name="jag26_gfhold_${region}_${variant}"
  case " ${VARIANTS} " in *" ${variant} "*) ;; *) return ;; esac
  local all_overrides="seed=42 data.holdout_patterns=[${region}_*] experiment_name=${name} ${overrides}"

  if squeue -h -u "${USER}" -n "${name}" 2>/dev/null | awk 'NF { found=1 } END { exit !found }'; then
    echo "[skip queued] ${name}"
    return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\n" "${region}" "${variant}" "DRY_RUN" "${overrides}" >> "${MANIFEST}"
    echo "[dry-run] ${name}: ${experiment} ${all_overrides}"
    return
  fi

  local -a sbatch_extra=()
  [[ -n "${EXCLUDE_NODES}" ]] && sbatch_extra+=(--exclude "${EXCLUDE_NODES}")
  [[ -n "${PARTITION}" ]] && sbatch_extra+=(--partition "${PARTITION}")
  local jid
  jid="$(
    sbatch \
      "${sbatch_extra[@]}" \
      --job-name "${name}" \
      --output "${LOG_DIR}/${name}_%j.out" \
      --error "${LOG_DIR}/${name}_%j.err" \
      scripts/run/train_gffloodnet.sh "${experiment}" "${all_overrides}" \
      | awk '{print $4}'
  )"
  printf "%s\t%s\t%s\t%s\n" "${region}" "${variant}" "${jid}" "${overrides}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

for region in ${REGIONS}; do
  submit_one "${region}" "resnet50_resnet50_gffloodnet" "inputconcat" \
    "${CANON} +data.output_image_key=true model.encoder.optical_channels=5 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity"
  submit_one "${region}" "resnet50_resnet50_gffloodnet" "dualconcat" \
    "${CANON} ${DUAL} model.fusion.fusion_type=concat"
  submit_one "${region}" "resnet50_resnet50_gffloodnet" "gated" \
    "${CANON} ${DUAL} model.fusion.fusion_type=canonical_gated"
  submit_one "${region}" "resnet50_resnet50_gffloodnet" "ma" \
    "${MA_CORE} model.fusion.xattn_align_bias=false ${NOALIGN}"
done

echo "Manifest: ${MANIFEST}"
