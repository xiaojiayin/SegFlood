#!/usr/bin/env bash

# Launchers for the matched fusion comparisons of the R1 manuscript.
#
# Usage:
#   bash scripts/run/jag_supplement/submit_alignment_repair.sh components               # MA-XAttn, ResNet-50, S1S2-Water (Tables 5, 6, 8a)
#   bash scripts/run/jag_supplement/submit_alignment_repair.sh canonical_baselines      # five fusion baselines, S1S2-Water (Tables 5, 8a)
#   PARTITION=h01 bash scripts/run/jag_supplement/submit_alignment_repair.sh canonical_baselines_gf   # same on GF-FloodNet (Table S3)
#   PARTITION=h01 bash scripts/run/jag_supplement/submit_alignment_repair.sh gf_homogeneous_backbones # backbone pairs on GF-FloodNet (Table 2)
#   bash scripts/run/jag_supplement/submit_alignment_repair.sh gf_homogeneous_minep     # lightweight pairs on GF-FloodNet (Table 2)
#   bash scripts/run/jag_supplement/submit_alignment_repair.sh cau_early                # early fusion on CAU-Flood (Table 4)
#
# Set DRY_RUN=1 to print without sbatch.

set -euo pipefail

TARGET="${1:-components}"
DRY_RUN="${DRY_RUN:-0}"
SEEDS="${SEEDS:-42 123 2026}"
VALIDATION_EPOCHS="${VALIDATION_EPOCHS:-20}"
EXCLUDE_NODES="${EXCLUDE_NODES:-g59}"
PARTITION="${PARTITION:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

case "${TARGET}" in
  diagnostic|diagnostic_gf|diagnostic_dino|candidates|components|component_residual|alignment_sweep|gf_baselines|final_selected|early_corrected|s1_early_optim_matched|shared_aux_ablation|canonical_baselines|canonical_baselines_gf|alignment_bias_final|alignment_rescue|gf_homogeneous_backbones|gf_homogeneous_minep|cau_early|robustness_gf|alignment_v2_gf|final|dino) ;;
  *)
    echo "Unsupported TARGET: ${TARGET}" >&2
    exit 2
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="scripts/run/jag_supplement/logs/alignment_repair"
MANIFEST="scripts/run/jag_supplement/submissions_alignment_repair_${TARGET}_${STAMP}.tsv"
mkdir -p "${LOG_ROOT}/${TARGET}"
printf "target\tvariant\tseed\tjob_id\toverrides\n" > "${MANIFEST}"

submit_train() {
  local experiment="$1"
  local variant="$2"
  local seed="$3"
  local overrides="$4"
  local name="jag26_alignfix_${TARGET}_${variant}_s${seed}"

  if squeue -h -u "${USER}" -n "${name}" 2>/dev/null | awk 'NF { found=1 } END { exit !found }'; then
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${TARGET}" "${variant}" "${seed}" "SKIPPED_QUEUED" "${overrides}" >> "${MANIFEST}"
    echo "[skip queued] ${name}"
    return
  fi

  local all_overrides="seed=${seed} experiment_name=${name} ${overrides}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${TARGET}" "${variant}" "${seed}" "DRY_RUN" "${overrides}" >> "${MANIFEST}"
    echo "[dry-run] ${name}: ${all_overrides}"
    return
  fi

  local jid
  local -a sbatch_extra=()
  if [[ -n "${EXCLUDE_NODES}" ]]; then
    sbatch_extra+=(--exclude "${EXCLUDE_NODES}")
  fi
  if [[ -n "${PARTITION}" ]]; then
    sbatch_extra+=(--partition "${PARTITION}")
  fi
  jid="$(
    sbatch \
      "${sbatch_extra[@]}" \
      --job-name "${name}" \
      --output "${LOG_ROOT}/${TARGET}/${name}_%j.out" \
      --error "${LOG_ROOT}/${TARGET}/${name}_%j.err" \
      scripts/run/train_experiment.sh "${experiment}" "" "${all_overrides}" \
      | awk '{print $4}'
  )"
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${TARGET}" "${variant}" "${seed}" "${jid}" "${overrides}" >> "${MANIFEST}"
  echo "[submitted] ${name}: ${jid}"
}

S1S2_BASE="data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=xattn"
GF_BASE="model.encoder.optical_channels=4 model.encoder.sar_channels=1 model.fusion.fusion_type=xattn"
MA_OPTIONS="+model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar"
S1S2_COMMON="${S1S2_BASE} ${MA_OPTIONS}"
GF_COMMON="${GF_BASE} ${MA_OPTIONS}"
SOFT_CAP="+model.alignment_cap_mode=soft +model.alignment_warmup_epochs=5"
INDEPENDENT="${SOFT_CAP} +model.alignment_independent_projection=true +model.alignment_projection_dim=128"
SEMANTIC="${INDEPENDENT} model.alignment_mode=semantic_local +model.alignment_exclude_boundaries=true model.alignment_patch_radius=1"
LOCAL_BIAS="${SEMANTIC} model.fusion.xattn_align_bias=true +model.fusion.xattn_align_bias_mode=local_cross +model.fusion.xattn_align_bias_radius=1 model.fusion.xattn_align_bias_scale=1.0"
SOFT_CORRESPONDENCE="${INDEPENDENT} model.alignment_mode=soft_correspondence +model.alignment_exclude_boundaries=true model.alignment_patch_radius=2 +model.alignment_position_weight=0.25 +model.alignment_stop_gradient=true"
PROTOTYPE="${INDEPENDENT} model.alignment_mode=prototype +model.alignment_exclude_boundaries=true +model.alignment_stop_gradient=true"

if [[ "${TARGET}" == "gf_homogeneous_backbones" ]]; then
  # Replace the duplicated hybrid EfficientNet-optical/MobileNet-SAR row with
  # true homogeneous dual-stream backbone comparisons.
  HOMO_COMMON="test=false model.encoder.optical_channels=4 model.encoder.sar_channels=1 model.encoder.optical_pretrained=true model.encoder.sar_pretrained=false"
  submit_train "efficientnetb4_mobilenetv3_gffloodnet" "gf_mobilenet_dualconcat" "42" \
    "${HOMO_COMMON} model.encoder.optical_pretrained=false model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.sar_model_name=mobilenetv3_large_100 model.fusion.fusion_type=concat model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_train "efficientnetb4_mobilenetv3_gffloodnet" "gf_mobilenet_ma" "42" \
    "${HOMO_COMMON} model.encoder.optical_pretrained=false model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.sar_model_name=mobilenetv3_large_100 model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_train "efficientnetb4_mobilenetv3_gffloodnet" "gf_efficientnet_dualconcat" "42" \
    "${HOMO_COMMON} model.encoder.optical_model_name=efficientnet_b4 model.encoder.sar_model_name=efficientnet_b4 model.fusion.fusion_type=concat model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_train "efficientnetb4_mobilenetv3_gffloodnet" "gf_efficientnet_ma" "42" \
    "${HOMO_COMMON} model.encoder.optical_model_name=efficientnet_b4 model.encoder.sar_model_name=efficientnet_b4 model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
elif [[ "${TARGET}" == "gf_homogeneous_minep" ]]; then
  # The first homogeneous MA runs were stopped by patience=10 at epochs 29/53,
  # before CosineAnnealingLR (T_max=50) reached its minimum, whereas the concat
  # runs trained to epochs 76-78. Guarantee the full annealing cycle for every
  # method with the same min_epochs so the comparison shares one protocol.
  HOMO_COMMON="test=false trainer.min_epochs=50 model.encoder.optical_channels=4 model.encoder.sar_channels=1 model.encoder.optical_pretrained=true model.encoder.sar_pretrained=false"
  MA_HOMO="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
  CONCAT_HOMO="model.fusion.fusion_type=concat model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
  submit_train "efficientnetb4_mobilenetv3_gffloodnet" "gf_mobilenet_dualconcat" "42" \
    "${HOMO_COMMON} model.encoder.optical_pretrained=false model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.sar_model_name=mobilenetv3_large_100 ${CONCAT_HOMO}"
  submit_train "efficientnetb4_mobilenetv3_gffloodnet" "gf_mobilenet_ma" "42" \
    "${HOMO_COMMON} model.encoder.optical_pretrained=false model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.sar_model_name=mobilenetv3_large_100 ${MA_HOMO}"
  submit_train "efficientnetb4_mobilenetv3_gffloodnet" "gf_efficientnet_dualconcat" "42" \
    "${HOMO_COMMON} model.encoder.optical_model_name=efficientnet_b4 model.encoder.sar_model_name=efficientnet_b4 ${CONCAT_HOMO}"
  submit_train "efficientnetb4_mobilenetv3_gffloodnet" "gf_efficientnet_ma" "42" \
    "${HOMO_COMMON} model.encoder.optical_model_name=efficientnet_b4 model.encoder.sar_model_name=efficientnet_b4 ${MA_HOMO}"
elif [[ "${TARGET}" == "cau_early" ]]; then
  # Same-period ResNet-50 early-fusion counterpart of Job 499168 (agreement
  # disabled MA-XAttn) so that Table 5 pairs two runs from one protocol.
  submit_train "resnet50_early_fusion_cauflood" "cau_resnet50_early" "42" "test=true"
elif [[ "${TARGET}" == "canonical_baselines_gf" ]]; then
  GF_CANONICAL_COMMON="test=false model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
  for seed in ${SEEDS}; do
    submit_train "resnet50_resnet50_gffloodnet" "gf_canonical_inputconcat" "${seed}" \
      "${GF_CANONICAL_COMMON} +data.output_image_key=true model.encoder.optical_channels=5 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity"
    submit_train "resnet50_resnet50_gffloodnet" "gf_canonical_dualadd" "${seed}" \
      "${GF_CANONICAL_COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=1 model.fusion.fusion_type=canonical_add"
    submit_train "resnet50_resnet50_gffloodnet" "gf_canonical_dualconcat" "${seed}" \
      "${GF_CANONICAL_COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=1 model.fusion.fusion_type=concat"
    submit_train "resnet50_resnet50_gffloodnet" "gf_canonical_stdcross" "${seed}" \
      "${GF_CANONICAL_COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=1 model.fusion.fusion_type=canonical_cross"
    submit_train "resnet50_resnet50_gffloodnet" "gf_canonical_gated" "${seed}" \
      "${GF_CANONICAL_COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=1 model.fusion.fusion_type=canonical_gated"
  done
elif [[ "${TARGET}" == "canonical_baselines" ]]; then
  # Canonical/default fusion systems on the primary S1S2 benchmark.
  # They share data, Focal main loss, augmentation, optimizer, 50 epochs and
  # validation selection, but do not inherit MA projections/residual/gate/
  # alignment or Dice auxiliary heads.
  CANONICAL_COMMON="data.add_dem=false data.add_slope=false test=false model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
  for seed in ${SEEDS}; do
    submit_train "resnet50_resnet50_s1s2water" "s1s2_canonical_inputconcat" "${seed}" \
      "${CANONICAL_COMMON} +data.output_image_key=true model.encoder.optical_channels=6 model.encoder.sar_channels=0 model.encoder.sar_pretrained=false model.fusion.fusion_type=identity"
    submit_train "resnet50_resnet50_s1s2water" "s1s2_canonical_dualadd" "${seed}" \
      "${CANONICAL_COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=canonical_add"
    submit_train "resnet50_resnet50_s1s2water" "s1s2_canonical_dualconcat" "${seed}" \
      "${CANONICAL_COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=concat"
    submit_train "resnet50_resnet50_s1s2water" "s1s2_canonical_stdcross" "${seed}" \
      "${CANONICAL_COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=canonical_cross"
    submit_train "resnet50_resnet50_s1s2water" "s1s2_canonical_gated" "${seed}" \
      "${CANONICAL_COMMON} model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=canonical_gated"
  done
elif [[ "${TARGET}" == "components" ]]; then
  for dataset in s1s2 gf; do
    if [[ "${dataset}" == "s1s2" ]]; then
      COMPONENT_EXPERIMENT="resnet50_resnet50_s1s2water"
      COMPONENT_COMMON="${S1S2_BASE}"
    else
      COMPONENT_EXPERIMENT="resnet50_resnet50_gffloodnet"
      COMPONENT_COMMON="${GF_BASE}"
    fi
    COMPONENT_COMMON="${COMPONENT_COMMON} test=false +model.fusion.xattn_attention_type=ma model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0"
    for seed in ${SEEDS}; do
      submit_train "${COMPONENT_EXPERIMENT}" "${dataset}_spatial" "${seed}" \
        "${COMPONENT_COMMON} +model.fusion.xattn_components=spatial +model.fusion.xattn_affinity_order=opt_sar"
      submit_train "${COMPONENT_EXPERIMENT}" "${dataset}_channel" "${seed}" \
        "${COMPONENT_COMMON} +model.fusion.xattn_components=channel +model.fusion.xattn_affinity_order=opt_sar"
      submit_train "${COMPONENT_EXPERIMENT}" "${dataset}_sum" "${seed}" \
        "${COMPONENT_COMMON} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=opt_sar"
      submit_train "${COMPONENT_EXPERIMENT}" "${dataset}_reverse" "${seed}" \
        "${COMPONENT_COMMON} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=sar_opt"
      submit_train "${COMPONENT_EXPERIMENT}" "${dataset}_gated" "${seed}" \
        "${COMPONENT_COMMON} +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated"
    done
  done
fi

echo "Manifest: ${MANIFEST}"
