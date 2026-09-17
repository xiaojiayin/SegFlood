#!/usr/bin/env bash
# 2026-09-04 batch 1: the agreement loss with its implementation actually active.
#
# Findings that motivate this (JAG_ALIGNMENT_LOSS_CODE_AUDIT.md, EVIDENCE_SYNTHESIS §5):
#   * cap = 0.3 x EMA(main loss) pinned the alignment term from epoch ~3 on -> zero gradient
#     in every previous agreement run (submitted and revised)          -> alignment_cap_mode=none
#   * `alignment_stop_gradient` is not wired into semantic_local; the real switch is
#     `alignment_semantic_stop_gradient` (default False)               -> set it true (optical = teacher)
#   * independent 128-d projection decoupled the loss from the fused features -> off
#   * only the 8x8 level was aligned                                   -> layers [-1,-2]
# Recipe identical to M0 (components gated: aux 0.3, patience early stop, seed 42, test=true).
# lambda bracketed so that lambda*L_align is ~25% / ~100% of the steady-state main loss (~1e-3).
#
#   DRY_RUN=1 bash scripts/run/jag_supplement/submit_align_fixed.sh
set -euo pipefail
DRY_RUN="${DRY_RUN:-0}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
LOG_ROOT="scripts/run/jag_supplement/logs/align_fixed"; mkdir -p "${LOG_ROOT}"

S1S2="seed=42 test=true data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2"
GATE="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
FIX="+model.alignment_cap_mode=none +model.alignment_warmup_epochs=2 +model.alignment_independent_projection=false model.alignment_layers=[-1,-2] model.alignment_enabled=true"
SEM="model.alignment_mode=semantic_local +model.alignment_in_image_only=true +model.alignment_exclude_boundaries=true model.alignment_patch_radius=1 model.alignment_temperature=0.12 +model.alignment_semantic_stop_gradient=true"
DIS="model.alignment_mode=distill"

submit() { # name overrides
  local name="jag26_alignfixed_$1"
  if [[ "${DRY_RUN}" == "1" ]]; then echo "[dry-run] ${name}: $2"; return; fi
  sbatch --exclude g59 --job-name "${name}" --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
    scripts/run/train_experiment.sh resnet50_resnet50_s1s2water "" "experiment_name=${name} ${S1S2} ${GATE} $2" \
    | awk -v n="${name}" '{print "[submitted] " n ": " $4}'
}
# in_image_only: cross-image pair matrix (B*K)^2 with layer -2 (K=256) OOMs at 80 GB; in-image negatives only.
# semantic-local BCE at steady state ~2-3  -> lambda 1e-4 (~25%) and 5e-4 (~100%+)
submit s1s2_semantic_l1e-4_s42 "${FIX} ${SEM} model.alignment_target_weight=0.0001"
submit s1s2_semantic_l5e-4_s42 "${FIX} ${SEM} model.alignment_target_weight=0.0005"
# distill (1-cos, optical detached teacher) ~0.3-1 -> lambda 3e-4 and 1.5e-3
submit s1s2_distill_l3e-4_s42  "${FIX} ${DIS} model.alignment_target_weight=0.0003"
submit s1s2_distill_l1.5e-3_s42 "${FIX} ${DIS} model.alignment_target_weight=0.0015"
