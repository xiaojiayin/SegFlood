#!/usr/bin/env bash
# Batch 2 (2026-09-04 10:00): confirm batch-1 winners (semantic l5e-4, distill l1.5e-3; both 97.45 test vs M0 97.06 seed42)
#   (a) S1S2 seeds 123 / 2026 for both; (b) GF seed42 for both; (c) one higher-lambda probe each on S1S2
#       (semantic share was ~0% of total loss at 5e-4 -> 5e-3; distill 6-9% -> 5e-3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
LOG_ROOT="scripts/run/jag_supplement/logs/align_fixed"; mkdir -p "${LOG_ROOT}"
S1S2="test=true data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2"
GF="test=true model.encoder.optical_channels=4 model.encoder.sar_channels=1"
GATE="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
FIX="+model.alignment_cap_mode=none +model.alignment_warmup_epochs=2 +model.alignment_independent_projection=false model.alignment_layers=[-1,-2] model.alignment_enabled=true"
SEM="model.alignment_mode=semantic_local +model.alignment_in_image_only=true +model.alignment_exclude_boundaries=true model.alignment_patch_radius=1 model.alignment_temperature=0.12 +model.alignment_semantic_stop_gradient=true"
DIS="model.alignment_mode=distill"
submit() { # exp name overrides
  local name="jag26_alignfixed_$2"
  sbatch --exclude g59 --job-name "${name}" --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
    scripts/run/train_experiment.sh "$1" "" "experiment_name=${name} $3" | awk -v n="${name}" '{print "[submitted] " n ": " $4}'
}
for seed in 123 2026; do
  submit resnet50_resnet50_s1s2water s1s2_semantic_l5e-4_s${seed}  "seed=${seed} ${S1S2} ${GATE} ${FIX} ${SEM} model.alignment_target_weight=0.0005"
  submit resnet50_resnet50_s1s2water s1s2_distill_l1.5e-3_s${seed} "seed=${seed} ${S1S2} ${GATE} ${FIX} ${DIS} model.alignment_target_weight=0.0015"
done
submit resnet50_resnet50_gffloodnet gf_semantic_l5e-4_s42  "seed=42 ${GF} ${GATE} ${FIX} ${SEM} model.alignment_target_weight=0.0005"
submit resnet50_resnet50_gffloodnet gf_distill_l1.5e-3_s42 "seed=42 ${GF} ${GATE} ${FIX} ${DIS} model.alignment_target_weight=0.0015"
submit resnet50_resnet50_s1s2water s1s2_semantic_l5e-3_s42 "seed=42 ${S1S2} ${GATE} ${FIX} ${SEM} model.alignment_target_weight=0.005"
submit resnet50_resnet50_s1s2water s1s2_distill_l5e-3_s42  "seed=42 ${S1S2} ${GATE} ${FIX} ${DIS} model.alignment_target_weight=0.005"
