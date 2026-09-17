#!/usr/bin/env bash
# Table 6 (S1S2 backbone x fusion): dual-stream feature-concatenation rows so the
# table has the same Concat-vs-MA structure as Table 2 (GF). Same protocol as
# the Table 2 concat reruns: aux off, agreement off, min_epochs=50, seed42, test.
# ResNet-50 concat comes from Table 5 (3 seeds) and is not rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
LOG_ROOT="scripts/run/jag_supplement/logs/t6_concat"; mkdir -p "${LOG_ROOT}"
COMMON="seed=42 test=true trainer.min_epochs=50 data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=concat model.aux_loss_weight=0.0 model.alignment_enabled=false model.alignment_target_weight=0.0"
submit() { # variant experiment overrides
  local name="jag26_t6concat_$1_s42"
  sbatch --exclude g59 --job-name "${name}" --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
    scripts/run/train_experiment.sh "$2" "" "experiment_name=${name} ${COMMON} $3" | awk -v n="${name}" '{print "[submitted] " n ": " $4}'
}
submit mobilenet    efficientnetb4_mobilenetv3_s1s2water "model.encoder.optical_model_name=mobilenetv3_large_100 model.encoder.sar_model_name=mobilenetv3_large_100 model.encoder.optical_pretrained=false model.encoder.sar_pretrained=false"
submit efficientnet efficientnetb4_mobilenetv3_s1s2water "model.encoder.optical_model_name=efficientnet_b4 model.encoder.sar_model_name=efficientnet_b4 model.encoder.sar_pretrained=false"
submit sam2         sam2_sam2_s1s2water ""
submit dinov3       dinov3_dinov3_s1s2water ""
