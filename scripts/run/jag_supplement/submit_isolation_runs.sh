#!/usr/bin/env bash
# Factor-isolation runs requested in the R1 self-review (S1S2-Water, ResNet-50, three seeds each):
#   (A) sym_noalign : the final symmetric occlusion-aware configuration WITHOUT the agreement term
#                     (isolates the agreement term inside the final recipe)
#   (B) std_minep50 : the standard protocol (no occlusion, no auxiliary passes, no agreement) trained for the
#                     full 50 epochs so that the LAST checkpoint can be compared with the occlusion-aware model
#                     (isolates the checkpoint rule / schedule)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
LOG_ROOT="scripts/run/jag_supplement/logs/isolation"; mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"; MANIFEST="scripts/run/jag_supplement/submissions_isolation_${STAMP}.tsv"
printf "variant\tseed\tjob_id\toverrides\n" > "${MANIFEST}"
BASE="data.add_dem=false data.add_slope=false model.encoder.optical_channels=4 model.encoder.sar_channels=2 model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false model.alignment_enabled=false model.alignment_target_weight=0.0 trainer.min_epochs=50 test=true"
OCC="+model.pixel_optical_dropout_p=0.5 +model.pixel_sar_dropout_p=0.3 +model.pixel_optical_dropout_area=0.3 +model.pixel_dropout_warmup_epochs=3 +model.pixel_dropout_ramp_epochs=5 +model.sar_only_head_weight=0.5 +model.optical_only_head_weight=0.5"
submit() { # variant seed overrides
  local name="jag26_iso_$1_s$2"
  local jid; jid="$(sbatch --exclude g59,g10 --time=26:00:00 --job-name "${name}" --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" scripts/run/train_experiment.sh resnet50_resnet50_s1s2water "" "experiment_name=${name} seed=$2 $3" | awk '{print $4}')"
  printf "%s\t%s\t%s\t%s\n" "$1" "$2" "${jid}" "$3" >> "${MANIFEST}"; echo "[submitted] ${name}: ${jid}"
}
for s in 42 123 2026; do
  submit sym_noalign "${s}" "${BASE} ${OCC}"
  submit std_minep50 "${s}" "${BASE}"
done
echo "Manifest: ${MANIFEST}"
