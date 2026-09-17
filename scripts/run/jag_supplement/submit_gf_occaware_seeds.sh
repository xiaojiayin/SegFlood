#!/usr/bin/env bash
# GF-FloodNet occlusion-aware MA-XAttn: two further training seeds (123, 2026)
# (the Table 8b row was a single seed-42 run). Overrides are copied verbatim from
# logs/jag26_smagnet_gf_sym_align_minep50_s42_2026-09-08_20-14-28/.hydra/overrides.yaml
# (symmetric occlusion + optical-only/SAR-only heads, 50-epoch schedule, last checkpoint,
# semantic-local agreement term lambda=5e-4). After each training job finishes, the four
# degraded conditions of Table 8 are evaluated on last.ckpt with test_degradation_seed=1
# (same draw as the seed-42 row) via Slurm afterok dependencies.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
L=${PROJECT_ROOT}/logs
LOG_ROOT="scripts/run/jag_supplement/logs/gf_occaware_seeds"; mkdir -p "${LOG_ROOT}" scripts/run/jag_supplement/logs/robustness_eval
STAMP="$(date +%Y%m%d_%H%M%S)"; MANIFEST="scripts/run/jag_supplement/submissions_gf_occaware_seeds_${STAMP}.tsv"
printf "stage\tseed\tcondition\tjob_id\tdepends_on\tnote\n" > "${MANIFEST}"
SEEDS="${SEEDS:-123 2026}"

GF="model.encoder.optical_channels=4 model.encoder.sar_channels=1"
MA="model.fusion.fusion_type=xattn +model.fusion.xattn_components=spatial+channel +model.fusion.xattn_attention_type=ma +model.fusion.xattn_affinity_order=opt_sar +model.fusion.xattn_component_fusion=gated model.fusion.xattn_align_bias=false"
OCC="+model.pixel_optical_dropout_p=0.5 +model.pixel_sar_dropout_p=0.3 +model.pixel_optical_dropout_area=0.3 +model.pixel_dropout_warmup_epochs=3 +model.pixel_dropout_ramp_epochs=5 +model.sar_only_head_weight=0.5 +model.optical_only_head_weight=0.5"
ALIGN_TRAIN="model.alignment_enabled=true model.alignment_mode=semantic_local +model.alignment_in_image_only=true +model.alignment_exclude_boundaries=true model.alignment_patch_radius=1 model.alignment_temperature=0.12 +model.alignment_semantic_stop_gradient=true +model.alignment_cap_mode=none +model.alignment_warmup_epochs=2 +model.alignment_independent_projection=false model.alignment_layers=[-1,-2] model.alignment_target_weight=0.0005"
# eval-time override set that reproduces the checkpoint architecture (mirrors SYMALIGN of submit_degradation_seed_repeats.sh)
ALIGN_EVAL="model.alignment_enabled=true model.alignment_mode=semantic_local +model.alignment_in_image_only=true +model.alignment_cap_mode=none +model.alignment_independent_projection=false model.alignment_layers=[-1,-2] model.alignment_target_weight=0.0005 +model.alignment_semantic_stop_gradient=true"

declare -A COND=([optical_cloud]="dual optical_cloud 0.3" [sar_noise]="dual sar_noise 0.5" [missing_optical]="sar_only none 0" [missing_sar]="optical_only none 0")

for s in ${SEEDS}; do
  name="jag26_smagnet_gf_sym_align_minep50_s${s}"
  train_jid="$(sbatch --exclude g59,g10 --time=08:00:00 --job-name "${name}" \
      --output "${LOG_ROOT}/${name}_%j.out" --error "${LOG_ROOT}/${name}_%j.err" \
      scripts/run/train_experiment.sh resnet50_resnet50_gffloodnet "" \
      "experiment_name=${name} seed=${s} test=true trainer.min_epochs=50 ${GF} ${MA} ${OCC} ${ALIGN_TRAIN}" | awk '{print $4}')"
  printf "train\t%s\t-\t%s\t-\t%s\n" "${s}" "${train_jid}" "${name}" >> "${MANIFEST}"
  echo "[submitted] train ${name}: ${train_jid}"

  # evaluation jobs: resolve run dir at run time (timestamp suffix unknown until the job starts)
  for c in optical_cloud sar_noise missing_optical missing_sar; do
    read -r mode deg st <<< "${COND[$c]}"
    ename="jag26_rob_gf_symLAST_s${s}_${c}"
    ejid="$(sbatch --exclude g59,g10 --partition a01,h01 --gres=gpu:1 --cpus-per-task=8 --time=01:00:00 --dependency=afterok:${train_jid} --job-name "${ename}" \
        --output "scripts/run/jag_supplement/logs/robustness_eval/${ename}_%j.out" \
        --error  "scripts/run/jag_supplement/logs/robustness_eval/${ename}_%j.err" \
        scripts/run/jag_supplement/eval_last_ckpt.sh "${name}" resnet50_resnet50_gffloodnet "${ename}" "${mode}" "${deg}" "${st}" "${GF} ${MA} ${ALIGN_EVAL} +model.test_degradation_seed=1" \
        | awk '{print $4}')"
    printf "eval\t%s\t%s\t%s\t%s\t%s\n" "${s}" "${c}" "${ejid}" "${train_jid}" "${ename}" >> "${MANIFEST}"
    echo "[submitted] eval ${ename}: ${ejid} (afterok:${train_jid})"
  done
done
echo "Manifest: ${MANIFEST}"
