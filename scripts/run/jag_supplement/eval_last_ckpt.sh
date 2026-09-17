#!/bin/bash
#SBATCH --partition=a01,h01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
# Usage: sbatch eval_last_ckpt.sh <run_prefix> <experiment> <run_name> <mode> <degradation> <strength> "<overrides>"
# Resolves the newest logs/<run_prefix>_* directory at run time and evaluates checkpoints/last.ckpt.
PROJECT_ROOT="${PROJECT_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}"
set -euo pipefail
cd ${PROJECT_ROOT}
prefix="$1"; exp="$2"; name="$3"; mode="$4"; deg="$5"; st="$6"; ovr="$7"
d="$(ls -d "logs/${prefix}_"* | tail -1)"
ck="${d}/checkpoints/last.ckpt"
[[ -f "${ck}" ]] || { echo "missing ${ck}" >&2; exit 1; }
echo "using ${ck}"
bash scripts/run/jag_supplement/eval_robustness.sh "${exp}" "${ck}" "${name}" "${mode}" "${deg}" "${st}" "${ovr}"
