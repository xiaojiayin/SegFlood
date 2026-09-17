#!/usr/bin/env python3
import os
"""Collect test Water IoU (%) from the robustness-eval Slurm logs of the isolation runs and the
GF occlusion-aware seed repeats, and print per-seed values plus mean +- SD in Table 8 layout.

Usage: python scripts/run/jag_supplement/collect_isolation_results.py [--md out.md]
"""
import re, sys, glob, statistics, pathlib
ROOT = pathlib.Path(os.environ.get("PROJECT_ROOT", os.getcwd()))
LOGDIR = ROOT / "scripts/run/jag_supplement/logs/robustness_eval"
COND = ["clean", "optical_cloud", "missing_optical", "missing_sar", "sar_noise"]
COND_LABEL = {"clean": "Clean", "optical_cloud": "Occlusion 30%", "missing_optical": "Missing optical",
              "missing_sar": "Missing SAR", "sar_noise": "SAR noise"}
GROUPS = {
    # label -> (glob prefix, seeds, condition -> job-name suffix)
    "S1S2 MA-XAttn occlusion-aware, NO agreement (sym_noalign, last ckpt)": ("jag26_isoeval_sym_noalign", [42, 123, 2026]),
    "S1S2 MA-XAttn standard protocol, 50 ep, LAST ckpt (std_minep50)": ("jag26_isoeval_std_minep50", [42, 123, 2026]),
    "GF MA-XAttn occlusion-aware (with agreement), last ckpt, seeds 123/2026": ("jag26_rob_gf_symLAST", [123, 2026]),
}
PAT = re.compile(r"test/water_iou\s*│\s*([0-9.]+)")

def latest_value(prefix, seed, cond):
    files = sorted(glob.glob(str(LOGDIR / f"{prefix}_s{seed}_{cond}_*.out")), key=lambda p: pathlib.Path(p).stat().st_mtime)
    for f in reversed(files):
        m = PAT.findall(pathlib.Path(f).read_text(errors="ignore"))
        if m:
            return 100 * float(m[-1])
    return None

def fmt(vals):
    v = [x for x in vals if x is not None]
    if not v: return "--"
    if len(v) == 1: return f"{v[0]:.2f} (1 seed)"
    return f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f} (n={len(v)})"

out = []
for label, (prefix, seeds) in GROUPS.items():
    out.append(f"\n### {label}\n")
    out.append("| seed | " + " | ".join(COND_LABEL[c] for c in COND) + " |")
    out.append("|---|" + "---|" * len(COND))
    per = {c: [] for c in COND}
    for s in seeds:
        row = []
        for c in COND:
            v = latest_value(prefix, s, c)
            per[c].append(v)
            row.append("--" if v is None else f"{v:.2f}")
        out.append(f"| {s} | " + " | ".join(row) + " |")
    out.append("| **mean±SD** | " + " | ".join(fmt(per[c]) for c in COND) + " |")
# GF seed-42 archived row for reference
out.append("\nGF seed-42 archived (Table 8b): clean 96.70 / occl 96.61 / miss-opt 80.20 / miss-SAR 94.79 / noise 96.14")
out.append("S1S2 occlusion-aware WITH agreement (Table 8a): 97.36±0.06 / 96.68±0.17 / 83.79±0.76 / 96.83±0.53 / 97.35±0.06")
out.append("S1S2 MA-XAttn standard, best-val (Table 8a): 97.30±0.30 / 68.5±0.2 / 0.4±0.7 / 49.8±29.6 / 97.3±0.3")
out.append("S1S2 Feature concat standard, best-val (Table 8a): 97.28±0.06 / 70.1±2.6 / 8.0±6.8 / 12.8±1.7 / 97.2±0.1")
text = "\n".join(out)
print(text)
if "--md" in sys.argv:
    pathlib.Path(sys.argv[sys.argv.index("--md") + 1]).write_text(text)
