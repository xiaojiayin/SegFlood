"""Inventory every TensorBoard run: config summary, best validation Water IoU,
and test metrics (if the run/evaluation logged them).

    python scripts/run/jag_supplement/inventory_runs.py --out paper/03_实验结果与计划/run_inventory.tsv
"""
import argparse
import glob
import os
import re
import sys

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
KEYS = ["iou", "water_iou", "flood_miss_rate", "bg_false_alarm_rate"]


def get(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def summarize(version_dir):
    hp_path = os.path.join(version_dir, "hparams.yaml")
    hp = yaml.safe_load(open(hp_path)) if os.path.exists(hp_path) else {}
    enc = hp.get("encoder", {}) or {}
    fus = hp.get("fusion", {}) or {}
    data = hp.get("data", {}) or {}
    row = {
        "dataset": str(get(data, "_target_", default="")).split(".")[-1] or str(get(data, "dataset_name", default="")),
        "seed": hp.get("seed", ""),
        "opt_model": enc.get("optical_model_name") or enc.get("model_name", ""),
        "sar_model": enc.get("sar_model_name", "") if enc.get("sar_channels", 1) else "",
        "opt_ch": enc.get("optical_channels", ""),
        "sar_ch": enc.get("sar_channels", ""),
        "fusion": fus.get("fusion_type", ""),
        "xattn": ",".join(
            f"{k.replace('xattn_', '')}={v}"
            for k, v in sorted(fus.items())
            if k in ("xattn_attention_type", "xattn_components", "xattn_component_fusion", "xattn_affinity_order", "xattn_align_bias")
        ),
        "align_loss": hp.get("alignment_enabled", ""),
        "align_w": hp.get("alignment_target_weight", ""),
        "aux_w": hp.get("aux_loss_weight", ""),
        "dem": f"{get(data, 'add_dem', default='')}/{get(data, 'add_slope', default='')}",
        "params_M": round(float(hp.get("model/params/total", 0)) / 1e6, 1) if hp.get("model/params/total") else "",
    }
    ea = EventAccumulator(version_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    if "val/water_iou" in tags:
        vals = ea.Scalars("val/water_iou")
        best = max(vals, key=lambda e: e.value)
        row["epochs"] = len(vals)
        row["best_val_water"] = round(best.value * 100, 3)
        row["best_val_epoch"] = vals.index(best)
        for k in ("iou", "flood_miss_rate", "bg_false_alarm_rate"):
            t = f"val/{k}"
            if t in tags:
                s = ea.Scalars(t)
                m = [e for e in s if e.step == best.step]
                row[f"val_{k}@best"] = round((m[0].value if m else s[min(len(s) - 1, vals.index(best))].value) * 100, 3)
    for k in KEYS:
        t = f"test/{k}"
        if t in tags:
            row[f"test_{k}"] = round(ea.Scalars(t)[-1].value * 100, 3)
    return row


def _one(vd):
    run = os.path.basename(os.path.dirname(vd))
    try:
        row = summarize(vd)
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {vd}: {exc}", file=sys.stderr)
        return None
    row["run"] = run
    row["version"] = os.path.basename(vd)
    cands = sorted(glob.glob(os.path.join(ROOT, "logs", run + "_20*")))
    row["ckpt_dir"] = ";".join(os.path.basename(c) for c in cands if glob.glob(os.path.join(c, "checkpoints", "*.ckpt")))
    print(f"{run} {row['version']} val={row.get('best_val_water','')} test={row.get('test_water_iou','')}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--filter", default="", help="regex on run name")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    cols = ["run", "version", "dataset", "seed", "opt_model", "sar_model", "opt_ch", "sar_ch", "fusion", "xattn",
            "align_loss", "align_w", "aux_w", "dem", "params_M", "epochs", "best_val_water", "best_val_epoch",
            "val_iou@best", "val_flood_miss_rate@best", "val_bg_false_alarm_rate@best",
            "test_iou", "test_water_iou", "test_flood_miss_rate", "test_bg_false_alarm_rate", "ckpt_dir"]
    pat = re.compile(args.filter) if args.filter else None
    dirs = []
    for vd in sorted(glob.glob(os.path.join(ROOT, "logs", "tensorboard", "*", "version_*"))):
        run = os.path.basename(os.path.dirname(vd))
        if pat and not pat.search(run):
            continue
        if glob.glob(os.path.join(vd, "events.out.tfevents.*")):
            dirs.append(vd)
    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        rows = [r for r in pool.imap_unordered(_one, dirs) if r]
    rows.sort(key=lambda r: (r["run"], r["version"]))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
