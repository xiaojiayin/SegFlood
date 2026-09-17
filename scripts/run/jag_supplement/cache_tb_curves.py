"""One-pass extraction of per-epoch training curves from every TensorBoard run
into a compact CSV cache, so later analyses never re-read the 25 GB of event
files (DeviceStatsMonitor logs dozens of scalars per step).

    sbatch --partition a01 --cpus-per-task 48 --job-name tbcache \
      --wrap "bash -lc 'cd $PROJECT_ROOT && source .../conda.sh && conda activate segflood && python scripts/run/jag_supplement/cache_tb_curves.py'"

Output: paper/03_实验结果与计划/tb_curves_cache.csv
  columns: run, version, epoch, val_water_iou, val_iou, val_loss, train_loss, lr
plus tb_curves_summary.csv (one row per run/version: n_epochs, best_epoch,
best_val, last_val, val_loss_min_epoch, lr_at_best).
"""
import csv
import glob
import os
import sys
from multiprocessing import Pool

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
OUT = os.environ.get("TB_CACHE_OUT", os.path.join(ROOT, "paper/03_实验结果与计划/tb_curves_cache.csv"))
SUMMARY = os.environ.get("TB_CACHE_SUMMARY", os.path.join(ROOT, "paper/03_实验结果与计划/tb_curves_summary.csv"))
RUN_FILTER = os.environ.get("TB_CACHE_FILTER", "")  # regex on run name
TAGS = {"val_water_iou": "val/water_iou", "val_iou": "val/iou", "val_loss": "val/loss_main_epoch",
        "train_loss": "train/loss_main_epoch", "lr": "lr-AdamW"}
if os.environ.get("TB_CACHE_EXTRA_TAGS"):  # e.g. "loss_align=train/loss_align_epoch,nce=train/align_nce_acc"
    for kv in os.environ["TB_CACHE_EXTRA_TAGS"].split(","):
        k, t = kv.split("=", 1)
        TAGS[k] = t


def one(vd):
    run = os.path.basename(os.path.dirname(vd))
    ver = os.path.basename(vd)
    try:
        ea = EventAccumulator(vd, size_guidance={"scalars": 0})
        ea.Reload()
        tags = set(ea.Tags()["scalars"])
        if "val/water_iou" not in tags:
            return run, ver, [], None
        series = {}
        for k, t in TAGS.items():
            if t in tags:
                series[k] = {e.step: e.value for e in ea.Scalars(t)}
        steps = sorted(series["val_water_iou"])
        rows = []
        for i, st in enumerate(steps):
            rows.append([run, ver, i] + [series.get(k, {}).get(st, "") for k in TAGS])
        vals = [series["val_water_iou"][s] for s in steps]
        b = max(range(len(vals)), key=lambda i: vals[i])
        vl = [series.get("val_loss", {}).get(s, None) for s in steps]
        vl_min_ep = min((i for i, x in enumerate(vl) if x is not None), key=lambda i: vl[i], default="")
        summ = [run, ver, len(vals), b, round(vals[b] * 100, 3), round(vals[-1] * 100, 3), vl_min_ep,
                series.get("lr", {}).get(steps[b], "")]
        return run, ver, rows, summ
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {vd}: {exc}", file=sys.stderr)
        return run, ver, [], None


def main():
    import re
    dirs = [d for d in sorted(glob.glob(os.path.join(ROOT, "logs/tensorboard/*/version_*")))
            if glob.glob(os.path.join(d, "events.out.tfevents.*"))
            and (not RUN_FILTER or re.search(RUN_FILTER, os.path.basename(os.path.dirname(d))))]
    print(f"{len(dirs)} version dirs", flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))) as pool, \
            open(OUT, "w", newline="") as f, open(SUMMARY, "w", newline="") as g:
        w = csv.writer(f); w.writerow(["run", "version", "epoch"] + list(TAGS))
        s = csv.writer(g); s.writerow(["run", "version", "n_epochs", "best_epoch", "best_val_water", "last_val_water", "val_loss_min_epoch", "lr_at_best"])
        done = 0
        for run, ver, rows, summ in pool.imap_unordered(one, dirs):
            done += 1
            for r in rows:
                w.writerow(r)
            if summ:
                s.writerow(summ)
            if done % 25 == 0:
                print(f"{done}/{len(dirs)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
