#!/usr/bin/env python
"""Inventory of every trained checkpoint and its test-set metrics.

Scans logs/*/checkpoints/water_iou_*.ckpt (best-validation checkpoint of each
training run), reads the Hydra overrides that produced it, and collects the
test metrics that already exist (end-of-training test in train.log, or a
later eval run whose experiment_name references the checkpoint). Runs
without test metrics can be submitted with --submit, which re-uses
eval_robustness.sh with the training-time model overrides so the state dict
loads unchanged.

    python scripts/run/jag_supplement/test_inventory.py            # write TSV
    python scripts/run/jag_supplement/test_inventory.py --submit   # + sbatch missing
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "logs"
OUT_DIR = ROOT / "scripts/run/jag_supplement/test_inventory"
EVAL_LOG_DIRS = [
    ROOT / "scripts/run/jag_supplement/logs",
]
SUPPORTED = {"s1s2_water": "s1s2", "gf_floodnet": "gf", "cau_flood": "cau"}
METRIC_RE = re.compile(r"test/(iou|water_iou|flood_miss_rate|bg_false_alarm_rate)[^0-9]*([0-9.]+)")
# overrides that only affect training, not the architecture / data definition
DROP_PREFIX = (
    "experiment=", "data=", "data.root=", "experiment_name=", "seed=", "test=",
    "trainer.", "callbacks", "logger", "hydra", "ckpt_path", "train=",
    "data.batch_size", "data.num_workers", "+trainer", "model.learning_rate",
    "model.weight_decay", "model.scheduler", "+model.scheduler", "tags",
)


def parse_metrics(text: str) -> dict[str, float] | None:
    found = {k: float(v) for k, v in METRIC_RE.findall(text)}
    return found if "water_iou" in found else None


def read_overrides(run_dir: Path) -> list[str]:
    f = run_dir / ".hydra" / "overrides.yaml"
    if not f.exists():
        return []
    return [ln[2:].strip() for ln in f.read_text().splitlines() if ln.startswith("- ")]


def best_ckpt(run_dir: Path) -> Path | None:
    cks = sorted(glob.glob(str(run_dir / "checkpoints" / "water_iou_*.ckpt")))
    return Path(cks[-1]) if cks else None


def existing_eval_results() -> dict[str, dict[str, float]]:
    """Map checkpoint path -> test metrics from earlier eval jobs (CKPT= line in .out)."""
    res: dict[str, dict[str, float]] = {}
    for base in EVAL_LOG_DIRS:
        for out in glob.glob(str(base / "**" / "*.out"), recursive=True):
            try:
                text = Path(out).read_text(errors="ignore")
            except OSError:
                continue
            m = re.search(r"CKPT=(\S+) MODE=dual DEG=none", text)
            if not m:
                continue
            metrics = parse_metrics(text)
            if metrics:
                res[os.path.realpath(m.group(1))] = metrics
    return res


def tb_test_metrics(name: str, run_dir: Path) -> dict[str, float] | None:
    """End-of-training test scalars from logs/tensorboard/<name>/version_*.

    Versions are matched to the run by event-file start timestamp (within
    15 min of the run directory timestamp), because version numbering is
    per experiment_name and not stored in the run directory.
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})$", run_dir.name)
    if not m:
        return None
    import datetime as dt
    t0 = dt.datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}", "%Y-%m-%d %H:%M:%S").timestamp()
    tb_root = LOGS / "tensorboard" / name
    if not tb_root.exists():
        return None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return None
    best: dict[str, float] | None = None
    for ver in sorted(tb_root.glob("version_*")):
        evs = sorted(glob.glob(str(ver / "events.out.tfevents.*")))
        if not evs:
            continue
        ts = min(float(Path(e).name.split(".")[3]) for e in evs)
        if abs(ts - t0) > 15 * 60:
            continue
        acc = EventAccumulator(str(ver), size_guidance={"scalars": 0})
        acc.Reload()
        tags = set(acc.Tags().get("scalars", []))
        if "test/water_iou" not in tags:
            continue
        out = {}
        for short, tag in (("iou", "test/iou"), ("water_iou", "test/water_iou"),
                           ("flood_miss_rate", "test/flood_miss_rate"), ("bg_false_alarm_rate", "test/bg_false_alarm_rate")):
            if tag in tags:
                out[short] = acc.Scalars(tag)[-1].value
        best = out
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--datasets", default="s1s2 gf cau")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-nodes", default="g59")
    args = ap.parse_args()
    want = set(args.datasets.split())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_hits = existing_eval_results()
    rows = []
    for run_dir in sorted(LOGS.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "tensorboard":
            continue
        ck = best_ckpt(run_dir)
        if ck is None:
            continue
        ovr = read_overrides(run_dir)
        exp = next((o.split("=", 1)[1] for o in ovr if o.startswith("experiment=")), "")
        data_key = next((o.split("=", 1)[1] for o in ovr if o.startswith("data=")), "")
        ds = SUPPORTED.get(data_key)
        if ds is None:
            continue
        seed = next((o.split("=", 1)[1] for o in ovr if o.startswith("seed=")), "")
        names = [o.split("=", 1)[1] for o in ovr if o.startswith("experiment_name=")]
        name = names[-1] if names else run_dir.name
        best_val = float(re.search(r"water_iou_(\d+\.\d+)", ck.name).group(1))
        model_ovr = " ".join(o for o in ovr if not o.startswith(DROP_PREFIX))

        metrics = eval_hits.get(os.path.realpath(ck))
        source = "eval_job" if metrics else ""
        if metrics is None:
            tl = run_dir / "train.log"
            if tl.exists():
                metrics = parse_metrics(tl.read_text(errors="ignore"))
                source = "train_log" if metrics else ""
        if metrics is None:
            metrics = tb_test_metrics(name, run_dir)
            source = "tensorboard" if metrics else ""
        rows.append({
            "dataset": ds, "experiment": exp, "name": name, "seed": seed,
            "run_dir": run_dir.name, "ckpt": str(ck.relative_to(ROOT)),
            "best_val_water": f"{best_val*100:.2f}",
            "test_miou": f"{metrics['iou']*100:.2f}" if metrics and "iou" in metrics else "",
            "test_water": f"{metrics['water_iou']*100:.2f}" if metrics else "",
            "test_miss": f"{metrics['flood_miss_rate']*100:.2f}" if metrics and "flood_miss_rate" in metrics else "",
            "test_fa": f"{metrics['bg_false_alarm_rate']*100:.2f}" if metrics and "bg_false_alarm_rate" in metrics else "",
            "test_source": source, "model_overrides": model_ovr,
        })

    out = OUT_DIR / "checkpoint_test_inventory.tsv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader(); w.writerows(rows)
    missing = [r for r in rows if not r["test_water"] and r["dataset"] in want]
    print(f"runs={len(rows)} with_test={sum(1 for r in rows if r['test_water'])} missing={len(missing)} -> {out}")

    if not args.submit:
        return
    log_root = OUT_DIR / "slurm"
    log_root.mkdir(exist_ok=True)
    manifest = OUT_DIR / "submitted.tsv"
    n = 0
    with manifest.open("a") as mf:
        for r in missing:
            if args.limit and n >= args.limit:
                break
            job = f"jag26_inv_{r['run_dir'][:60]}"
            cmd = [
                "sbatch", "--exclude", args.exclude_nodes, "--job-name", job,
                "--output", str(log_root / f"{job}_%j.out"), "--error", str(log_root / f"{job}_%j.err"),
                "scripts/run/jag_supplement/eval_robustness.sh", r["experiment"], str(ROOT / r["ckpt"]),
                job, "dual", "none", "0.0", r["model_overrides"],
            ]
            jid = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()[-1]
            mf.write(f"{r['run_dir']}\t{jid}\t{r['ckpt']}\n")
            n += 1
            print(f"[submitted] {job}: {jid}")


if __name__ == "__main__":
    sys.exit(main())
