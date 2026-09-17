#!/usr/bin/env python
"""Image-level paired bootstrap of dataset Water IoU between two methods.

Each input CSV is `event_metrics/test_per_image_confusion.csv` written by the
Lightning module during `test` (rows: idx,activation,tp,fp,fn,tn in test-loader
order, which is identical for every model on the same dataset).

Usage:
  python paired_bootstrap.py --a runA_s42.csv runA_s123.csv --b runB_s42.csv runB_s123.csv \
      [--iters 5000] [--seed 0] [--metric water_iou|miou|miss|fa]

Seeds are paired positionally (a[i] with b[i]); the statistic is the mean over
seed pairs of the dataset-level metric computed on a common bootstrap resample
of image indices. Reports delta (A - B), 95% percentile CI and a two-sided
bootstrap p-value (fraction of resamples whose sign differs from the point estimate).
"""
import argparse
import numpy as np


def load(path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, 2:6]  # tp, fp, fn, tn


def metric(c, name):
    tp, fp, fn, tn = (c[:, i].sum(dtype=np.float64) for i in range(4))
    if name == "water_iou":
        return tp / max(tp + fp + fn, 1)
    if name == "miou":
        return 0.5 * (tp / max(tp + fp + fn, 1) + tn / max(tn + fp + fn, 1))
    if name == "miss":
        return fn / max(tp + fn, 1)
    if name == "fa":
        return fp / max(fp + tn, 1)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="water_iou")
    args = ap.parse_args()
    if len(args.a) != len(args.b):
        raise SystemExit("--a and --b must have the same number of seed files")
    A = [load(p) for p in args.a]
    B = [load(p) for p in args.b]
    n = A[0].shape[0]
    for x in A + B:
        if x.shape[0] != n:
            raise SystemExit(f"image count mismatch: {x.shape[0]} vs {n}")
    rng = np.random.default_rng(args.seed)
    point = np.mean([metric(a, args.metric) - metric(b, args.metric) for a, b in zip(A, B)])
    deltas = np.empty(args.iters)
    for i in range(args.iters):
        idx = rng.integers(0, n, n)
        deltas[i] = np.mean([metric(a[idx], args.metric) - metric(b[idx], args.metric) for a, b in zip(A, B)])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    ma = np.mean([metric(a, args.metric) for a in A])
    mb = np.mean([metric(b, args.metric) for b in B])
    print(f"metric={args.metric} images={n} seeds={len(A)} iters={args.iters}")
    print(f"A={100*ma:.3f}  B={100*mb:.3f}  delta(A-B)={100*point:+.3f} pp  95%CI=[{100*lo:+.3f}, {100*hi:+.3f}]  p={p:.4f}")


if __name__ == "__main__":
    main()
