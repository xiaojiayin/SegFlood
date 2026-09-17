#!/usr/bin/env python3
"""Qualitative FP/FN overlay panels from a dump_predictions.py tile dump.

One row per selected tile; columns are Optical RGB, SAR, GT, then one
FP/FN overlay per --tags. Optical uses dump channel order B,G,R,NIR
(display RGB = channels 2,1,0). Ignore pixels are grey; cloud occlusion
is hatched on the Optical column when --cond cloud.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# Optical dump order is B, G, R, NIR (see dump_predictions.py / dataset).
RGB_IDX = (2, 1, 0)
WATER_RANGE = (0.15, 0.70)
COLOR_TP = np.array([0.15, 0.40, 0.95])
COLOR_FP = np.array([0.90, 0.12, 0.10])
COLOR_FN = np.array([0.98, 0.86, 0.12])
COLOR_IGN = np.array([0.55, 0.55, 0.55])
COLOR_WATER = np.array([0.15, 0.40, 0.95])


def load_tile(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def pred_key(tag, cond):
    return f"pred__{tag}__{cond}"


def percentile_stretch(arr, lo=2.0, hi=98.0):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        a, b = np.percentile(finite, (lo, hi))
        if b <= a:
            return np.zeros_like(arr, dtype=np.float32)
        return np.clip((arr - a) / (b - a), 0.0, 1.0)
    out = np.empty(arr.shape, dtype=np.float32)
    for c in range(arr.shape[-1]):
        out[..., c] = percentile_stretch(arr[..., c], lo, hi)
    return out


def optical_rgb(optical):
    """optical is (C,H,W); first four channels are B,G,R,NIR."""
    rgb = np.stack([optical[i] for i in RGB_IDX], axis=-1)
    return percentile_stretch(rgb)


def sar_grey(sar):
    return percentile_stretch(np.asarray(sar[0], dtype=np.float32))


def tile_iou(pred, mask):
    valid = mask != -1
    gt = mask == 1
    pr = (pred == 1) & valid
    inter = int((pr & gt).sum())
    union = int(((pr | gt) & valid).sum())
    return 1.0 if union == 0 else inter / union


def select_tiles(dump, index, tags, cond, n, mode, ids):
    n_all = int(index["n_tiles"])
    if mode == "ids":
        chosen = [int(x) for x in ids.split(",") if x.strip() != ""]
        return chosen[:n] if n and n < len(chosen) else chosen

    ious_first, ious_last = [], []
    first, last = tags[0], tags[-1]
    for i in range(n_all):
        rec = load_tile(dump / "tiles" / f"tile_{i:05d}.npz")
        mask = rec["mask"]
        ious_first.append(tile_iou(rec[pred_key(first, cond)], mask))
        ious_last.append(tile_iou(rec[pred_key(last, cond)], mask))
    drop = np.asarray(ious_first) - np.asarray(ious_last)

    if mode == "random":
        rng = np.random.default_rng(0)
        order = rng.permutation(n_all)
    elif mode == "water":
        # water-rich tiles: GT water fraction inside [lo, hi], then a fixed-seed draw among them
        lo, hi = WATER_RANGE
        fracs = []
        for i in range(n_all):
            m = load_tile(dump / "tiles" / f"tile_{i:05d}.npz")["mask"]
            fracs.append(float((m == 1).mean()))
        cand = [i for i, f in enumerate(fracs) if lo <= f <= hi]
        rng = np.random.default_rng(0)
        order = np.asarray(rng.permutation(cand) if cand else rng.permutation(n_all))
    elif mode == "worst":
        # largest drop of first-tag IoU vs last-tag IoU (most negative first-last)
        order = np.argsort(drop, kind="stable")
    elif mode == "best":
        order = np.argsort(-drop, kind="stable")
    else:
        raise ValueError(f"unknown --select {mode}")
    chosen = order[: min(n, n_all)].tolist()
    return chosen, drop, ious_first, ious_last


def overlay_rgba(pred, mask, base_rgb, alpha=0.80):
    """TP blue / FP red / FN yellow over base; ignore grey; TN transparent."""
    h, w = mask.shape
    valid = mask != -1
    gt = mask == 1
    pr = (pred == 1) & valid
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = base_rgb
    rgba[..., 3] = 1.0
    color = np.zeros((h, w, 3), dtype=np.float32)
    paint = np.zeros((h, w), dtype=bool)
    for sel, col in (
        (pr & gt, COLOR_TP),
        (pr & ~gt & valid, COLOR_FP),
        (~pr & gt, COLOR_FN),
        (~valid, COLOR_IGN),
    ):
        color[sel] = col
        paint[sel] = True
    a = np.where(paint, alpha, 0.0).astype(np.float32)
    a[~valid] = 1.0
    out = rgba.copy()
    out[..., :3] = rgba[..., :3] * (1.0 - a[..., None]) + color * a[..., None]
    return out


def gt_panel(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[mask == 1] = COLOR_WATER
    rgb[mask == -1] = COLOR_IGN
    return rgb


def add_cloud_hatch(ax, cloud, h, w):
    occl = cloud == 0
    if not occl.any():
        return
    wash = np.zeros((h, w, 4), dtype=np.float32)
    wash[..., :3] = 1.0
    wash[..., 3] = np.where(occl, 0.35, 0.0)
    ax.imshow(wash, interpolation="nearest")
    ys, xs = np.where(occl)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ax.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            x1 - x0,
            y1 - y0,
            linewidth=0.4,
            edgecolor="white",
            facecolor="none",
            hatch="////",
            alpha=0.85,
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tags", default="M0,MA-XAttn-R,Concat")
    ap.add_argument("--cond", default="clean")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--select", default="worst", choices=("worst", "best", "random", "water", "ids"))
    ap.add_argument("--water-range", default="0.15,0.70", help="GT water fraction range for --select water")
    ap.add_argument("--ids", default="")
    ap.add_argument("--base", default="sar", choices=("sar", "rgb"))
    ap.add_argument("--rename", default="", help="display names, e.g. 'MA-XAttn=MA-XAttn (std.),MA-XAttn-R=MA-XAttn'")
    args = ap.parse_args()

    index = json.loads((args.dump / "index.json").read_text())
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    cond = args.cond
    missing = [t for t in tags if t not in index["tags"]]
    if missing:
        raise SystemExit(f"tags not in dump index: {missing}")
    if cond not in index["conditions"]:
        raise SystemExit(f"condition {cond!r} not in dump index")

    global WATER_RANGE
    WATER_RANGE = tuple(float(x) for x in args.water_range.split(","))
    sel = select_tiles(args.dump, index, tags, cond, args.n, args.select, args.ids)
    chosen = sel if args.select == "ids" else sel[0]

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"tile_ids_{cond}_{args.select}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tile_id", "select", "cond", "iou_first", "iou_last", "drop"] + [f"iou_{t}" for t in tags])
        for i in chosen:
            rec = load_tile(args.dump / "tiles" / f"tile_{i:05d}.npz")
            ious = [tile_iou(rec[pred_key(t, cond)], rec["mask"]) for t in tags]
            d = (ious[0] - ious[-1]) if len(ious) >= 2 else 0.0
            w.writerow(
                [i, args.select, cond, f"{ious[0]:.6f}", f"{ious[-1]:.6f}", f"{d:.6f}"]
                + [f"{x:.6f}" for x in ious]
            )

    n_row = len(chosen)
    n_col = 3 + len(tags)
    fig_w = 7.2
    cell = fig_w / n_col
    fig_h = cell * n_row + 0.42
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"wspace": 0.04, "hspace": 0.06},
    )
    rename = dict(kv.split("=", 1) for kv in args.rename.split(",") if "=" in kv)
    col_titles = ["Optical", "SAR", "Reference"] + [rename.get(t, t) for t in tags]

    for r, tid in enumerate(chosen):
        rec = load_tile(args.dump / "tiles" / f"tile_{tid:05d}.npz")
        mask = rec["mask"]
        h, w = mask.shape
        rgb = optical_rgb(rec["optical"])
        sar = sar_grey(rec["sar"])
        sar_rgb = np.stack([sar, sar, sar], axis=-1)
        base_rgb = rgb if args.base == "rgb" else sar_rgb

        panels = [rgb, sar_rgb, gt_panel(mask)]
        ious = []
        for tag in tags:
            pred = rec[pred_key(tag, cond)]
            panels.append(overlay_rgba(pred, mask, base_rgb))
            ious.append(tile_iou(pred, mask))

        for c, img in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(img, interpolation="nearest")
            if c == 0 and cond == "cloud":
                add_cloud_hatch(ax, rec["cloud_mask"].astype(np.uint8), h, w)
            ax.set_axis_off()
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, pad=3)
            if c >= 3:
                ax.text(
                    0.03,
                    0.05,
                    f"{100.0 * ious[c - 3]:.1f}",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="white",
                    ha="left",
                    va="bottom",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.0, "edgecolor": "none"},
                )

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_TP, markersize=6, label="TP"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_FP, markersize=6, label="FP"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_FN, markersize=6, label="FN"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLOR_IGN, markersize=6, label="ignore"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.07)
    stem = args.out / f"overlays_{cond}_{args.select}"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {stem.with_suffix('.pdf')}", flush=True)
    print(f"wrote {stem.with_suffix('.png')}", flush=True)
    print(f"wrote {csv_path}", flush=True)
    print("tiles:", ",".join(str(i) for i in chosen), flush=True)


if __name__ == "__main__":
    main()
