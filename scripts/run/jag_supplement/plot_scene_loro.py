#!/usr/bin/env python
"""Scene-level figure for the GF-FloodNet leave-one-region-out case (Supplement S11).
Rows: scenes; columns: optical RGB mosaic, reference water, MA-XAttn overlay, feature-concat overlay.
Overlay colours: blue TP, red FP, yellow FN (as in the main-text qualitative figures)."""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TP = np.array([0.15, 0.40, 0.95]); FP = np.array([0.90, 0.12, 0.10]); FN = np.array([0.98, 0.86, 0.12])

def overlay(rgb, pred, ref, cover, alpha=0.8):
    out = rgb.astype(np.float32) / 255.0
    for m, c in (((pred == 1) & (ref == 1), TP), ((pred == 1) & (ref == 0), FP), ((pred == 0) & (ref == 1), FN)):
        m = m & cover
        out[m] = out[m] * (1 - alpha) + c * alpha
    out[~cover] = 1.0
    return out

def down(a, f):
    return a[::f, ::f]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--scenes", default="Russia_013,Russia_012")
    ap.add_argument("--models", default="maxattn,concat"); ap.add_argument("--names", default="MA-XAttn,Feature concat")
    ap.add_argument("--out", required=True); ap.add_argument("--factor", type=int, default=4)
    a = ap.parse_args()
    scenes = a.scenes.split(","); models = a.models.split(","); names = a.names.split(",")
    stats = json.load(open(os.path.join(a.dir, "scene_stats.json")))["scenes"]
    ncol = 2 + len(models)
    fig, axes = plt.subplots(len(scenes), ncol, figsize=(2.3 * ncol, 2.3 * len(scenes) * 0.75 + 0.3), squeeze=False)
    for r, s in enumerate(scenes):
        d = np.load(os.path.join(a.dir, f"{s}.npz"))
        rgb, cover, ref = d["rgb"], d["cover"], d["reference"]
        f = a.factor
        rgbd, coverd, refd = down(rgb, f), down(cover, f), down(ref, f)
        refimg = np.ones(rgbd.shape, dtype=np.float32); refimg[(refd == 1) & coverd] = TP; refimg[(refd == 0) & coverd] = 0.0
        axes[r, 0].imshow(rgbd); axes[r, 1].imshow(refimg)
        for k, (m, n) in enumerate(zip(models, names)):
            pred = down(d[f"pred__{m}"], f)
            axes[r, 2 + k].imshow(overlay(rgbd, pred, refd, coverd))
            st = stats[s]["models"][m]
            axes[r, 2 + k].text(0.02, 0.03, f"IoU {st['water_iou']:.1f}  area {st['area_error_pct']:+.0f}%", transform=axes[r, 2 + k].transAxes,
                                fontsize=6.5, color="w", ha="left", va="bottom", bbox=dict(facecolor="k", alpha=0.6, pad=1.5, lw=0))
        axes[r, 0].set_ylabel(f"{s.replace('_', ' ')}\n{stats[s]['n_tiles']} tiles, {stats[s]['reference_area_km2']:.1f} km$^2$ water", fontsize=7)
        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values(): sp.set_visible(False)
    for c, t in enumerate(["Optical (GF-2)", "Reference water", *names]):
        axes[0, c].set_title(t, fontsize=8)
    plt.subplots_adjust(wspace=0.03, hspace=0.05, left=0.06, right=0.995, top=0.93, bottom=0.01)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight"); fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=150, bbox_inches="tight")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
