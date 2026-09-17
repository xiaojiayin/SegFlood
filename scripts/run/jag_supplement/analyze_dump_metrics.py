#!/usr/bin/env python3
"""CPU-only metrics from a dump_predictions.py tile dump.

Reads <dump>/index.json and tiles/tile_XXXXX.npz, then writes pooled and
per-tile water / boundary / width / connectivity / area metrics. Ignore
pixels (mask == -1) are excluded everywhere. For cond == cloud, pixel
metrics are also split into inside- vs outside-occlusion.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from skimage.segmentation import find_boundaries

BF_TOLS = (1, 2, 3)
CC_STRUCT = np.ones((3, 3), dtype=np.int8)
MIN_CC_AREA = 10
MIN_AREA_TILE = 100
# local width ≈ 2 * EDT; bins follow the paper revision plan (px).
WIDTH_BINS = (
    ("le3", "<=3", 0.0, 3.0),
    ("4_10", "4-10", 3.0, 10.0),
    ("11_30", "11-30", 10.0, 30.0),
    ("gt30", ">30", 30.0, math.inf),
)


def _safe_div(num, den):
    return float(num) / float(den) if den else float("nan")


def _prf(tp, fp, fn):
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    if tp + fp == 0 and tp + fn == 0:
        prec = rec = 1.0
    iou = _safe_div(tp, tp + fp + fn)
    if tp + fp + fn == 0:
        iou = 1.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * prec * rec / (prec + rec)
    return {"iou": iou, "precision": prec, "recall": rec, "f1": f1}


def _f1(p, r):
    if np.isnan(p) or np.isnan(r):
        return float("nan")
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def valid_pred_water(pred, valid):
    return (pred == 1) & valid


def confusion(pred_w, gt_w, valid):
    tp = int((pred_w & gt_w).sum())
    fp = int((pred_w & ~gt_w & valid).sum())
    fn = int((~pred_w & gt_w).sum())
    return tp, fp, fn


def boundary_hits(gt_w, pred_w, valid, tols=BF_TOLS):
    """Pooled BF ingredients: pred/gt boundary counts and in-tolerance hits."""
    gb = find_boundaries(gt_w, mode="inner") & valid
    pb = find_boundaries(pred_w, mode="inner") & valid
    out = {}
    n_pb, n_gb = int(pb.sum()), int(gb.sum())
    for tol in tols:
        selem = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
        hit_p = int((pb & binary_dilation(gb, structure=selem)).sum()) if n_pb else 0
        hit_g = int((gb & binary_dilation(pb, structure=selem)).sum()) if n_gb else 0
        out[tol] = {"n_pred": n_pb, "n_gt": n_gb, "hit_pred": hit_p, "hit_gt": hit_g}
    return out


def width_hits(gt_w, pred_w):
    """Recall of GT water pixels binned by the *object* width of the connected water component
    they belong to. Object width = 2 * median EDT over the component's skeleton-like core
    (pixels whose EDT is >= 0.8 * the component max EDT), which measures the typical
    cross-section of a river / pond instead of the distance of each pixel to the shoreline.
    All pixels of a component share its width, so the <=3 px bin really contains narrow rivers."""
    counts = {k: {"tp": 0, "n": 0} for k, *_ in WIDTH_BINS}
    if not gt_w.any():
        return counts
    edt = distance_transform_edt(gt_w)
    lbl, n = label(gt_w, structure=CC_STRUCT)
    width = np.zeros(gt_w.shape, dtype=np.float32)
    for c in range(1, n + 1):
        comp = lbl == c
        e = edt[comp]
        core = e >= 0.8 * e.max()
        width[comp] = 2.0 * float(np.median(e[core]))
    w = width[gt_w]
    hit = pred_w[gt_w]
    for key, _lab, lo, hi in WIDTH_BINS:
        sel = (w > lo) & (w <= hi) if lo > 0 else (w <= hi)
        counts[key]["n"] = int(sel.sum())
        counts[key]["tp"] = int(hit[sel].sum())
    return counts


def connectivity_stats(gt_w, pred_w):
    gt_lbl, n_gt = label(gt_w, structure=CC_STRUCT)
    pr_lbl, n_pr = label(pred_w, structure=CC_STRUCT)
    n_gt_big = n_pred_big = hit = spur = 0
    fragments = []
    for i in range(1, n_gt + 1):
        comp = gt_lbl == i
        area = int(comp.sum())
        if area < MIN_CC_AREA:
            continue
        n_gt_big += 1
        labs = np.unique(pr_lbl[comp])
        labs = labs[labs > 0]
        n_ov = int(labs.size)
        fragments.append(n_ov)
        if n_ov:
            hit += 1
    for i in range(1, n_pr + 1):
        comp = pr_lbl == i
        if int(comp.sum()) < MIN_CC_AREA:
            continue
        n_pred_big += 1
        if not gt_w[comp].any():
            spur += 1
    return {
        "n_gt": int(n_gt),
        "n_pred": int(n_pr),
        "n_gt_big": n_gt_big,
        "n_pred_big": n_pred_big,
        "hit": hit,
        "spur": spur,
        "fragments_sum": int(sum(fragments)),
        "fragments_n": len(fragments),
    }


def load_tile(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def pred_key(tag, cond):
    return f"pred__{tag}__{cond}"


def new_acc():
    acc = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "n_valid": 0,
        "bf": {t: {"n_pred": 0, "n_gt": 0, "hit_pred": 0, "hit_gt": 0} for t in BF_TOLS},
        "width": {k: {"tp": 0, "n": 0} for k, *_ in WIDTH_BINS},
        "cc": defaultdict(int),
        "area_rel": [],
        "sum_a_pred": 0,
        "sum_a_gt": 0,
        "n_area_tiles": 0,
        "occ": {
            "inside": {"tp": 0, "fp": 0, "fn": 0, "n_valid": 0},
            "outside": {"tp": 0, "fp": 0, "fn": 0, "n_valid": 0},
        },
    }
    return acc


def add_cc(dst, src):
    for k, v in src.items():
        dst[k] += v


def finalize(acc, cond):
    pix = _prf(acc["tp"], acc["fp"], acc["fn"])
    pix["n_valid"] = acc["n_valid"]
    pix["tp"], pix["fp"], pix["fn"] = acc["tp"], acc["fp"], acc["fn"]

    boundary = {}
    for t in BF_TOLS:
        b = acc["bf"][t]
        if b["n_pred"] == 0 and b["n_gt"] == 0:
            p = r = 1.0
        else:
            p = _safe_div(b["hit_pred"], b["n_pred"]) if b["n_pred"] else 0.0
            r = _safe_div(b["hit_gt"], b["n_gt"]) if b["n_gt"] else 0.0
        boundary[str(t)] = {
            "precision": p,
            "recall": r,
            "f1": _f1(p, r),
            "n_pred": b["n_pred"],
            "n_gt": b["n_gt"],
        }

    width = {}
    for key, lab, *_ in WIDTH_BINS:
        w = acc["width"][key]
        width[lab] = {"recall": _safe_div(w["tp"], w["n"]), "n": w["n"], "tp": w["tp"]}

    cc = acc["cc"]
    connectivity = {
        "n_gt": cc["n_gt"],
        "n_pred": cc["n_pred"],
        "ratio": _safe_div(cc["n_pred"], cc["n_gt"]),
        "component_recall": _safe_div(cc["hit"], cc["n_gt_big"]),
        "spurious_rate": _safe_div(cc["spur"], cc["n_pred_big"]),
        "mean_fragments": _safe_div(cc["fragments_sum"], cc["fragments_n"]),
        "n_gt_ge10": cc["n_gt_big"],
        "n_pred_ge10": cc["n_pred_big"],
    }

    rel = np.asarray(acc["area_rel"], dtype=np.float64)
    area = {
        "median_abs_rel": float(np.median(rel)) if rel.size else float("nan"),
        "mean_abs_rel": float(np.mean(rel)) if rel.size else float("nan"),
        "bias": _safe_div(acc["sum_a_pred"] - acc["sum_a_gt"], acc["sum_a_gt"]),
        "n_tiles": acc["n_area_tiles"],
    }

    out = {
        "pixel": pix,
        "boundary": boundary,
        "width_recall": width,
        "connectivity": connectivity,
        "area": area,
        "occlusion": None,
    }
    if cond == "cloud":
        out["occlusion"] = {
            "inside": {
                **_prf(acc["occ"]["inside"]["tp"], acc["occ"]["inside"]["fp"], acc["occ"]["inside"]["fn"]),
                **{k: acc["occ"]["inside"][k] for k in ("tp", "fp", "fn", "n_valid")},
            },
            "outside": {
                **_prf(acc["occ"]["outside"]["tp"], acc["occ"]["outside"]["fp"], acc["occ"]["outside"]["fn"]),
                **{k: acc["occ"]["outside"][k] for k in ("tp", "fp", "fn", "n_valid")},
            },
        }
    return out


def fmt(x, pct=False, nd=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if pct:
        return f"{100.0 * x:.{nd}f}"
    return f"{x:.{nd}f}"


def write_outputs(results, out_dir, pixel_size, n_tiles, dump, index):
    out_dir.mkdir(parents=True, exist_ok=True)
    width_keys = [lab for _k, lab, *_ in WIDTH_BINS]
    meta = {
        "dump": str(dump),
        "pixel_size_m": pixel_size,
        "n_tiles": n_tiles,
        "tags": index["tags"],
        "conditions": index["conditions"],
        "width_bins_px": width_keys,
        "width_bins_m": [
            f"<= {3 * pixel_size:g} m",
            f"{4 * pixel_size:g}–{10 * pixel_size:g} m",
            f"{11 * pixel_size:g}–{30 * pixel_size:g} m",
            f"> {30 * pixel_size:g} m",
        ],
        "bf_tolerances_px": list(BF_TOLS),
        "min_component_area_px": MIN_CC_AREA,
        "min_tile_area_for_rel_error_px": MIN_AREA_TILE,
        "results": results,
    }

    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    tsv_cols = [
        "tag", "cond", "subset",
        "iou", "precision", "recall", "f1",
        "n_valid", "tp", "fp", "fn",
        "bf1_tol1", "bf1_tol2", "bf1_tol3",
        "bf_prec_tol2", "bf_rec_tol2",
        "width_le3_recall", "width_le3_n",
        "width_4_10_recall", "width_4_10_n",
        "width_11_30_recall", "width_11_30_n",
        "width_gt30_recall", "width_gt30_n",
        "n_gt_cc", "n_pred_cc", "cc_ratio",
        "component_recall", "spurious_rate", "mean_fragments",
        "area_rel_err_median", "area_rel_err_mean", "area_bias", "n_tiles_area",
    ]

    def row_from(tag, cond, subset, pix, full=None):
        r = {c: "" for c in tsv_cols}
        r.update(tag=tag, cond=cond, subset=subset)
        for k in ("iou", "precision", "recall", "f1", "n_valid", "tp", "fp", "fn"):
            r[k] = pix.get(k, "")
        if full is not None:
            r["bf1_tol1"] = full["boundary"]["1"]["f1"]
            r["bf1_tol2"] = full["boundary"]["2"]["f1"]
            r["bf1_tol3"] = full["boundary"]["3"]["f1"]
            r["bf_prec_tol2"] = full["boundary"]["2"]["precision"]
            r["bf_rec_tol2"] = full["boundary"]["2"]["recall"]
            wr = full["width_recall"]
            r["width_le3_recall"], r["width_le3_n"] = wr["<=3"]["recall"], wr["<=3"]["n"]
            r["width_4_10_recall"], r["width_4_10_n"] = wr["4-10"]["recall"], wr["4-10"]["n"]
            r["width_11_30_recall"], r["width_11_30_n"] = wr["11-30"]["recall"], wr["11-30"]["n"]
            r["width_gt30_recall"], r["width_gt30_n"] = wr[">30"]["recall"], wr[">30"]["n"]
            cc, ar = full["connectivity"], full["area"]
            r["n_gt_cc"], r["n_pred_cc"], r["cc_ratio"] = cc["n_gt"], cc["n_pred"], cc["ratio"]
            r["component_recall"], r["spurious_rate"], r["mean_fragments"] = (
                cc["component_recall"], cc["spurious_rate"], cc["mean_fragments"],
            )
            r["area_rel_err_median"] = ar["median_abs_rel"]
            r["area_rel_err_mean"] = ar["mean_abs_rel"]
            r["area_bias"] = ar["bias"]
            r["n_tiles_area"] = ar["n_tiles"]
        return r

    tsv_rows = []
    for rec in results:
        tsv_rows.append(row_from(rec["tag"], rec["cond"], "all", rec["pixel"], rec))
        if rec["occlusion"]:
            tsv_rows.append(row_from(rec["tag"], rec["cond"], "inside_occlusion", rec["occlusion"]["inside"]))
            tsv_rows.append(row_from(rec["tag"], rec["cond"], "outside_occlusion", rec["occlusion"]["outside"]))

    tsv_path = out_dir / "metrics.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(tsv_cols) + "\n")
        for r in tsv_rows:
            f.write("\t".join("" if r[c] == "" else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])) for c in tsv_cols) + "\n")
    return json_path, tsv_path


def print_markdown(results):
    headers = ["tag", "cond", "WaterIoU", "BF@2", "Rec≤3px", "CompRec", "Spurious", "MedAreaErr"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for rec in results:
        print(
            "| "
            + " | ".join(
                [
                    rec["tag"],
                    rec["cond"],
                    fmt(rec["pixel"]["iou"], pct=True, nd=1),
                    fmt(rec["boundary"]["2"]["f1"], pct=True, nd=1),
                    fmt(rec["width_recall"]["<=3"]["recall"], pct=True, nd=1),
                    fmt(rec["connectivity"]["component_recall"], pct=True, nd=1),
                    fmt(rec["connectivity"]["spurious_rate"], pct=True, nd=1),
                    fmt(rec["area"]["median_abs_rel"], pct=True, nd=1),
                ]
            )
            + " |"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pixel-size", type=float, default=10.0, help="metres per pixel (labels only)")
    args = ap.parse_args()

    index = json.loads((args.dump / "index.json").read_text())
    tags = list(index["tags"])
    conds = list(index["conditions"])
    n_tiles = int(index["n_tiles"])
    tile_dir = args.dump / "tiles"

    accs = {(tag, cond): new_acc() for tag in tags for cond in conds}
    per_tile_cols = [
        "tile_id", "tag", "cond",
        "iou", "precision", "recall", "f1",
        "bf1_tol1", "bf1_tol2", "bf1_tol3",
        "a_gt", "a_pred", "area_rel_err",
        "n_gt_cc", "n_pred_cc",
        "width_le3_recall", "width_le3_n",
        "width_4_10_recall", "width_4_10_n",
        "width_11_30_recall", "width_11_30_n",
        "width_gt30_recall", "width_gt30_n",
    ]
    per_tile_rows = []

    for i in range(n_tiles):
        rec = load_tile(tile_dir / f"tile_{i:05d}.npz")
        mask = rec["mask"].astype(np.int16)
        valid = mask != -1
        gt_w = mask == 1
        cloud = rec["cloud_mask"].astype(np.uint8)
        for tag in tags:
            for cond in conds:
                pred = rec[pred_key(tag, cond)]
                pred_w = valid_pred_water(pred, valid)
                tp, fp, fn = confusion(pred_w, gt_w, valid)
                pix = _prf(tp, fp, fn)
                bf = boundary_hits(gt_w, pred_w, valid)
                wh = width_hits(gt_w, pred_w)
                cc = connectivity_stats(gt_w, pred_w)
                a_gt = int(gt_w.sum())
                a_pred = int(pred_w.sum())
                area_rel = abs(a_pred - a_gt) / a_gt if a_gt >= MIN_AREA_TILE else float("nan")

                acc = accs[(tag, cond)]
                acc["tp"] += tp
                acc["fp"] += fp
                acc["fn"] += fn
                acc["n_valid"] += int(valid.sum())
                for t in BF_TOLS:
                    for k in acc["bf"][t]:
                        acc["bf"][t][k] += bf[t][k]
                for key, *_ in WIDTH_BINS:
                    acc["width"][key]["tp"] += wh[key]["tp"]
                    acc["width"][key]["n"] += wh[key]["n"]
                add_cc(acc["cc"], cc)
                if a_gt >= MIN_AREA_TILE:
                    acc["area_rel"].append(area_rel)
                    acc["sum_a_pred"] += a_pred
                    acc["sum_a_gt"] += a_gt
                    acc["n_area_tiles"] += 1

                if cond == "cloud":
                    for name, extra in (("inside", cloud == 0), ("outside", cloud == 1)):
                        v2 = valid & extra
                        gt2 = gt_w & extra
                        pr2 = pred_w & extra
                        t2, f2, n2 = confusion(pr2, gt2, v2)
                        acc["occ"][name]["tp"] += t2
                        acc["occ"][name]["fp"] += f2
                        acc["occ"][name]["fn"] += n2
                        acc["occ"][name]["n_valid"] += int(v2.sum())

                wrow = {}
                for key, lab, *_ in WIDTH_BINS:
                    wrow[f"width_{key}_recall"] = _safe_div(wh[key]["tp"], wh[key]["n"])
                    wrow[f"width_{key}_n"] = wh[key]["n"]
                per_tile_rows.append({
                    "tile_id": i,
                    "tag": tag,
                    "cond": cond,
                    "iou": pix["iou"],
                    "precision": pix["precision"],
                    "recall": pix["recall"],
                    "f1": pix["f1"],
                    "bf1_tol1": (
                        1.0 if bf[1]["n_pred"] == 0 and bf[1]["n_gt"] == 0
                        else _f1(
                            0.0 if bf[1]["n_pred"] == 0 else bf[1]["hit_pred"] / bf[1]["n_pred"],
                            0.0 if bf[1]["n_gt"] == 0 else bf[1]["hit_gt"] / bf[1]["n_gt"],
                        )
                    ),
                    "bf1_tol2": (
                        1.0 if bf[2]["n_pred"] == 0 and bf[2]["n_gt"] == 0
                        else _f1(
                            0.0 if bf[2]["n_pred"] == 0 else bf[2]["hit_pred"] / bf[2]["n_pred"],
                            0.0 if bf[2]["n_gt"] == 0 else bf[2]["hit_gt"] / bf[2]["n_gt"],
                        )
                    ),
                    "bf1_tol3": (
                        1.0 if bf[3]["n_pred"] == 0 and bf[3]["n_gt"] == 0
                        else _f1(
                            0.0 if bf[3]["n_pred"] == 0 else bf[3]["hit_pred"] / bf[3]["n_pred"],
                            0.0 if bf[3]["n_gt"] == 0 else bf[3]["hit_gt"] / bf[3]["n_gt"],
                        )
                    ),
                    "a_gt": a_gt,
                    "a_pred": a_pred,
                    "area_rel_err": area_rel,
                    "n_gt_cc": cc["n_gt"],
                    "n_pred_cc": cc["n_pred"],
                    **wrow,
                })

    results = []
    for tag in tags:
        for cond in conds:
            rec = finalize(accs[(tag, cond)], cond)
            rec["tag"] = tag
            rec["cond"] = cond
            results.append(rec)

    args.out.mkdir(parents=True, exist_ok=True)
    write_outputs(results, args.out, args.pixel_size, n_tiles, args.dump, index)
    with (args.out / "per_tile.tsv").open("w", encoding="utf-8") as f:
        f.write("\t".join(per_tile_cols) + "\n")
        for r in per_tile_rows:
            f.write(
                "\t".join(
                    "" if (isinstance(r[c], float) and math.isnan(r[c]))
                    else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]))
                    for c in per_tile_cols
                )
                + "\n"
            )
    print_markdown(results)
    print(f"wrote {args.out / 'metrics.tsv'}", flush=True)
    print(f"wrote {args.out / 'metrics.json'}", flush=True)
    print(f"wrote {args.out / 'per_tile.tsv'}", flush=True)


if __name__ == "__main__":
    main()
