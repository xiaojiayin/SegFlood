#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KuroSiwo event-level sliding-window inference on mosaics.

Input:  event_*_post_flood_vv_vh_*.tif
Output: predicted masks with the same spatial size (+ optional metrics export)

This script is a thin wrapper around the core implementation in `src/infer/kurosiwo.py`.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Optional

try:
    import rootutils  # type: ignore
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except Exception:
    pass

import numpy as np  # type: ignore
import torch

from src.models.lightning_module import MultiModalSegmentationModule
from src.infer.kurosiwo import infer_on_dual_mosaic, compute_event_metrics_from_files


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--root", required=True, type=str, help="KuroSiwo dataset root (contains event dirs and pickle/)")
    ap.add_argument("--out_dir", type=str, default=os.path.join("outputs", "infer", "kurosiwo"))
    ap.add_argument("--events", type=str, default="all", help="Comma-separated event IDs or 'all'")
    ap.add_argument("--batch_size", type=int, default=128, help="Sliding-window batch size")
    ap.add_argument("--tile_size", type=int, default=256, help="Sliding window tile size")
    ap.add_argument("--overlap", type=int, default=64, help="Sliding window overlap (pixels)")
    ap.add_argument("--mosaic_scale", type=str, default="db", choices=["linear", "db"], help="Mosaic value domain: linear->dB, or db")
    ap.add_argument("--swap_bands", action="store_true", help="Set if mosaic band order is [VH,VV] (default assumes [VV,VH])")
    ap.add_argument("--zscore", action="store_true", help="Apply global z-score in dB domain (must match training config)")
    ap.add_argument("--use_ratio", action="store_true", help="Append vh/vv ratio channel (must match training config)")
    ap.add_argument("--use_dem", action="store_true", help="Use DEM mosaic (must match training config)")
    ap.add_argument("--dem_scale_mode", type=str, default="zscore", choices=["zscore", "none"], help="DEM scaling (must match training)")
    ap.add_argument("--dem_mean", type=float, default=93.4313, help="DEM mean (zscore mode)")
    ap.add_argument("--dem_std", type=float, default=1410.8382, help="DEM std (zscore mode)")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    print("[INFO] Start KuroSiwo mosaic inference")
    print(f"[INFO] checkpoint={args.checkpoint}")
    print(f"[INFO] root={args.root}")
    print(f"[INFO] out_dir={args.out_dir}")
    print(f"[INFO] events={args.events}")
    print(f"[INFO] tile_size={args.tile_size}, overlap={args.overlap}, batch_size={args.batch_size}")
    print(
        f"[INFO] mosaic_scale={args.mosaic_scale}, zscore={args.zscore}, use_ratio={args.use_ratio}, "
        f"use_dem={args.use_dem}, dem_scale_mode={args.dem_scale_mode}"
    )

    rows: List[Dict[str, object]] = []
    done = 0

    all_mosaics = [
        os.path.join(args.root, fn)
        for fn in os.listdir(args.root)
        if fn.startswith("event_") and fn.endswith(".tif") and ("_post_flood_vv_vh_" in fn)
    ] if os.path.isdir(args.root) else []
    if args.events.strip().lower() != "all":
        allow = set([s.strip() for s in args.events.split(",") if s.strip()])
        def _keep(p: str) -> bool:
            m = re.match(r"event_(\d+)_post_flood_vv_vh_", os.path.basename(p))
            return (m is not None) and (m.group(1) in allow)
        mosaics = [p for p in all_mosaics if _keep(p)]
    else:
        mosaics = sorted(all_mosaics)

    if len(mosaics) == 0:
        raise SystemExit("No mosaics found under root. Expect files like: event_*_post_flood_vv_vh_*.tif")

    print(f"[INFO] Found {len(mosaics)} mosaics. Running sliding-window inference.")
    print(f"[INFO] Load checkpoint: {args.checkpoint}")
    model = MultiModalSegmentationModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", strict=True, weights_only=False
    )

    means = (-11.870900531759382, -19.075417829198912) if args.zscore else None
    stds  = ( 6.873472038814271,   7.244895909919173) if args.zscore else None

    for tif in mosaics:
        _tile = max(1, int(args.tile_size))
        _stride = max(1, _tile - max(0, int(args.overlap)))
        out = infer_on_dual_mosaic(
            model, tif, args.out_dir, tile=_tile, stride=_stride,
            zscore=bool(args.zscore), data_mean=means, data_std=stds,
            batch_size=max(1, int(args.batch_size)),
            mosaic_scale=str(args.mosaic_scale), swap_bands=bool(args.swap_bands),
            use_ratio=bool(args.use_ratio),
            use_dem=bool(args.use_dem),
            dem_scale_mode=str(args.dem_scale_mode),
            dem_mean=float(args.dem_mean),
            dem_std=float(args.dem_std),
        )
        if out is not None:
            done += 1
            metrics = compute_event_metrics_from_files(out, args.root)
            if metrics is not None:
                rows.append(metrics)

    if rows:
        keys = [
            "actid", "date",
            "test/acc",
            "test/bg_false_alarm_rate",
            "test/bg_recall",
            "test/bg_specificity",
            "test/f1",
            "test/flood_miss_rate",
            "test/flood_recall",
            "test/flood_specificity",
            "test/iou",
            "test/precision",
            "test/recall",
            "test/specificity",
            "test/water_iou",
        ]
        import pandas as pd  # type: ignore
        data_rows = [{k: r.get(k) for k in keys} for r in rows]
        df = pd.DataFrame(data_rows)
        if "actid" in df.columns:
            df["_actid_sort"] = pd.to_numeric(df["actid"], errors="coerce")
            df = df.sort_values("_actid_sort").drop(columns=["_actid_sort"])
            df["actid"] = df["actid"].astype(str)
        xlsx_path = os.path.join(args.out_dir, "event_mosaic_metrics.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        print(f"[INFO] Metrics written: {xlsx_path}")

    print(f"[INFO] Done. Successful mosaics: {done}")


if __name__ == "__main__":
    main()




