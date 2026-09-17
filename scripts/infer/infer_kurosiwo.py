#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KuroSiwo 事件级大图滑窗推理（输入: event_*_post_flood_vv_vh_*.tif；输出: 同尺寸增强掩码）
—— 完全复现旧版脚本的行为，仅将核心逻辑下放到 src/infer/kurosiwo.py
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
    ap.add_argument("--root", required=True, type=str, help="KuroSiwo 数据根目录（包含事件子目录与 pickle）")
    ap.add_argument("--out_dir", type=str, default=os.path.join("scripts", "infer", "output", "kurosiwo"))
    ap.add_argument("--events", type=str, default="all", help="逗号分隔的事件ID列表或 'all'")
    ap.add_argument("--batch_size", type=int, default=128, help="大图滑窗批量大小")
    ap.add_argument("--tile_size", type=int, default=256, help="滑窗尺寸")
    ap.add_argument("--overlap", type=int, default=64, help="滑窗重叠(像素)")
    ap.add_argument("--mosaic_scale", type=str, default="db", choices=["linear", "db"], help="大图数值域：linear先转dB；db直接按dB使用(默认)")
    ap.add_argument("--swap_bands", action="store_true", help="若大图 band 顺序为 [VH,VV] 则开启，默认假设 [VV,VH]")
    ap.add_argument("--zscore", action="store_true", help="是否在 dB 基础上做全局 z-score；默认关闭以对齐训练配置")
    ap.add_argument("--use_ratio", action="store_true", help="是否追加 vh/vv 比值通道（需与训练时 data.channels 对齐）")
    ap.add_argument("--use_dem", action="store_true", help="是否在推理时使用 DEM 大图（需与训练时 dem=True 对齐）")
    ap.add_argument("--dem_scale_mode", type=str, default="zscore", choices=["zscore", "none"], help="DEM 归一化方式，应与训练时一致")
    ap.add_argument("--dem_mean", type=float, default=93.4313, help="DEM 均值（zscore 模式）")
    ap.add_argument("--dem_std", type=float, default=1410.8382, help="DEM 标准差（zscore 模式）")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    print("[INFO] 开始 KuroSiwo 事件级推理与拼接")
    print(f"[INFO] 参数: checkpoint={args.checkpoint}")
    print(f"[INFO] 参数: root={args.root}")
    print(f"[INFO] 参数: out_dir={args.out_dir}")
    print(f"[INFO] 参数: events={args.events}")
    print(f"[INFO] 参数: tile_size={args.tile_size}, overlap={args.overlap}, batch_size={args.batch_size}")
    print(f"[INFO] 参数: mosaic_scale={args.mosaic_scale}, zscore={args.zscore}, use_ratio={args.use_ratio}, use_dem={args.use_dem}, dem_scale_mode={args.dem_scale_mode}")

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
        raise SystemExit("未在 root 下找到任何 'event_*_post_flood_vv_vh_*.tif' 大图，请先运行生成脚本后再试。")

    print(f"[INFO] 检测到已拼接的大图 {len(mosaics)} 张，将基于大图进行滑窗推理")
    print(f"[INFO] 加载模型权重: {args.checkpoint}")
    # 选择与训练一致的严格加载
    model = MultiModalSegmentationModule.load_from_checkpoint(args.checkpoint, map_location="cpu", strict=True)

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
            df = df.sort_values("_actid_sort").drop(columns=["_actid_sort"])  # 数值排序
            df["actid"] = df["actid"].astype(str)
        xlsx_path = os.path.join(args.out_dir, "event_mosaic_metrics.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Sheet1")
        print(f"[INFO] 指标已写出: {xlsx_path}")

    print(f"[INFO] 基于大图推理完成: 成功 {done}")
    print("[INFO] 全部完成。")


if __name__ == "__main__":
    main()




