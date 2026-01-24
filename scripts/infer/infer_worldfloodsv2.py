#!/usr/bin/env python3
"""
Minimal inference entry for WorldFloodsv2.

Implementation notes:
- Uses `Trainer.predict` + `WorldFloodsPredictWriter`
- Writer handles sliding-window inference on full scenes, GeoTIFF saving, and metric aggregation
"""

import argparse
import os

try:
    import rootutils  # type: ignore
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except Exception:
    pass
import torch
from lightning.pytorch import Trainer, seed_everything  # type: ignore
from src.models.lightning_module import MultiModalSegmentationModule
from src.data.datamodules.worldfloodsv2 import WorldFloodsv2DataModule
from src.infer.worldfloodsv2 import WorldFloodsPredictWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldFloodsv2 inference (save GeoTIFF predictions by event)")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path (.ckpt)")
    p.add_argument("--dataset-path", type=str, required=True, help="Dataset root path (contains train/val/test)")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save-format", type=str, default="auto", help="Placeholder for script compatibility (Writer outputs GeoTIFF)")
    # WF2 data controls
    p.add_argument("--wf2-channels", type=str, default="bgri", help="Channels config name (e.g., rgb|bgr|bgri|bgriswirs|...)")
    p.add_argument("--wf2-water-values", type=str, default="2", help="Comma-separated mask values treated as water (binary=1). Default: '2'")
    p.add_argument("--wf2-ignore-values", type=str, default="0,3", help="Comma-separated mask values to ignore. Default: '0,3'")
    p.add_argument("--wf2-official-normalization", action="store_true", help="Use official SENTINEL2 Z-Score normalization")
    p.add_argument("--wf2-mean", type=float, nargs="*", default=None, help="Custom mean per-channel (disables official normalization unless flag is set)")
    p.add_argument("--wf2-std", type=float, nargs="*", default=None, help="Custom std per-channel (disables official normalization unless flag is set)")
    p.add_argument("--wf2-add-mndwi-input", action="store_true", help="Append MNDWI band computed from B3/B11 to inputs")
    # Sliding window params (kept for DataModule context; Writer reads dm config internally)
    p.add_argument("--wf2-tile-size", type=int, default=1024)
    p.add_argument("--wf2-pad-size", type=int, default=32)
    p.add_argument("--wf2-multiple-of", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load model (strict load)
    model = MultiModalSegmentationModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", strict=True, weights_only=False
    )
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    # 2) DataModule (full-scene inference)
    water_vals = [int(x) for x in args.wf2_water_values.split(",") if x.strip()] if isinstance(args.wf2_water_values, str) else [2]
    ignore_vals = [int(x) for x in args.wf2_ignore_values.split(",") if x.strip()] if isinstance(args.wf2_ignore_values, str) else [3]
    off_norm = bool(args.wf2_official_normalization) or (args.wf2_mean is None and args.wf2_std is None)
    dm = WorldFloodsv2DataModule(
        root=args.dataset_path,
        num_workers=args.num_workers,
        channels=args.wf2_channels,
        target_type="binary",
        water_values=water_vals,
        ignore_index=-1,
        ignore_values=ignore_vals,
        test_use_tiles=False,
        official_normalization=off_norm,
        normalization_mean=args.wf2_mean,
        normalization_std=args.wf2_std,
        add_mndwi_input=bool(args.wf2_add_mndwi_input),
        window_size=[1024, 1024],
        sliding_window={"multiple_of": 16},
    )

    # 3) Lightning prediction (Writer handles sliding window, saving, and metrics)
    seed_everything(42, workers=True)
    precision = "16-mixed" if (args.amp and torch.cuda.is_available()) else "32-true"
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        callbacks=[WorldFloodsPredictWriter(args.output_dir, save_predictions=True, amp=bool(args.amp))],
    )
    # Model is already restored from `args.checkpoint` above; do NOT pass ckpt_path again.
    trainer.predict(model=model, datamodule=dm)


if __name__ == "__main__":
    main()


