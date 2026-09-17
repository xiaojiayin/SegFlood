#!/usr/bin/env python3
"""
S1S2-Water 测试集推理并按场景拼接大图（GeoTIFF）。

改为 Lightning 最佳实践：
- 使用 Trainer.predict + S1S2WaterPredictWriter
- Writer 内部完成 tiles 预测保存与 mosaics 拼接
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

try:
    import rootutils  # type: ignore
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except Exception:
    pass

import torch
from lightning.pytorch import Trainer, seed_everything  # type: ignore
from src.models.lightning_module import MultiModalSegmentationModule
from src.data.datamodules.s1s2_water import S1S2WaterDataModule
from src.infer.s1s2_water import S1S2WaterPredictWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S1S2-Water 推理并拼接场景大图")
    p.add_argument("--checkpoint", required=True, type=str, help="模型权重 .ckpt")
    p.add_argument("--data-root", required=True, type=str, help="数据根目录（包含 S1S2-Water/train|val|test）")
    p.add_argument("--out-dir", required=True, type=str, help="输出目录")
    p.add_argument("--modal-type", type=str, default=None, choices=["optical", "sar", "dual"], help="可选：覆盖模态类型")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", help="CUDA 上启用 autocast fp16")
    p.add_argument("--add-dem", action="store_true", help="是否在光学分支启用 DEM 通道（需与训练时一致）")
    p.add_argument("--add-slope", action="store_true", help="是否在光学分支启用坡度通道（需与训练时一致）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    model = MultiModalSegmentationModule.load_from_checkpoint(args.checkpoint, map_location="cpu", strict=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    print(
        f"[S1S2-Water-INFER] checkpoint={args.checkpoint}, "
        f"modal_type={args.modal_type or 'dual'}, add_dem={args.add_dem}, add_slope={args.add_slope}"
    )

    dm = S1S2WaterDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        modal_type=(args.modal_type or "dual"),
        add_dem=bool(args.add_dem),
        add_slope=bool(args.add_slope),
        root=args.data_root,
    )

    seed_everything(42, workers=True)
    precision = "16-mixed" if (args.amp and torch.cuda.is_available()) else "32-true"

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        callbacks=[S1S2WaterPredictWriter(args.out_dir, save_predictions=True, modal_type=(args.modal_type or "dual"))],
    )
    trainer.predict(model=model, datamodule=dm, ckpt_path=args.checkpoint)


if __name__ == "__main__":
    main()


