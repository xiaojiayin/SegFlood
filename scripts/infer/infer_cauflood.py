#!/usr/bin/env python3
import argparse
import os
try:
    import rootutils  # type: ignore
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
except Exception:
    pass
import torch
torch.set_float32_matmul_precision('high')
from lightning.pytorch import Trainer, seed_everything  # type: ignore
from src.models.lightning_module import MultiModalSegmentationModule
from src.data.datamodules.cau_flood import CAUFloodDataModule
from src.infer.cauflood import CAUFloodPredictWriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAU-Flood inference & evaluation (per-tile)")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--dataset-path", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--save-format", type=str, default="auto", choices=["auto", "png", "tif", "both"])
    p.add_argument("--amp", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Load model (strict load to match training)
    model = MultiModalSegmentationModule.load_from_checkpoint(
        args.checkpoint, map_location="cpu", strict=True, weights_only=False
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.benchmark = True
    # DataModule
    dm = CAUFloodDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        modal_type="dual",
        root=args.dataset_path,
    )
    # Lightning Trainer + Writer
    seed_everything(42, workers=True)
    precision = "16-mixed" if (args.amp and torch.cuda.is_available()) else "32-true"
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        callbacks=[CAUFloodPredictWriter(args.output_dir, args.save_predictions, args.save_format)],
    )
    # Model is already restored from `args.checkpoint` above; do NOT pass ckpt_path again.
    trainer.predict(model=model, datamodule=dm)


if __name__ == "__main__":
    main()


