"""Profile model efficiency with a synthetic batch for supplementary tables.

This script does not train or evaluate on data. It instantiates a Hydra
experiment, applies optional overrides, builds a synthetic batch, and reports
parameter counts, optional FLOPs/MACs, peak CUDA memory, and warmed-up latency.

Example:
    python scripts/run/jag_supplement/profile_model.py \
      --experiment resnet50_resnet50_gffloodnet \
      --overrides "model.fusion.fusion_type=gated model.alignment_enabled=false"
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import hydra
import rootutils
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf


ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def infer_dataset_key(experiment: str) -> str:
    if "gffloodnet" in experiment:
        return "gf_floodnet"
    if "s1s2water" in experiment:
        return "s1s2_water"
    if "cauflood" in experiment:
        return "cau_flood"
    if "kurosiwo" in experiment:
        return "kurosiwo"
    if "worldfloodsv2" in experiment:
        return "worldfloodsv2"
    raise ValueError(f"Cannot infer dataset from experiment name: {experiment}")


def split_overrides(overrides: str) -> list[str]:
    return [item for item in overrides.split() if item]


def make_batch(cfg, batch_size: int, image_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    modal_type = str(cfg.data.get("modal_type", "dual"))
    optical_channels = int(cfg.data.get("optical_channels", 4))
    sar_channels = int(cfg.data.get("sar_channels", 1))
    encoder_cfg = cfg.get("model", {}).get("encoder", {})
    model_optical_channels = int(
        encoder_cfg.get("optical_channels", optical_channels)
    )
    model_sar_channels = int(encoder_cfg.get("sar_channels", sar_channels))

    # Early-fusion experiments keep a dual-modality datamodule but concatenate
    # all channels into the sole optical encoder before the model forward.
    if model_optical_channels > 0 and model_sar_channels == 0:
        return {
            "image": torch.randn(
                batch_size,
                model_optical_channels,
                image_size,
                image_size,
                device=device,
            ),
            "mask": torch.zeros(
                batch_size,
                image_size,
                image_size,
                dtype=torch.long,
                device=device,
            ),
        }

    if modal_type == "optical":
        return {
            "image": torch.randn(batch_size, optical_channels, image_size, image_size, device=device),
            "mask": torch.zeros(batch_size, image_size, image_size, dtype=torch.long, device=device),
        }
    if modal_type == "sar":
        return {
            "image": torch.randn(batch_size, sar_channels, image_size, image_size, device=device),
            "mask": torch.zeros(batch_size, image_size, image_size, dtype=torch.long, device=device),
        }
    return {
        "image_optical": torch.randn(batch_size, optical_channels, image_size, image_size, device=device),
        "image_sar": torch.randn(batch_size, sar_channels, image_size, image_size, device=device),
        "mask": torch.zeros(batch_size, image_size, image_size, dtype=torch.long, device=device),
    }


class _ProfileWrapper(nn.Module):
    """Adapt tensor-based FLOP profilers to the model's dictionary input."""

    def __init__(self, model: nn.Module, keys: list[str]):
        super().__init__()
        self.model = model
        self.keys = keys

    def forward(self, *values: torch.Tensor):
        return self.model(dict(zip(self.keys, values, strict=True)))["main_logits"]


def profile_operations(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[float | None, float | None, str]:
    inputs = tuple(value for key, value in batch.items() if key != "mask")
    keys = [key for key in batch if key != "mask"]
    wrapper = _ProfileWrapper(model, keys)
    failures: list[str] = []

    try:
        from thop import profile as thop_profile

        macs, _ = thop_profile(wrapper, inputs=inputs, verbose=False)
        return float(2 * macs), float(macs), "thop"
    except ImportError:
        pass
    except Exception as exc:
        failures.append(f"thop failed: {exc}")

    try:
        from fvcore.nn import FlopCountAnalysis

        macs = float(FlopCountAnalysis(wrapper, inputs).total())
        return 2.0 * macs, macs, "fvcore (FLOPs=2*MACs convention)"
    except ImportError:
        if failures:
            return None, None, "; ".join(failures + ["fvcore unavailable"])
        return None, None, "unavailable (install thop or fvcore)"
    except Exception as exc:
        failures.append(f"fvcore failed: {exc}")
        return None, None, "; ".join(failures)


def format_optional(value: float | None, scale: float = 1.0) -> str:
    return "" if value is None else f"{value / scale:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--overrides", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load pretrained weights. Disabled by default because weights do not affect architecture-level profiling.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="scripts/run/jag_supplement/profile_results/model_profiles.csv",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be >= 0 and --iters must be > 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA device requested but CUDA is unavailable: {args.device}")

    dataset_key = infer_dataset_key(args.experiment)
    config_dir = str(Path(ROOT) / "configs")
    overrides = [
        f"experiment={args.experiment}",
        f"data={dataset_key}",
        *split_overrides(args.overrides),
    ]

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="train.yaml",
            overrides=overrides,
            return_hydra_config=True,
        )

    # The project path configuration uses ${hydra:runtime.output_dir}. The
    # Compose API does not populate HydraConfig unless explicitly requested and
    # registered, unlike an @hydra.main entry point.
    HydraConfig.instance().set_config(cfg)
    device = torch.device(args.device)
    # Resolve only the model subtree. Resolving the entire composed job would
    # attempt to mutate Hydra's read-only runtime subtree and is unnecessary
    # for synthetic profiling.
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if not args.pretrained and "encoder" in model_cfg:
        for key in ("optical_pretrained", "sar_pretrained"):
            if key in model_cfg.encoder:
                model_cfg.encoder[key] = False
    model = hydra.utils.instantiate(model_cfg, _recursive_=False).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch = make_batch(cfg, args.batch_size, args.image_size, device)
    with torch.no_grad():
        flops, macs, operations_backend = profile_operations(model, batch)
    # Some third-party FLOP profilers temporarily toggle training mode and do
    # not reliably restore nested modules. Latency and memory must be measured
    # in inference mode.
    model.eval()

    peak_memory_mb: float | None = None
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.warmup):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        elapsed = time.perf_counter() - start

    avg_ms = elapsed / args.iters * 1000.0
    name = args.name or args.experiment

    output = Path(ROOT) / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "experiment",
                "overrides",
                "device",
                "batch_size",
                "image_size",
                "total_params_m",
                "trainable_params_m",
                "flops_g",
                "macs_g",
                "operations_backend",
                "peak_cuda_memory_mb",
                "warmup_iters",
                "latency_iters",
                "avg_forward_ms",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "name": name,
                "experiment": args.experiment,
                "overrides": args.overrides,
                "device": str(device),
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "total_params_m": f"{total_params / 1e6:.3f}",
                "trainable_params_m": f"{trainable_params / 1e6:.3f}",
                "flops_g": format_optional(flops, 1e9),
                "macs_g": format_optional(macs, 1e9),
                "operations_backend": operations_backend,
                "peak_cuda_memory_mb": format_optional(peak_memory_mb),
                "warmup_iters": args.warmup,
                "latency_iters": args.iters,
                "avg_forward_ms": f"{avg_ms:.3f}",
            }
        )

    print(f"profile saved: {output}")
    print(f"name={name}")
    print(f"total_params_m={total_params / 1e6:.3f}")
    print(f"trainable_params_m={trainable_params / 1e6:.3f}")
    print(f"flops_g={format_optional(flops, 1e9) or 'N/A'}")
    print(f"macs_g={format_optional(macs, 1e9) or 'N/A'}")
    print(f"operations_backend={operations_backend}")
    print(
        f"peak_cuda_memory_mb={format_optional(peak_memory_mb) or 'N/A (CUDA only)'}"
    )
    print(f"warmup_iters={args.warmup}")
    print(f"avg_forward_ms={avg_ms:.3f}")


if __name__ == "__main__":
    main()
