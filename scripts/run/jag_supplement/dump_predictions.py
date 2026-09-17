"""Dump per-tile predictions of several checkpoints under several input conditions.

Feeds the qualitative figures (Fig. 4/5 FP/FN overlays, SF5 failure cases) and the CPU
analyses (boundary F1 / width stratification, connectivity, area error).

One .npz per tile in <out>/tiles/tile_XXXXX.npz with keys
  optical (C_opt,H,W) float16, sar (C_sar,H,W) float16, mask (H,W) int8,
  pred__<tag>__<cond> (H,W) uint8            for every model tag and condition
plus <out>/index.json (tile ids, tags, conditions, checkpoints).

Conditions: clean | cloud (30 % rectangular occlusion, same mask for every model) |
            optical_only (SAR dropped) | sar_only (optical dropped).

Usage (sbatch, GPU):
  python dump_predictions.py --exp resnet50_resnet50_s1s2water --data s1s2_water \
     --run <dir> --tag M0 --run <dir> --tag MA-XAttn-R --run <dir> --tag Concat \
     --n-tiles 200 --out paper/03_实验结果与计划/pred_dump_s1s2
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd()))
os.chdir(os.environ.get("PROJECT_ROOT", os.getcwd()))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CONDITIONS = ("clean", "cloud", "optical_only", "sar_only")


def load_model(run_dir, exp, data, device):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    ckpts = sorted(f for f in os.listdir(os.path.join(run_dir, "checkpoints")) if f.startswith("water_iou_"))
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpts[-1])
    with initialize_config_dir(config_dir=os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "configs"), version_base="1.3"):
        base = compose(config_name="train.yaml", overrides=[f"experiment={exp}", f"data={data}"], return_hydra_config=True)
    HydraConfig.instance().set_config(base)
    os.environ.setdefault("PROJECT_ROOT", os.environ.get("PROJECT_ROOT", os.getcwd()))
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.model.encoder.optical_pretrained = False
    if "sar_pretrained" in cfg.model.encoder:
        cfg.model.encoder.sar_pretrained = False
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    model.on_load_checkpoint({"state_dict": sd})
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[{os.path.basename(run_dir)}] {ckpts[-1]} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    cfg.data.batch_size = 16
    cfg.data.num_workers = 4
    return model.to(device).eval(), cfg.data, ckpt_path


def select_tiles(loader, n_tiles, min_water=0.02, max_nowater_frac=0.25):
    """Deterministic tile selection: mostly water-bearing tiles plus a share of no-water tiles (for FP)."""
    tiles, n_nowater = [], 0
    for bi, batch in enumerate(loader):
        m = batch["mask"]
        frac = (m == 1).flatten(1).float().mean(1)
        for i in range(m.shape[0]):
            has_water = frac[i] >= min_water
            if not has_water:
                if n_nowater >= int(max_nowater_frac * n_tiles) or frac[i] > 0:
                    continue
                n_nowater += 1
            tiles.append({k: v[i].clone() for k, v in batch.items() if torch.is_tensor(v)})
            if len(tiles) >= n_tiles:
                return tiles
    return tiles


def cloud_mask(b, h, w, s, gen):
    side_h, side_w = max(1, int(round(h * math.sqrt(s)))), max(1, int(round(w * math.sqrt(s))))
    mask = torch.ones((b, 1, h, w))
    for i in range(b):
        y0 = int(torch.randint(0, h - side_h + 1, (1,), generator=gen))
        x0 = int(torch.randint(0, w - side_w + 1, (1,), generator=gen))
        mask[i, :, y0:y0 + side_h, x0:x0 + side_w] = 0
    return mask


@torch.no_grad()
def predict(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(batch)
    logits = out["main_logits"]
    if logits.shape[-2:] != batch["mask"].shape[-2:]:
        logits = F.interpolate(logits, size=batch["mask"].shape[-2:], mode="bilinear", align_corners=False)
    return logits.argmax(1).to(torch.uint8).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--n-tiles", type=int, default=200)
    ap.add_argument("--cloud", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    args = ap.parse_args()
    conds = [c for c in args.conditions.split(",") if c]
    os.makedirs(os.path.join(args.out, "tiles"), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models, ckpts, data_cfg = {}, {}, None
    for run, tag in zip(args.run, args.tag):
        models[tag], data_cfg, ckpts[tag] = load_model(run, args.exp, args.data, device)
    dm = hydra.utils.instantiate(data_cfg)
    dm.setup("test")
    tiles = select_tiles(dm.test_dataloader(), args.n_tiles)
    print(f"selected {len(tiles)} tiles", flush=True)

    gen = torch.Generator().manual_seed(20260907)
    bs = 16
    preds = {t: {c: [] for c in conds} for t in models}
    cmasks = []
    # GF-FloodNet normalises inside the datamodule (process_batch: percentile + mean/std); S1S2 stores
    # pre-scaled values and has no test-time transform. Apply the datamodule's own batch processing
    # so the dump feeds the model exactly what Lightning would.
    proc = getattr(dm, "process_batch", None)
    for s in range(0, len(tiles), bs):
        chunk = tiles[s:s + bs]
        batch = {k: torch.stack([t[k] for t in chunk]) for k in chunk[0]}
        if proc is not None:
            batch = proc(dict(batch), "test")
            for j, t in enumerate(chunk):  # store the model-input (normalised) tensors
                t["image_optical"], t["image_sar"], t["mask"] = batch["image_optical"][j].cpu(), batch["image_sar"][j].cpu(), batch["mask"][j].cpu()
        b, _, h, w = batch["image_optical"].shape
        cm = cloud_mask(b, h, w, args.cloud, gen)
        cmasks.append(cm[:, 0].to(torch.uint8))
        for tag, model in models.items():
            for c in conds:
                bb = {k: v for k, v in batch.items()}
                if c == "cloud":
                    bb["image_optical"] = bb["image_optical"] * cm.to(bb["image_optical"].dtype)
                elif c == "optical_only":
                    bb.pop("image_sar")
                elif c == "sar_only":
                    bb.pop("image_optical")
                preds[tag][c].append(predict(model, bb, device))
        print(f"  {s + b}/{len(tiles)}", flush=True)
    cmasks = torch.cat(cmasks)
    preds = {t: {c: torch.cat(v) for c, v in d.items()} for t, d in preds.items()}

    for i, t in enumerate(tiles):
        rec = {
            "optical": t["image_optical"].to(torch.float16).numpy(),
            "sar": t["image_sar"].to(torch.float16).numpy(),
            "mask": t["mask"].to(torch.int8).numpy(),
            "cloud_mask": cmasks[i].numpy(),
        }
        for tag in models:
            for c in conds:
                rec[f"pred__{tag}__{c}"] = preds[tag][c][i].numpy()
        np.savez_compressed(os.path.join(args.out, "tiles", f"tile_{i:05d}.npz"), **rec)
    json.dump({"n_tiles": len(tiles), "tags": list(models), "conditions": conds, "checkpoints": ckpts,
               "cloud_fraction": args.cloud, "exp": args.exp, "data": args.data},
              open(os.path.join(args.out, "index.json"), "w"), indent=2)
    # quick sanity: water IoU per tag/cond on the dumped tiles
    gt = torch.stack([t["mask"] for t in tiles])
    valid = gt >= 0
    for tag in models:
        row = []
        for c in conds:
            p = preds[tag][c].long()
            inter = ((p == 1) & (gt == 1) & valid).sum().item()
            union = (((p == 1) | (gt == 1)) & valid).sum().item()
            row.append(f"{c}={100 * inter / max(1, union):.2f}")
        print(tag, " ".join(row), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
