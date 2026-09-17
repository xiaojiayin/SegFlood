#!/usr/bin/env python
"""Scene-level flood maps for the GF-FloodNet leave-one-region-out models.

For every run directory (trained with data.holdout_patterns=[<Region>_*]) the held-out tiles are
predicted, stitched back into whole scenes using the tile geotransforms, and compared with the
reference mask: per-scene Water IoU, reference / predicted water area (km^2) and relative area error.

Usage:
  python predict_scene_loro.py --run logs/jag26_gfhold_Russia_ma_* --tag "MA-XAttn" ... --out <dir>
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.environ.get("PROJECT_ROOT", os.getcwd()))
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import rasterio as rio  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def load(run_dir, device):
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    ckpts = sorted(f for f in os.listdir(os.path.join(run_dir, "checkpoints")) if f.startswith("water_iou_"))
    ckpt_path = os.path.join(run_dir, "checkpoints", ckpts[-1])
    with initialize_config_dir(config_dir=os.path.join(os.environ.get("PROJECT_ROOT", os.getcwd()), "configs"), version_base="1.3"):
        base = compose(config_name="train.yaml", overrides=["experiment=resnet50_resnet50_gffloodnet", "data=gf_floodnet"], return_hydra_config=True)
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
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("test")
    return model.to(device).eval(), dm, ckpt_path


@torch.no_grad()
def predict_all(model, dm, device):
    ds = dm.test_dataset  # Subset over the held-out tiles
    files = [ds.dataset.files[i] for i in ds.indices]
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    preds, gts = [], []
    for batch in loader:
        batch = dm.process_batch({k: v for k, v in batch.items()}, "test")
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(batch)
        logits = out["main_logits"]
        if logits.shape[-2:] != batch["mask"].shape[-2:]:
            logits = F.interpolate(logits, size=batch["mask"].shape[-2:], mode="bilinear", align_corners=False)
        preds.append(logits.argmax(1).to(torch.uint8).cpu())
        gts.append(batch["mask"].to(torch.int8).cpu())
    return files, torch.cat(preds).numpy(), torch.cat(gts).numpy()


def stitch(files, arrays):
    """arrays: dict name -> (N,H,W). Returns dict scene -> dict name -> mosaic, plus rgb mosaic and pixel size."""
    scenes = {}
    meta = []
    for f in files:
        with rio.open(f) as r:
            t = r.transform
            meta.append((os.path.basename(f).rsplit("_", 2)[0], t.c, t.f, abs(t.a), r.height, r.width))
    for s in sorted(set(m[0] for m in meta)):
        idx = [i for i, m in enumerate(meta) if m[0] == s]
        xs = [meta[i][1] for i in idx]; ys = [meta[i][2] for i in idx]
        px = meta[idx[0]][3]; h, w = meta[idx[0]][4], meta[idx[0]][5]
        x0, y0 = min(xs), max(ys)
        W = int(round((max(xs) - x0) / px)) + w
        H = int(round((y0 - min(ys)) / px)) + h
        out = {n: np.full((H, W), 255, dtype=np.uint8) for n in arrays}
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        cover = np.zeros((H, W), dtype=bool)
        for i in idx:
            c = int(round((meta[i][1] - x0) / px)); r_ = int(round((y0 - meta[i][2]) / px))
            for n, a in arrays.items():
                out[n][r_:r_ + h, c:c + w] = a[i]
            with rio.open(files[i]) as rr:
                img = rr.read([3, 2, 1]).astype(np.float32)  # R,G,B
            rgb[r_:r_ + h, c:c + w] = np.moveaxis(img, 0, -1)
            cover[r_:r_ + h, c:c + w] = True
        lo, hi = np.percentile(rgb[cover], [2, 98], axis=0)
        rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        rgb[~cover] = 1.0
        scenes[s] = {"masks": out, "rgb": rgb, "cover": cover, "px_m": px, "n_tiles": len(idx)}
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--tag", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files0, preds, gt = None, {}, None
    ckpts = {}
    for run, tag in zip(args.run, args.tag):
        model, dm, ckpts[tag] = load(run, device)
        files, p, g = predict_all(model, dm, device)
        if files0 is None:
            files0, gt = files, g
        else:
            assert files == files0, "held-out tile order differs between runs"
        preds[tag] = p
        del model; torch.cuda.empty_cache()
    arrays = {"reference": gt.astype(np.uint8)}
    arrays.update({f"pred__{t}": p for t, p in preds.items()})
    scenes = stitch(files0, arrays)
    stats = {}
    for s, d in scenes.items():
        ref = d["masks"]["reference"]; valid = ref != 255
        px_km2 = (d["px_m"] ** 2) / 1e6
        ref_area = float((ref == 1).sum() * px_km2)
        row = {"n_tiles": d["n_tiles"], "pixel_m": d["px_m"], "reference_area_km2": ref_area, "models": {}}
        for t, p in preds.items():
            pm = d["masks"][f"pred__{t}"]
            inter = int(((pm == 1) & (ref == 1) & valid).sum()); union = int((((pm == 1) | (ref == 1)) & valid).sum())
            area = float(((pm == 1) & valid).sum() * px_km2)
            fp = int(((pm == 1) & (ref == 0) & valid).sum()); fn = int(((pm == 0) & (ref == 1) & valid).sum())
            row["models"][t] = {"water_iou": 100 * inter / max(union, 1), "pred_area_km2": area,
                                "area_error_pct": 100 * (area - ref_area) / max(ref_area, 1e-9),
                                "fp_km2": fp * px_km2, "fn_km2": fn * px_km2}
        stats[s] = row
        np.savez_compressed(os.path.join(args.out, f"{s}.npz"), rgb=(d["rgb"] * 255).astype(np.uint8), cover=d["cover"], **d["masks"])
        print(s, json.dumps(row, indent=1), flush=True)
    json.dump({"checkpoints": ckpts, "scenes": stats}, open(os.path.join(args.out, "scene_stats.json"), "w"), indent=2)
    print("done", flush=True)


if __name__ == "__main__":
    main()
