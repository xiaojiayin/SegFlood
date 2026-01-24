import os
import re
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window


def enhance_mask_for_qgis(input_path: str, output_path: str) -> None:
    with rasterio.open(input_path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    # Normalize value range to {0,255}: map 1 -> 255 for visualization tools (e.g., QGIS).
    data = np.where(data == 1, 255, data).astype(np.uint8)
    profile.update({
        "photometric": "palette",
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 3,
    })
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)
        colormap = {
            0: (0, 0, 0, 255),        # background=black
            255: (255, 0, 0, 255),    # water=red
            3: (211, 211, 211, 255),  # nodata=light gray
        }
        dst.write_colormap(1, colormap)
        dst.set_band_description(1, "Flood Mask (0=Land,255=Water,3=NoData)")


def _db_scale(two_band: np.ndarray) -> np.ndarray:
    eps = 1e-7
    x = np.clip(two_band, eps, None).astype(np.float32, copy=False)
    x = np.log10(x)
    x *= 10.0
    return x


@torch.inference_mode()
def infer_on_dual_mosaic(
    model,
    dual_tif: str,
    out_dir: str,
    tile: int = 256,
    stride: int = 128,
    zscore: bool = False,
    data_mean: Optional[Tuple[float, float]] = None,
    data_std: Optional[Tuple[float, float]] = None,
    batch_size: int = 128,
    mosaic_scale: str = "linear",  # "linear" or "db" (source scale of the SAR channels)
    swap_bands: bool = False,
    use_ratio: bool = False,
    use_dem: bool = False,
    dem_scale_mode: str = "zscore",  # "zscore" or "none"
    dem_mean: float = 93.4313,
    dem_std: float = 1410.8382,
) -> Optional[str]:
    if not os.path.isfile(dual_tif):
        return None
    base = os.path.basename(dual_tif)
    m = re.match(r"event_(\d+)_post_flood_vv_vh_(20\d{6}|unknown)\.tif", base)
    if not m:
        return None
    event_id, date = m.group(1), m.group(2)
    enh_out = os.path.join(out_dir, f"event_{event_id}_flood_mask_{date}_enhanced.tif")
    if os.path.exists(enh_out):
        print(f"[INFO] Event {event_id}: output already exists and will be overwritten: {enh_out}")

    print(f"[INFO] Event {event_id}: running inference on mosaic -> {dual_tif} (tile={tile}, stride={stride})")
    if use_dem:
        print(f"[INFO] Event {event_id}: DEM channel enabled (dem_scale_mode={dem_scale_mode})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()
    model.eval()

    # If DEM is enabled, infer the DEM mosaic path.
    dem_src = None
    dem_path = None
    if use_dem:
        dem_path = os.path.join(os.path.dirname(dual_tif), f"event_{event_id}_dem_{date}.tif")
        if not os.path.isfile(dem_path):
            raise FileNotFoundError(f"use_dem=True but DEM mosaic was not found: {dem_path}")

    with rasterio.open(dual_tif) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()
        try:
            src_mask1 = src.read_masks(1)
            src_mask2 = src.read_masks(2) if src.count >= 2 else src_mask1
            valid_src = (src_mask1 > 0) & (src_mask2 > 0)
            # Robustness: if masks are missing/invalid (all zeros or too sparse), treat everything as valid.
            if not np.any(valid_src) or (np.mean(valid_src) < 0.05):
                valid_src = np.ones((H, W), dtype=bool)
        except Exception:
            valid_src = np.ones((H, W), dtype=bool)

        if use_dem and dem_path is not None:
            dem_src = rasterio.open(dem_path)
            if dem_src.height != H or dem_src.width != W:
                raise ValueError(f"DEM and SAR mosaics have different shapes: dem={dem_src.height}x{dem_src.width}, sar={H}x{W}")

        pred = np.zeros((H, W), dtype=np.uint8)
        covered = np.zeros((H, W), dtype=np.uint8)

        ys = list(range(0, H, max(1, int(stride))))
        xs = list(range(0, W, max(1, int(stride))))
        coords = [(y0, min(y0 + tile, H), x0, min(x0 + tile, W)) for y0 in ys for x0 in xs]

        for i in range(0, len(coords), max(1, int(batch_size))):
            batch_coords = coords[i:i + max(1, int(batch_size))]
            patches: List[np.ndarray] = []
            shapes: List[Tuple[int, int]] = []
            for (y0, y1, x0, x1) in batch_coords:
                win = Window.from_slices((y0, y1), (x0, x1))
                patch_sar = src.read(indexes=(1, 2), window=win, out_dtype=np.float32)  # [2,h,w]

                if swap_bands:
                    patch_sar = patch_sar[[1, 0], ...]
                if mosaic_scale == "linear":
                    patch_sar = _db_scale(patch_sar)
                elif mosaic_scale == "db":
                    pass
                else:
                    raise ValueError("mosaic_scale must be 'linear' or 'db'")

                # If enabled, append VH/VV ratio in dB domain: (vh_db - vv_db), matching training channels=["vv","vh","vh/vv"].
                if use_ratio:
                    vv_db = patch_sar[0]
                    vh_db = patch_sar[1]
                    ratio_db = vh_db - vv_db
                    patch_sar = np.concatenate([patch_sar, ratio_db[None, ...]], axis=0)

                # Match Dataset._scale_image behavior: if mean/std are shorter than C, repeat the last value.
                if zscore and data_mean is not None and data_std is not None:
                    C = patch_sar.shape[0]
                    means = np.asarray(list(data_mean), dtype=np.float32)
                    stds = np.asarray(list(data_std), dtype=np.float32)
                    if means.shape[0] < C:
                        means = np.concatenate([means, np.repeat(means[-1:], C - means.shape[0])])
                        stds = np.concatenate([stds, np.repeat(stds[-1:], C - stds.shape[0])])
                    elif means.shape[0] > C:
                        means = means[:C]
                        stds = stds[:C]
                    for c in range(C):
                        std_c = stds[c] if abs(stds[c]) > 1e-6 else 1.0
                        patch_sar[c] = (patch_sar[c] - float(means[c])) / float(std_c)

                # DEM patch (optional)
                if use_dem and dem_src is not None:
                    dem_patch = dem_src.read(1, window=win, out_dtype=np.float32)
                    if dem_scale_mode.lower() == "zscore":
                        std = dem_std if abs(dem_std) > 1e-6 else 1.0
                        dem_patch = (dem_patch - float(dem_mean)) / float(std)
                    elif dem_scale_mode.lower() == "none":
                        pass
                    else:
                        raise ValueError("dem_scale_mode must be 'zscore' or 'none'")
                    dem_patch = np.nan_to_num(dem_patch, nan=0.0, posinf=0.0, neginf=0.0)
                    patch = np.concatenate([patch_sar, dem_patch[None, ...]], axis=0)
                else:
                    patch = patch_sar

                hh, ww = patch.shape[1], patch.shape[2]
                pad_h = max(0, tile - hh)
                pad_w = max(0, tile - ww)
                if pad_h > 0 or pad_w > 0:
                    patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
                patches.append(patch)
                shapes.append((hh, ww))

            batch_tensor = torch.from_numpy(np.stack(patches, axis=0))  # [B,C,tile,tile]
            if device == "cuda":
                outputs = model({
                    "image": batch_tensor.cuda(non_blocking=True),
                    "mask": torch.zeros(len(batch_coords), tile, tile, device="cuda", dtype=torch.long),
                })
            else:
                outputs = model({
                    "image": batch_tensor,
                    "mask": torch.zeros(len(batch_coords), tile, tile, dtype=torch.long),
                })
            logits = outputs["main_logits"] if isinstance(outputs, dict) else outputs
            if logits.shape[-2:] != (tile, tile):
                logits = torch.nn.functional.interpolate(logits, size=(tile, tile), mode="bilinear", align_corners=False)
            preds_full = torch.argmax(logits, dim=1).detach().cpu().to(torch.uint8).numpy()  # [B,tile,tile]

            for bi, (y0, y1, x0, x1) in enumerate(batch_coords):
                hh, ww = shapes[bi]
                pred_patch = preds_full[bi][:hh, :ww]
                pred[y0:y1, x0:x1] = pred_patch
                covered[y0:y1, x0:x1] = 1

    if dem_src is not None:
        try:
            dem_src.close()
        except Exception:
            pass

    # Mark uncovered/invalid regions as nodata=3 (gray) and ignore them during evaluation.
    pred[covered == 0] = 3
    pred[~valid_src] = 3

    # Write a single-band enhanced mask (with palette), suitable for GIS tools.
    tmp_mask = os.path.join(out_dir, f"._event_{event_id}_{date}_tmp.tif")
    profile.update({"count": 1, "dtype": "uint8", "nodata": 3})
    profile.pop("compress", None)
    profile.pop("tiled", None)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    with rasterio.open(tmp_mask, "w", **profile) as dst:
        dst.write(pred, 1)
    enhance_mask_for_qgis(tmp_mask, enh_out)
    try:
        os.remove(tmp_mask)
    except Exception:
        pass
    print(f"[INFO] Event {event_id}: enhanced mask written: {enh_out}")
    return enh_out


def compute_event_metrics_from_files(enh_pred_path: str, data_root: str) -> Optional[Dict[str, object]]:
    base = os.path.basename(enh_pred_path)
    m = re.match(r"event_(\d+)_flood_mask_(20\d{6}|unknown)_enhanced\.tif", base)
    if not m:
        return None
    event_id, date = m.group(1), m.group(2)
    gt_path = os.path.join(data_root, f"event_{event_id}_flood_mask_{date}.tif")
    if not os.path.isfile(gt_path):
        print(f"[WARN] Event {event_id}: GT mosaic not found; skip metrics: {gt_path}")
        return None

    with rasterio.open(enh_pred_path) as sp:
        pred = sp.read(1)
        pred_h, pred_w = sp.height, sp.width
        pred_transform = sp.transform
        pred_crs = sp.crs

    with rasterio.open(gt_path) as sg:
        dst = np.zeros((pred_h, pred_w), dtype=np.uint8)
        try:
            reproject(
                source=rasterio.band(sg, 1),
                destination=dst,
                src_transform=sg.transform,
                src_crs=sg.crs,
                dst_transform=pred_transform,
                dst_crs=pred_crs,
                resampling=Resampling.nearest,
            )
            gt = dst
        except Exception as e:
            print(f"[WARN] Event {event_id}: failed to reproject GT to prediction grid; fall back to cropping: {e}")
            raw_gt = sg.read(1)
            h = min(pred.shape[0], raw_gt.shape[0])
            w = min(pred.shape[1], raw_gt.shape[1])
            pred = pred[:h, :w].astype(np.uint8)
            gt = raw_gt[:h, :w].astype(np.uint8)
    # Optional: load event-level validity mask mosaic (e.g., merged from MK0_MNA).
    # If present, it is combined with GT ignore pixels (=3) to define the valid region.
    valid = None
    valid_mask_path = os.path.join(data_root, f"event_{event_id}_valid_mask_{date}.tif")
    if os.path.isfile(valid_mask_path):
        try:
            with rasterio.open(valid_mask_path) as sv:
                vdst = np.zeros((pred_h, pred_w), dtype=np.uint8)
                reproject(
                    source=rasterio.band(sv, 1),
                    destination=vdst,
                    src_transform=sv.transform,
                    src_crs=sv.crs,
                    dst_transform=pred_transform,
                    dst_crs=pred_crs,
                    resampling=Resampling.nearest,
                )
                valid = (vdst == 1)
        except Exception as e:
            print(f"[WARN] Event {event_id}: failed to read/reproject validity mask; ignore it: {e}")
            valid = None

    # Adaptive binarization and ignore handling:
    # - Ignore: prefer 3 or 255 (cloud/invalid)
    # - Water: prefer 1; if absent use 2; if neither exists but positives exist, fall back to (gt > 0)
    uniq = np.unique(gt)
    ignore_mask = np.zeros_like(gt, dtype=bool)
    if 3 in uniq:
        ignore_mask |= (gt == 3)
    if 255 in uniq:
        ignore_mask |= (gt == 255)
    # Water label decision
    if 1 in uniq and np.any(gt == 1):
        water_mask = (gt == 1)
    elif 2 in uniq and np.any(gt == 2):
        water_mask = (gt == 2)
    else:
        water_mask = (gt > 0)
    gt_bin = water_mask.astype(np.uint8)
    valid = ((~ignore_mask) & valid) if isinstance(valid, np.ndarray) else (~ignore_mask)

    # Diagnostics: positive ratio over valid pixels
    valid_count = float(max(1, valid.sum()))
    gt_pos_ratio = float(((gt_bin == 1) & valid).sum()) / valid_count
    pred_pos_ratio = float(((pred == 255) & valid).sum()) / valid_count
    print(f"[INFO] act={event_id} date={date} GT_pos={gt_pos_ratio:.4f} Pred_pos={pred_pos_ratio:.4f} (valid_mask={'on' if os.path.isfile(valid_mask_path) else 'off'})")

    tp = int(((pred == 255) & (gt_bin == 1) & valid).sum())
    fp = int(((pred == 255) & (gt_bin == 0) & valid).sum())
    tn = int(((pred == 0) & (gt_bin == 0) & valid).sum())
    fn = int(((pred == 0) & (gt_bin == 1) & valid).sum())
    total = max(1, tp + fp + tn + fn)

    acc = (tp + tn) / total
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))  # water recall
    specificity = tn / max(1, (tn + fp))
    f1 = (2 * precision * recall / max(1e-12, (precision + recall))) if (precision + recall) > 0 else 0.0

    # IoU
    iou_water = tp / max(1, (tp + fp + fn))
    iou_bg = tn / max(1, (tn + fp + fn))
    miou = 0.5 * (iou_water + iou_bg)

    # Aliases for backwards-compatible metric names.
    bg_recall = specificity
    bg_specificity = specificity
    bg_false_alarm_rate = fp / max(1, (fp + tn))
    flood_recall = recall
    flood_specificity = specificity
    flood_miss_rate = 1.0 - flood_recall

    return {
        "actid": int(event_id),
        "date": date,
        "test/acc": acc,
        "test/bg_false_alarm_rate": bg_false_alarm_rate,
        "test/bg_recall": bg_recall,
        "test/bg_specificity": bg_specificity,
        "test/f1": f1,
        "test/flood_miss_rate": flood_miss_rate,
        "test/flood_recall": flood_recall,
        "test/flood_specificity": flood_specificity,
        "test/iou": miou,
        "test/precision": precision,
        "test/recall": recall,
        "test/specificity": specificity,
        "test/water_iou": iou_water,
    }

