import os
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
import pandas as pd  # type: ignore
import rasterio as rio  # type: ignore
import numpy as np  # type: ignore
from lightning.pytorch.callbacks import Callback  # type: ignore
from src.data.datasets.worldfloodsv2 import CHANNELS_CONFIGURATIONS  # type: ignore


class WorldFloodsPredictWriter(Callback):
    """
    Driven by `Trainer.predict`, but performs full-scene sliding-window inference in `on_predict_end`
    (reusing DataModule configuration and preprocessing).

    Outputs:
    - predictions/<stem>.tif (GeoTIFF; values in {0,255})
    - overall_metrics.xlsx, detailed_samples.xlsx

    Note: batch outputs during the predict loop are not used. We run inference at the end to
    reduce peak memory usage and avoid OOM.
    """
    def __init__(self, output_dir: str, save_predictions: bool, amp: bool = False) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.save_predictions = bool(save_predictions)
        self.amp = bool(amp)
        self.preds_dir = os.path.join(self.output_dir, "predictions")
        self.rows: List[Dict[str, Any]] = []
        self.total_tp = self.total_fp = self.total_tn = self.total_fn = 0

    def setup(self, trainer, pl_module, stage: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_predictions:
            os.makedirs(self.preds_dir, exist_ok=True)

    def on_predict_end(self, trainer, pl_module) -> None:
        # Get test dataset from DataModule and run full-scene sliding-window inference.
        dm = getattr(trainer, "datamodule", None)
        if dm is None or not hasattr(dm, "test_dataset"):
            raise RuntimeError("WorldFloods: missing test_dataset")
        test_ds = dm.test_dataset
        device = pl_module.device
        for idx in range(len(test_ds)):
            # Read GT and resolve image path
            item = test_ds[idx]
            mask = item["mask"].to(device)
            # Resolve full-scene image path from dataset (if available)
            img_path = None
            if hasattr(test_ds, "samples"):
                sam = test_ds.samples[idx]
                img_path = sam.get("img")
            if img_path is None:
                # Unsupported dataset type (e.g., tiles-only); fall back to a synthetic name.
                stem = f"sample_{idx:06d}"
            else:
                stem = os.path.splitext(os.path.basename(img_path))[0]
            # Sliding-window inference
            pred = self._predict_scene(pl_module, dm, img_path, device) if img_path else torch.argmax(pl_module(item.unsqueeze(0) if torch.is_tensor(item) else item)["main_logits"], dim=1)[0]
            # Metrics (ignore -1)
            g_flat = mask.flatten()
            p_flat = pred.to(device).flatten()
            valid = (g_flat != -1)
            if valid.any():
                pv = p_flat[valid]
                gv = g_flat[valid]
                tp = int(((pv == 1) & (gv == 1)).sum().item())
                fp = int(((pv == 1) & (gv == 0)).sum().item())
                tn = int(((pv == 0) & (gv == 0)).sum().item())
                fn = int(((pv == 0) & (gv == 1)).sum().item())
            else:
                tp = fp = tn = fn = 0
            self.total_tp += tp; self.total_fp += fp; self.total_tn += tn; self.total_fn += fn
            denom_iou_w = tp + fp + fn
            denom_iou_bg = tn + fp + fn
            denom_all = tp + fp + tn + fn
            water_iou = (tp / denom_iou_w) if denom_iou_w > 0 else 0.0
            bg_iou = (tn / denom_iou_bg) if denom_iou_bg > 0 else 0.0
            present_w = denom_iou_w > 0
            present_bg = denom_iou_bg > 0
            if present_w and present_bg:
                miou = 0.5 * (water_iou + bg_iou)
            elif present_w:
                miou = water_iou
            elif present_bg:
                miou = bg_iou
            else:
                miou = 0.0
            # Consistent with training logs: macro average for precision/recall/f1
            # Positive class (flood/water)
            prec_pos = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec_pos = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0
            spec_pos = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # equivalent to specificity of the positive class
            # Negative class (background)
            tp_neg = tn
            fp_neg = fn
            fn_neg = fp
            prec_neg = (tp_neg / (tp_neg + fp_neg)) if (tp_neg + fp_neg) > 0 else 0.0  # = tn/(tn+fn)
            rec_neg = (tp_neg / (tp_neg + fn_neg)) if (tp_neg + fn_neg) > 0 else 0.0  # = tn/(tn+fp) (also specificity of the positive class)
            f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0
            spec_neg = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            precision_macro = 0.5 * (prec_pos + prec_neg)
            recall_macro = 0.5 * (rec_pos + rec_neg)
            specificity_macro = 0.5 * (spec_pos + spec_neg)
            f1_macro = 0.5 * (f1_pos + f1_neg)
            acc_v = ((tp + tn) / denom_all) if denom_all > 0 else 0.0
            bg_false_alarm_rate = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            flood_miss_rate = 1.0 - rec_pos
            self.rows.append({
                "sample_name": stem,
                "iou": miou,
                "water_iou": water_iou,
                "bg_iou": bg_iou,
                "f1": f1_macro,
                "accuracy": acc_v,
                "precision": precision_macro,
                "recall": recall_macro,
                "specificity": specificity_macro,
                "bg_false_alarm_rate": bg_false_alarm_rate,
                "flood_miss_rate": flood_miss_rate,
            })
            # Write GeoTIFF
            if self.save_predictions and img_path is not None:
                with rio.open(img_path) as src:
                    profile = src.profile.copy()
                profile.update({"count":1,"dtype":"uint8","compress":"lzw","photometric":"palette","tiled":True,"blockxsize":256,"blockysize":256})
                tif_path = os.path.join(self.preds_dir, f"{stem}.tif")
                with rio.open(tif_path, "w", **profile) as dst:
                    arr = (pred.detach().cpu() > 0).to(torch.uint8).numpy() * 255
                    # Visualization: cloud pixels (from GT=3) -> 3; image nodata (band mask=0) -> 4
                    try:
                        msk_path = None
                        if hasattr(test_ds, "samples"):
                            msk_path = test_ds.samples[idx].get("msk")  # type: ignore[index]
                        cloud_mask = None
                        if msk_path is not None and os.path.isfile(msk_path):
                            with rio.open(msk_path) as ms:
                                if ms.count >= 2:
                                    gt = ms.read(2)
                                else:
                                    gt = ms.read(1)
                            cloud_mask = (gt == 3)
                        with rio.open(img_path) as src_img:  # type: ignore[arg-type]
                            band_mask = src_img.read_masks(1)
                            nodata_mask = (band_mask == 0)
                        if cloud_mask is not None and cloud_mask.shape == arr.shape:
                            arr = np.where(cloud_mask, 3, arr)
                        if nodata_mask.shape == arr.shape:
                            arr = np.where(nodata_mask, 4, arr)
                        arr = arr.astype(np.uint8)
                    except Exception:
                        arr = arr.astype(np.uint8)
                    dst.write(arr, 1)
                    # Colormap: 0=black (land), 255=red (flood), 3=light gray (cloud), 4=dark gray (image nodata)
                    try:
                        dst.write_colormap(1, {
                            0: (0, 0, 0, 255),
                            3: (200, 200, 200, 255),   # cloud
                            4: (128, 128, 128, 255),   # image nodata / missing data
                            255: (255, 0, 0, 255),
                        })
                    except Exception:
                        pass
        # Overall summary
        denom_all = self.total_tp + self.total_fp + self.total_tn + self.total_fn
        denom_iou_w = self.total_tp + self.total_fp + self.total_fn
        denom_iou_bg = self.total_tn + self.total_fp + self.total_fn
        overall_acc = (self.total_tp + self.total_tn) / denom_all if denom_all > 0 else 0.0
        water_iou = self.total_tp / denom_iou_w if denom_iou_w > 0 else 0.0
        bg_iou = self.total_tn / denom_iou_bg if denom_iou_bg > 0 else 0.0
        present_w_all = denom_iou_w > 0
        present_bg_all = denom_iou_bg > 0
        if present_w_all and present_bg_all:
            miou = 0.5 * (water_iou + bg_iou)
        elif present_w_all:
            miou = water_iou
        elif present_bg_all:
            miou = bg_iou
        else:
            miou = 0.0
        # Macro average (consistent with training)
        # Positive class (flood/water)
        prec_pos = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        rec_pos = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0
        spec_pos = self.total_tn / (self.total_tn + self.total_fp) if (self.total_tn + self.total_fp) > 0 else 0.0
        # Negative class (background)
        tp_neg = self.total_tn
        fp_neg = self.total_fn
        fn_neg = self.total_fp
        prec_neg = tp_neg / (tp_neg + fp_neg) if (tp_neg + fp_neg) > 0 else 0.0
        rec_neg = tp_neg / (tp_neg + fn_neg) if (tp_neg + fn_neg) > 0 else 0.0
        f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0
        spec_neg = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        precision_v = 0.5 * (prec_pos + prec_neg)
        recall_v = 0.5 * (rec_pos + rec_neg)
        specificity_v = 0.5 * (spec_pos + spec_neg)
        f1_v = 0.5 * (f1_pos + f1_neg)
        bg_false_alarm_rate = self.total_fp / (self.total_fp + self.total_tn) if (self.total_fp + self.total_tn) > 0 else 0.0
        flood_miss_rate = 1.0 - recall_v
        overall = {
            "iou": miou,
            "f1": f1_v,
            "accuracy": overall_acc,
            "precision": precision_v,
            "recall": recall_v,
            "water_iou": water_iou,
            "bg_iou": bg_iou,
            "specificity": specificity_v,
            "bg_false_alarm_rate": bg_false_alarm_rate,
            "flood_miss_rate": flood_miss_rate,
        }
        overall_path = os.path.join(self.output_dir, "overall_metrics.xlsx")
        detailed_path = os.path.join(self.output_dir, "detailed_samples.xlsx")
        pd.DataFrame([overall]).to_excel(overall_path, index=False)
        pd.DataFrame(self.rows).to_excel(detailed_path, index=False)
        print()
        print(f"[INFO] Overall metrics saved: {overall_path}")
        print(f"[INFO] Per-sample metrics saved: {detailed_path}")
        print(f"[INFO] Predictions dir: {self.preds_dir}")

    @torch.no_grad()
    def _predict_scene(self, pl_module, dm, img_path: str, device) -> torch.Tensor:
        import numpy as np
        from rasterio.windows import Window
        with rio.open(img_path) as src:
            H, W = src.height, src.width
            # Use the same channel configuration as the DataModule
            channels_cfg = getattr(dm, "kwargs", {}).get("channels", "bgri")
            idxs_0b = None
            if isinstance(channels_cfg, str) and channels_cfg in CHANNELS_CONFIGURATIONS:
                idxs_0b = CHANNELS_CONFIGURATIONS[channels_cfg]
            elif isinstance(channels_cfg, (list, tuple)):
                try:
                    idxs_0b = [int(i) for i in channels_cfg]
                except Exception:
                    idxs_0b = None
            # Convert to rasterio 1-based indexing; if unknown, conservatively take the first 4 bands.
            if idxs_0b is not None and len(idxs_0b) > 0:
                idxs = [i + 1 for i in idxs_0b if (i + 1) <= src.count]
                if len(idxs) == 0:
                    idxs = list(range(1, min(src.count + 1, 5)))
            else:
                idxs = list(range(1, min(src.count + 1, 5)))

            out_acc = torch.zeros((H, W), device=device, dtype=torch.long)
            tile = int(getattr(dm, "window_size", (256, 256))[0]) if hasattr(dm, "window_size") else 1024
            pad = 32
            multiple_of = int(getattr(dm, "multiple_of_pad", 32))
            for i in range(0, H, tile):
                for j in range(0, W, tile):
                    i0 = max(i - pad, 0)
                    j0 = max(j - pad, 0)
                    i1 = min(i + tile + pad, H)
                    j1 = min(j + tile + pad, W)
                    win = Window.from_slices((i0, i1), (j0, j1))
                    arr = src.read(indexes=idxs, window=win, boundless=True, fill_value=0).astype(np.float32)
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    tile_tensor = torch.from_numpy(arr).unsqueeze(0).to(device)
                    dummy_mask = torch.zeros((1, tile_tensor.shape[2], tile_tensor.shape[3]), dtype=torch.long, device=device)
                    batch = {"image": tile_tensor, "mask": dummy_mask}
                    batch = dm.process_batch(batch, stage="test") if hasattr(dm, "process_batch") else batch
                    if self.amp and torch.cuda.is_available():
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            outputs = pl_module.predict_step(batch, 0) if hasattr(pl_module, "predict_step") else pl_module(batch)
                    else:
                        outputs = pl_module.predict_step(batch, 0) if hasattr(pl_module, "predict_step") else pl_module(batch)
                    logits = outputs["main_logits"] if isinstance(outputs, dict) else outputs
                    pred = torch.argmax(logits, dim=1)  # [1,hp,wp]
                    h_valid = min(tile, H - i)
                    w_valid = min(tile, W - j)
                    pad_h = i - i0
                    pad_w = j - j0
                    pred_crop = pred[0, pad_h:pad_h + h_valid, pad_w:pad_w + w_valid]
                    out_acc[i:i + h_valid, j:j + w_valid] = pred_crop
        return out_acc.detach().cpu()


