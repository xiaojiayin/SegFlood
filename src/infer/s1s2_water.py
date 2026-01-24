import os
import re
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import pandas as pd  # type: ignore
import rasterio as rio  # type: ignore
from lightning.pytorch.callbacks import Callback  # type: ignore


class S1S2WaterPredictWriter(Callback):
    """
    Prediction writer for S1S2-Water.

    - Save per-tile predictions to predictions/ (GeoTIFF with georeference; values in {0,255})
    - Record metrics (per-sample rows + overall summary)
    - At the end, automatically mosaic tiles into mosaics/ based on the `sampleN_` prefix
    """
    def __init__(self, output_dir: str, save_predictions: bool, modal_type: str = "dual") -> None:
        super().__init__()
        self.output_dir = output_dir
        self.save_predictions = bool(save_predictions)
        self.modal_type = modal_type
        self.preds_dir = os.path.join(self.output_dir, "predictions")
        self.mosaics_dir = os.path.join(self.output_dir, "mosaics")
        self.rows: List[Dict[str, Any]] = []
        self.total_tp = 0
        self.total_fp = 0
        self.total_tn = 0
        self.total_fn = 0
        self.sample_counter = 0
        self._ds = None

    def setup(self, trainer, pl_module, stage: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_predictions:
            os.makedirs(self.preds_dir, exist_ok=True)
        os.makedirs(self.mosaics_dir, exist_ok=True)

    def on_predict_start(self, trainer, pl_module) -> None:
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "test_dataset"):
            self._ds = dm.test_dataset
        self.rows.clear()
        self.total_tp = self.total_fp = self.total_tn = self.total_fn = 0
        self.sample_counter = 0

    def _get_ref(self, global_idx: int) -> Tuple[str, str]:
        if self._ds is None:
            raise RuntimeError("S1S2-Water: missing test_dataset")
        samples = getattr(self._ds, "samples", None)
        if not isinstance(samples, list) or not (0 <= global_idx < len(samples)):
            raise RuntimeError("S1S2-Water: failed to locate sample path")
        ref_path = samples[global_idx]["img"]
        stem = os.path.splitext(os.path.basename(ref_path))[0]
        return stem, ref_path

    def _save_tile_tif(self, pred_2d: torch.Tensor, stem: str, ref_path: str, nodata_mask: Optional[torch.Tensor] = None) -> None:
        m = (pred_2d.detach().cpu() > 0).to(torch.uint8) * 255
        if isinstance(nodata_mask, torch.Tensor):
            nd = (nodata_mask.detach().cpu() > 0)
            if nd.shape == m.shape:
                m[nd] = 3
        with rio.open(ref_path) as src:
            profile = src.profile.copy()
        profile.update({"count":1,"dtype":"uint8","compress":"lzw","tiled":True,"blockxsize":256,"blockysize":256,"photometric":"palette"})
        save_path = os.path.join(self.preds_dir, f"{stem}.tif")
        with rio.open(save_path, "w", **profile) as dst:
            dst.write(m.numpy(), 1)
            # Colormap: 0=black (land), 255=red (water), 3=gray (nodata)
            colormap = {
                0: (0, 0, 0, 255),
                3: (200, 200, 200, 255),
                255: (255, 0, 0, 255),
            }
            try:
                dst.write_colormap(1, colormap)
            except Exception:
                pass

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        logits = outputs.get("main_logits") if isinstance(outputs, dict) else outputs
        assert isinstance(logits, torch.Tensor)
        masks = batch.get("mask")
        if isinstance(masks, torch.Tensor) and logits.shape[2:] != masks.shape[1:]:
            logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
        preds = outputs.get("preds") if isinstance(outputs, dict) else torch.argmax(logits, dim=1)
        assert isinstance(preds, torch.Tensor)
        bs = preds.shape[0]
        for i in range(bs):
            global_idx = self.sample_counter + i
            stem, ref_path = self._get_ref(global_idx)
            pred_i = preds[i]
            assert isinstance(masks, torch.Tensor)
            mask_i = masks[i]
            p = pred_i.detach().flatten()
            g = mask_i.detach().flatten()
            valid = (g != -1)
            if valid.any():
                pv = p[valid]
                gv = g[valid]
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
            # Macro average (consistent with training logs)
            # Positive class (water)
            prec_pos = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec_pos = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0
            spec_pos = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            # Negative class (background)
            tp_neg = tn
            fp_neg = fn
            fn_neg = fp
            prec_neg = (tp_neg / (tp_neg + fp_neg)) if (tp_neg + fp_neg) > 0 else 0.0
            rec_neg = (tp_neg / (tp_neg + fn_neg)) if (tp_neg + fn_neg) > 0 else 0.0
            f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0
            spec_neg = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            precision_v = 0.5 * (prec_pos + prec_neg)
            recall_v = 0.5 * (rec_pos + rec_neg)
            specificity_v = 0.5 * (spec_pos + spec_neg)
            f1_v = 0.5 * (f1_pos + f1_neg)
            acc_v = ((tp + tn) / denom_all) if denom_all > 0 else 0.0
            bg_false_alarm_rate = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            flood_miss_rate = 1.0 - rec_pos
            self.rows.append({
                "sample_name": stem,
                "iou": miou,
                "water_iou": water_iou,
                "bg_iou": bg_iou,
                "f1": f1_v,
                "accuracy": acc_v,
                "precision": precision_v,
                "recall": recall_v,
                "specificity": specificity_v,
                "bg_false_alarm_rate": bg_false_alarm_rate,
                "flood_miss_rate": flood_miss_rate,
            })
            if self.save_predictions:
                self._save_tile_tif(pred_i, stem, ref_path, nodata_mask=(mask_i == -1))
        self.sample_counter += bs

    def _group_tiles_by_scene(self, tile_paths: List[str]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        # Parse numeric scene id from tile filenames; fall back to full prefix if unmatched.
        pat = re.compile(r"^sample(?P<scene_id>\d+)_tile_r\d+_c\d+\.tif$", re.IGNORECASE)
        for p in tile_paths:
            stem = os.path.basename(p)
            m = pat.match(stem)
            if m:
                scene = m.group("scene_id")
                groups.setdefault(scene, []).append(p)
        return groups

    def _mosaic_scene(self, paths: List[str]) -> Tuple[torch.Tensor, dict]:
        import numpy as np
        from rasterio.merge import merge as rio_merge
        datasets = [rio.open(p) for p in paths]
        try:
            mosaic, out_transform = rio_merge(datasets, method="first")
            profile = datasets[0].profile.copy()
        finally:
            for ds in datasets:
                try:
                    ds.close()
                except Exception:
                    pass
        profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "count": 1,
            "dtype": "uint8",
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "photometric": "palette",
            "nodata": 3,
        })
        mosaic = (mosaic > 0).astype(np.uint8) * 255
        return torch.from_numpy(mosaic), profile

    def on_predict_end(self, trainer, pl_module) -> None:
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
        # Macro average
        prec_pos = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        rec_pos = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0
        spec_pos = self.total_tn / (self.total_tn + self.total_fp) if (self.total_tn + self.total_fp) > 0 else 0.0
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
        flood_miss_rate = 1.0 - rec_pos
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
        # Build mosaics
        try:
            tile_files = [os.path.join(self.preds_dir, fn) for fn in os.listdir(self.preds_dir) if fn.lower().endswith(".tif")]
            if len(tile_files) > 0:
                groups = self._group_tiles_by_scene(tile_files)
                for scene, files in sorted(groups.items(), key=lambda kv: kv[0]):
                    out_path = os.path.join(self.mosaics_dir, f"{scene}_pred.tif")
                    mosaic, profile = self._mosaic_scene(sorted(files))
                    # Write with colormap: 0=black, 255=red, 3=gray
                    profile.update({"photometric": "palette"})
                    with rio.open(out_path, "w", **profile) as dst:
                        dst.write(mosaic[0].numpy(), 1)
                        try:
                            dst.write_colormap(1, {
                                0: (0, 0, 0, 255),
                                3: (200, 200, 200, 255),
                                255: (255, 0, 0, 255),
                            })
                        except Exception:
                            pass
                    print(f"[INFO] Mosaic saved: {out_path}")
        except Exception as e:
            print(f"[WARN] Mosaic failed: {e}")


