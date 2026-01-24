import os
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
import pandas as pd  # type: ignore
from lightning.pytorch.callbacks import Callback  # type: ignore


class CAUFloodPredictWriter(Callback):
    """
    Prediction writer + evaluator driven by `Trainer.predict`.

    Outputs:
    - predictions/ (PNG)
    - overall_metrics.xlsx, detailed_samples.xlsx

    Contract: `pl_module.predict_step` should return either:
    - a dict containing {"main_logits": ..., "preds": ...}, or
    - a logits tensor directly (then preds are derived via argmax).
    """

    def __init__(self, output_dir: str, save_predictions: bool, save_format: str = "auto") -> None:
        super().__init__()
        self.output_dir = output_dir
        self.save_predictions = bool(save_predictions)
        self.save_format = str(save_format)
        self.preds_dir = os.path.join(self.output_dir, "predictions")
        self.rows: List[Dict[str, Any]] = []
        self.total_tp = 0
        self.total_fp = 0
        self.total_tn = 0
        self.total_fn = 0
        self.sample_counter = 0
        self._ds = None  # dataset reference

    def setup(self, trainer, pl_module, stage: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_predictions:
            os.makedirs(self.preds_dir, exist_ok=True)

    def on_predict_start(self, trainer, pl_module) -> None:
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "test_dataset"):
            self._ds = dm.test_dataset
        self.rows.clear()
        self.total_tp = self.total_fp = self.total_tn = self.total_fn = 0
        self.sample_counter = 0

    def _get_sample_name(self, global_idx: int) -> str:
        name = f"sample_{global_idx:06d}"
        ds = self._ds
        if ds is not None and hasattr(ds, "sample_ids"):
            ids = getattr(ds, "sample_ids", None)
            if isinstance(ids, list) and 0 <= global_idx < len(ids):
                name = str(ids[global_idx])
        return name

    def _save_pred(self, pred_2d: torch.Tensor, name: str) -> None:
        from PIL import Image  # lazy
        import numpy as np
        # RGB visualization: 0=black (land), 255=red (water)
        m = (pred_2d.detach().cpu() > 0).to(torch.uint8) * 255
        arr = m.numpy()
        rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = arr  # R channel = mask
        # G/B are kept 0 -> red mask overlay
        png_path = os.path.join(self.preds_dir, f"{name}.png")
        Image.fromarray(rgb, mode="RGB").save(png_path)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        logits = outputs.get("main_logits") if isinstance(outputs, dict) else outputs
        assert isinstance(logits, torch.Tensor), "`predict_step` must return a dict containing 'main_logits' or a logits tensor"
        masks = batch.get("mask")
        if isinstance(masks, torch.Tensor) and logits.shape[2:] != masks.shape[1:]:
            logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
        preds = outputs.get("preds") if isinstance(outputs, dict) else torch.argmax(logits, dim=1)
        assert isinstance(preds, torch.Tensor), "`predict_step` must return 'preds' or allow deriving preds from logits"
        bs = preds.shape[0]
        for i in range(bs):
            global_idx = self.sample_counter + i
            name = self._get_sample_name(global_idx)
            pred_i = preds[i]
            assert isinstance(masks, torch.Tensor), "Missing 'mask' in batch; cannot compute metrics"
            mask_i = masks[i]
            p = pred_i.detach().flatten()
            g = mask_i.detach().flatten()
            tp = int(((p == 1) & (g == 1)).sum().item())
            fp = int(((p == 1) & (g == 0)).sum().item())
            tn = int(((p == 0) & (g == 0)).sum().item())
            fn = int(((p == 0) & (g == 1)).sum().item())
            self.total_tp += tp; self.total_fp += fp; self.total_tn += tn; self.total_fn += fn

            denom_iou_w = tp + fp + fn
            denom_iou_bg = tn + fp + fn
            denom_spec = tn + fp
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
            precision_v = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall_v = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity_v = (tn / denom_spec) if denom_spec > 0 else 0.0
            f1_v = (2 * precision_v * recall_v / (precision_v + recall_v)) if (precision_v + recall_v) > 0 else 0.0
            acc_v = ((tp + tn) / denom_all) if denom_all > 0 else 0.0
            bg_false_alarm_rate = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            flood_miss_rate = 1.0 - recall_v
            bg_recall = specificity_v
            flood_recall = recall_v
            bg_specificity = specificity_v
            flood_specificity = specificity_v

            self.rows.append({
                "sample_name": name,
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
                "bg_recall": bg_recall,
                "bg_specificity": bg_specificity,
                "flood_recall": flood_recall,
                "flood_specificity": flood_specificity,
            })

            if self.save_predictions:
                self._save_pred(pred_i, name)

        self.sample_counter += bs

    def on_predict_end(self, trainer, pl_module) -> None:
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
        precision_v = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        recall_v = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0
        specificity_v = self.total_tn / (self.total_tn + self.total_fp) if (self.total_tn + self.total_fp) > 0 else 0.0
        f1_v = (2 * precision_v * recall_v / (precision_v + recall_v)) if (precision_v + recall_v) > 0 else 0.0
        bg_false_alarm_rate = self.total_fp / (self.total_fp + self.total_tn) if (self.total_fp + self.total_tn) > 0 else 0.0
        flood_miss_rate = 1.0 - recall_v
        bg_recall = specificity_v
        flood_recall = recall_v
        bg_specificity = specificity_v
        flood_specificity = specificity_v
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
            "bg_recall": bg_recall,
            "bg_specificity": bg_specificity,
            "flood_recall": flood_recall,
            "flood_specificity": flood_specificity,
        }
        overall_path = os.path.join(self.output_dir, "overall_metrics.xlsx")
        detailed_path = os.path.join(self.output_dir, "detailed_samples.xlsx")
        pd.DataFrame([overall]).to_excel(overall_path, index=False)
        pd.DataFrame(self.rows).to_excel(detailed_path, index=False)
        # Print a blank line before paths (keeps logs readable next to tqdm output).
        print()
        print(f"[INFO] Overall metrics saved: {overall_path}")
        print(f"[INFO] Per-sample metrics saved: {detailed_path}")


