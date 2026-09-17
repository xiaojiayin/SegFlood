import os
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import rasterio as rio  # type: ignore
from rasterio.merge import merge as rio_merge  # type: ignore
from lightning.pytorch.callbacks import Callback  # type: ignore


class GFFloodNetPredictWriter(Callback):
    """
    基于 Trainer.predict 的落盘与评估：
    - 写出 predictions/（GeoTIFF 或 PNG）
    - 写出 overall_metrics.xlsx 与 detailed_samples.xlsx
    依赖于 pl_module.predict_step 返回 {"main_logits":..., "preds":...}
    """

    def __init__(self, output_dir: str, save_predictions: bool, save_format: str = "auto",
                 dataset_root: Optional[str] = None, mosaic_enabled: bool = True, mosaic_alpha: float = 0.5) -> None:
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
        self.dataset_root = dataset_root
        self.mosaic_enabled = bool(mosaic_enabled)
        self.mosaic_alpha = float(mosaic_alpha)

    def setup(self, trainer, pl_module, stage: str) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_predictions:
            os.makedirs(self.preds_dir, exist_ok=True)

    def on_predict_start(self, trainer, pl_module) -> None:
        # 记录数据集引用以供文件名与GeoTIFF profile
        dm = getattr(trainer, "datamodule", None)
        if dm is not None:
            # 优先使用 predict_use_full 时构建的完整数据集
            if hasattr(dm, "predict_dataset") and getattr(dm, "predict_dataset", None) is not None:
                self._ds = dm.predict_dataset
            # 否则退回到 test split（与评估口径一致）
            elif hasattr(dm, "test_dataset") and getattr(dm, "test_dataset", None) is not None:
                self._ds = dm.test_dataset
        self.rows.clear()
        self.total_tp = self.total_fp = self.total_tn = self.total_fn = 0
        self.sample_counter = 0

    def _get_sample_name_and_ref(self, global_idx: int) -> Tuple[str, Optional[str]]:
        ref_path = None
        name = f"sample_{global_idx:06d}"
        ds = self._ds
        # 优先处理 Subset 包装（random_split 产生）
        if ds is not None and hasattr(ds, "dataset") and hasattr(ds, "indices"):
            base = getattr(ds, "dataset", None)
            idxs = getattr(ds, "indices", None)
            if isinstance(base, object) and isinstance(idxs, list) and 0 <= global_idx < len(idxs):
                orig = idxs[global_idx]
                files = getattr(base, "files", None)
                if isinstance(files, list) and 0 <= orig < len(files):
                    ref_path = files[orig]
                    name = os.path.splitext(os.path.basename(ref_path))[0]
                    return name, ref_path
        # 直接数据集（无 Subset）
        if ds is not None and hasattr(ds, "files"):
            files = getattr(ds, "files", None)
            if isinstance(files, list) and 0 <= global_idx < len(files):
                ref_path = files[global_idx]
                name = os.path.splitext(os.path.basename(ref_path))[0]
                return name, ref_path
        # 未能解析样本名：按需求直接报错终止
        raise RuntimeError("GF-FloodNet: 无法解析样本名称与参考路径（禁止使用默认 sample_XXXXXX 命名）。")

    def _save_pred(self, pred_2d: torch.Tensor, name: str, ref_path: Optional[str]) -> None:
        # 二值掩码：陆地=0(黑)，洪水=255(红色显示)
        m = ((pred_2d.detach().cpu() > 0).to(torch.uint8) * 255)
        if (self.save_format in ("auto", "tif", "both")) and ref_path is not None:
            tif_path = os.path.join(self.preds_dir, f"{name}.tif")
            with rio.open(ref_path) as src:
                profile = src.profile.copy()
            profile.update({
                "count": 1,
                "dtype": "uint8",
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "photometric": "palette",
            })
            with rio.open(tif_path, "w", **profile) as dst:
                dst.write(m.numpy(), 1)
                # 统一调色板：0=黑(陆地), 255=红(水体), 3=灰(无效/忽略)
                try:
                    dst.write_colormap(1, {
                        0: (0, 0, 0, 255),
                        255: (255, 0, 0, 255),
                        3: (200, 200, 200, 255),
                    })
                except Exception:
                    pass
            # auto 默认仅写 TIF，不再额外写 PNG
            if self.save_format in ("tif", "auto"):
                return
        if (self.save_format in ("png", "both")) and (ref_path is None or self.save_format != "tif"):
            from PIL import Image  # lazy
            png_path = os.path.join(self.preds_dir, f"{name}.png")
            Image.fromarray((m * 255).numpy()).save(png_path)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        logits = outputs.get("main_logits") if isinstance(outputs, dict) else outputs
        assert isinstance(logits, torch.Tensor), "predict_step 必须返回包含 'main_logits' 的字典或张量"
        masks = batch.get("mask")
        if isinstance(masks, torch.Tensor) and logits.shape[2:] != masks.shape[1:]:
            logits = F.interpolate(logits, size=masks.shape[1:], mode="bilinear", align_corners=False)
        preds = outputs.get("preds") if isinstance(outputs, dict) else torch.argmax(logits, dim=1)
        assert isinstance(preds, torch.Tensor), "predict_step 必须返回 'preds' 或可从 logits 计算得到"
        bs = preds.shape[0]
        for i in range(bs):
            global_idx = self.sample_counter + i
            name, ref_path = self._get_sample_name_and_ref(global_idx)
            pred_i = preds[i]
            assert isinstance(masks, torch.Tensor), "batch 中缺少 'mask'，无法计算指标"
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
            # 同义指标，保持与历史口径相同
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
                self._save_pred(pred_i, name, ref_path)

        self.sample_counter += bs

    def on_predict_end(self, trainer, pl_module) -> None:
        denom_all = self.total_tp + self.total_fp + self.total_tn + self.total_fn
        denom_iou_w = self.total_tp + self.total_fp + self.total_fn
        denom_iou_bg = self.total_tn + self.total_fp + self.total_fn
        overall_acc = (self.total_tp + self.total_tn) / denom_all if denom_all > 0 else 0.0
        water_iou = self.total_tp / denom_iou_w if denom_iou_w > 0 else 0.0
        bg_iou = self.total_tn / denom_iou_bg if denom_iou_bg > 0 else 0.0
        present_w_all = (denom_iou_w > 0)
        present_bg_all = (denom_iou_bg > 0)
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
        # 同义指标
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
        # 与 tqdm 分隔，打印前先换行
        print()
        print(f"[INFO] Overall metrics saved: {overall_path}")
        print(f"[INFO] Per-sample metrics saved: {detailed_path}")

        # 可选：在推理结束时生成事件级大图与事件级指标（整图马赛克）
        if self.mosaic_enabled and self.dataset_root is not None and self.save_predictions:
            try:
                images_dir = os.path.join(self.dataset_root, "images")
                ann_dir = os.path.join(self.dataset_root, "annotations")
                preds_dir = self.preds_dir
                out_dir = os.path.join(self.output_dir, "mosaics")
                detailed_file = detailed_path if os.path.isfile(detailed_path) else None
                _generate_gffloodnet_mosaics(
                    images_dir=images_dir,
                    annotations_dir=ann_dir,
                    predictions_dir=preds_dir,
                    output_dir=out_dir,
                    detailed_file=detailed_file,
                    alpha=self.mosaic_alpha,
                )
                print(f"[INFO] Event-level mosaics & metrics saved under: {out_dir}")
            except Exception as e:
                print(f"[WARN] Failed to generate GF-FloodNet mosaics: {e}")


def _list_event_keys(images_dir: str) -> List[str]:
    paths = sorted([p for p in os.listdir(images_dir) if p.endswith(".tif")])
    keys: set[str] = set()
    for name in paths:
        stem = os.path.splitext(name)[0]
        parts = stem.split("_")
        if len(parts) >= 2:
            keys.add(parts[0] + "_" + parts[1])
    return sorted(keys)


def _filter_by_event(paths: List[str], event_key: str) -> List[str]:
    pref = event_key + "_"
    return [p for p in paths if os.path.basename(p).startswith(pref)]


def _mosaic(paths: List[str]) -> Tuple[np.ndarray, dict]:
    with rio.Env(GDAL_NUM_THREADS="ALL_CPUS", NUM_THREADS="ALL_CPUS"):
        datasets = [rio.open(p) for p in paths]
        arr, transform = rio_merge(datasets)
        profile = datasets[0].profile
        for ds in datasets:
            ds.close()
        profile.update({
            "height": arr.shape[1],
            "width": arr.shape[2],
            "transform": transform,
            "count": arr.shape[0],
        })
        return arr, profile


def _write_tif(path: str, arr: np.ndarray, profile: dict, dtype: str | None = None,
               count: int | None = None, nodata: int | None = None) -> None:
    prof = profile.copy()
    if dtype is not None:
        prof["dtype"] = dtype
    if count is not None:
        prof["count"] = count
    if nodata is not None:
        prof["nodata"] = nodata
    prof.update({
        "compress": "DEFLATE",
        "predictor": 2,
        "zlevel": 9,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff": "yes",
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rio.open(path, "w", **prof) as dst:
        dst.write(arr)


def _process_event(event_key: str, images_dir: str, annotations_dir: str, predictions_dir: str,
                   output_dir: str, alpha: float) -> None:
    # 以「有预测的 tiles」为基准，再去匹配对应的 GT tiles，避免包含没有预测的区域
    pr_paths = _filter_by_event(sorted(
        [os.path.join(predictions_dir, p) for p in os.listdir(predictions_dir) if p.endswith(".tif")]), event_key)
    if not pr_paths:
        return

    gt_paths: List[str] = []
    for p in pr_paths:
        bn = os.path.basename(p)
        cand = os.path.join(annotations_dir, bn)
        if os.path.isfile(cand):
            gt_paths.append(cand)
    if not gt_paths:
        return

    gt_arr, gt_prof = _mosaic(gt_paths)
    pr_arr, pr_prof = _mosaic(pr_paths)

    # 仅输出与推理结果直接相关的掩码类结果：GT 大图与 Pred 大图
    # GT uint8（保持原语义：0=invalid, 1=water, 255=background，方便可视化脚本直接使用）
    gt_raw = gt_arr[0].astype(np.uint8)
    valid = (gt_raw != 0)
    gt_bin = ((gt_raw == 1) & valid).astype(np.uint8)[None, ...]
    _write_tif(os.path.join(output_dir, event_key, "gt.tif"), gt_raw[None, ...], gt_prof, dtype="uint8", count=1, nodata=0)

    # Pred uint8
    pred_bin = (pr_arr[0] > 0).astype(np.uint8)[None, ...]
    _write_tif(os.path.join(output_dir, event_key, "pred.tif"), pred_bin, pr_prof, dtype="uint8", count=1, nodata=0)

    # 7) event-level mosaic metrics（同样裁剪到公共区域，避免尺寸不一致）
    gt_ev = gt_bin[0].astype(np.uint8)
    pdm_ev = pred_bin[0].astype(np.uint8)
    h_gt, w_gt = gt_ev.shape
    h_pr, w_pr = pdm_ev.shape
    h_v, w_v = valid.shape
    hh = min(h_gt, h_pr, h_v)
    ww = min(w_gt, w_pr, w_v)
    gt_ev_c = gt_ev[:hh, :ww]
    pdm_ev_c = pdm_ev[:hh, :ww]
    valid_c = valid[:hh, :ww]

    tp = int(((gt_ev_c == 1) & (pdm_ev_c == 1) & valid_c).sum())
    fp = int(((gt_ev_c == 0) & (pdm_ev_c == 1) & valid_c).sum())
    fn = int(((gt_ev_c == 1) & (pdm_ev_c == 0) & valid_c).sum())
    union = tp + fp + fn
    iou = float(tp / union) if union > 0 else 0.0
    denom = 2 * tp + fp + fn
    f1 = float((2 * tp) / denom) if denom > 0 else 0.0
    out_dir_evt = os.path.join(output_dir, event_key)
    pd.DataFrame([{
        "event": event_key,
        "tp": tp, "fp": fp, "fn": fn,
        "gt_pos": int(((gt_ev_c == 1) & valid_c).sum()),
        "pred_pos": int(((pdm_ev_c == 1) & valid_c).sum()),
        "iou_mosaic": iou,
        "f1_mosaic": f1,
    }]).to_csv(os.path.join(out_dir_evt, "metrics.csv"), index=False)


def _generate_gffloodnet_mosaics(
    images_dir: str,
    annotations_dir: str,
    predictions_dir: str,
    output_dir: str,
    detailed_file: Optional[str],
    alpha: float,
) -> None:
    if not (os.path.isdir(images_dir) and os.path.isdir(annotations_dir) and os.path.isdir(predictions_dir)):
        return

    keys = _list_event_keys(images_dir)
    for k in keys:
        _process_event(k, images_dir, annotations_dir, predictions_dir, output_dir, alpha)

    # event-level metrics based on mosaic (metrics.csv per event)
    rows2: List[Dict[str, Any]] = []
    for k in keys:
        p = os.path.join(output_dir, k, "metrics.csv")
        if os.path.isfile(p):
            df1 = pd.read_csv(p)
            if not df1.empty:
                rows2.append(df1.iloc[0].to_dict())
    if rows2:
        out_df2 = pd.DataFrame(rows2)
        os.makedirs(output_dir, exist_ok=True)
        out_df2.to_excel(os.path.join(output_dir, "event_metrics_mosaic.xlsx"), index=False)

    # 可选：基于 detailed_samples.xlsx 聚合为事件级 tiles 指标
    if detailed_file and os.path.isfile(detailed_file):
        df = pd.read_excel(detailed_file)
        if "sample_name" in df.columns:
            metric_cols = [c for c in df.columns if c not in {"sample_name", "global_idx", "dataset"}]
            rows_tiles: List[Dict[str, Any]] = []
            for key in keys:
                pref = key + "_"
                sub = df[df["sample_name"].astype(str).str.startswith(pref)]
                if len(sub) == 0:
                    continue
                agg = {"event": key, "num_tiles": len(sub)}
                for c in metric_cols:
                    if pd.api.types.is_numeric_dtype(sub[c]):
                        agg[c] = float(sub[c].mean())
                rows_tiles.append(agg)
            if rows_tiles:
                out_df_tiles = pd.DataFrame(rows_tiles)
                os.makedirs(output_dir, exist_ok=True)
                out_df_tiles.to_excel(os.path.join(output_dir, "event_metrics_tiles.xlsx"), index=False)



