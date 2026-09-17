"""Lightning DataModule for WorldFloodsv2.

职责分离：
- Dataset 仅做 I/O 与标签映射；
- 本 DataModule 负责批处理级别的归一化与几何增强；

官方对齐（默认）：
- 关闭百分位缩放，直接使用 WorldFloods 固定 per-band 统计 (SENTINEL2_NORMALIZATION) 做 Z-Score；
- 通道需为官方定义的配置之一（如 "rgb"、"bgriswirs" 等）。

"""

from __future__ import annotations

from typing import Any, Callable, Optional, List, Dict, Tuple

import torch  # type: ignore[import]
import kornia.augmentation as K  # type: ignore[import]
from torch.utils.data import random_split  # type: ignore[import]
from torchgeo.datamodules import NonGeoDataModule  # type: ignore[import]
import numpy as np  # type: ignore[import]
import rasterio as rio  # type: ignore[import]
import math

# 🔧 修复 Kornia 0.8.1 掩码变换 bug（与 GF-FloodNet 相同补丁）
import kornia.augmentation._2d.geometric.base as kornia_base  # type: ignore[import]
from kornia.constants import Resample  # type: ignore[import]

def _fixed_apply_transform_mask(self, input, params, flags, transform=None):
    resample_method = None
    if "resample" in flags:
        resample_method = flags["resample"]
        flags["resample"] = Resample.get("nearest")
    output = self.apply_transform(input, params, flags, transform)
    if resample_method is not None:
        flags["resample"] = resample_method
    return output

base_class = kornia_base.GeometricAugmentationBase2D
if not hasattr(base_class, "_wf2_patch_applied"):
    base_class._original_apply_transform_mask = base_class.apply_transform_mask
    base_class.apply_transform_mask = _fixed_apply_transform_mask
    base_class._wf2_patch_applied = True

from ..datasets.worldfloodsv2 import (
    WorldFloodsv2,
    WorldFloodsv2Tiled,
    CHANNELS_CONFIGURATIONS,
    get_list_of_window_slices,
)
from torch.utils.data import DataLoader  # type: ignore[import]
from torch.utils.data import Dataset as TorchDataset  # type: ignore[import]


# ==== 滑窗整图推理 Callback（内嵌，便于由 data 配置统一控制）====
import lightning as L  # type: ignore[import]


def _safe_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate that always creates fresh, contiguous storages for tensors.

    仅堆叠张量键；其它类型保持为列表。
    """
    if len(batch) == 0:
        return {}
    keys = batch[0].keys()
    output: Dict[str, Any] = {}
    for k in keys:
        items = [b[k] for b in batch]
        if torch.is_tensor(items[0]):
            output[k] = torch.stack([t.clone().contiguous() for t in items], dim=0)
        else:
            output[k] = items
    return output


def _pad_to_multiple(t: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    b, c, h, w = t.shape
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return t, (h, w)
    pad = torch.nn.ReflectionPad2d((0, pad_w, 0, pad_h))
    return pad(t), (h, w)


def _tile_batch(x: torch.Tensor, tile: int, pad: int) -> Tuple[List[Tuple[slice, slice]], List[torch.Tensor]]:
    _, _, H, W = x.shape
    coords: List[Tuple[slice, slice]] = []
    tiles: List[torch.Tensor] = []
    step = tile
    for i in range(0, H, step):
        for j in range(0, W, step):
            i0 = max(i - pad, 0)
            j0 = max(j - pad, 0)
            i1 = min(i + tile + pad, H)
            j1 = min(j + tile + pad, W)
            tiles.append(x[:, :, i0:i1, j0:j1])
            coords.append((slice(i, min(i + tile, H)), slice(j, min(j + tile, W))))
    return coords, tiles


class WF2SlidingWindowCallback(L.Callback):
    def __init__(
        self,
        stage: str = "test",
        tile_size: int = 1024,
        pad_size: int = 32,
        multiple_of: int = 8,
        save_predictions: bool = False,
        save_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert stage in {"val", "test"}
        self.stage = stage
        self.tile_size = tile_size
        self.pad_size = pad_size
        self.multiple_of = multiple_of
        self.save_predictions = save_predictions
        self.save_dir = save_dir

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.stage != "val":
            return
        self._run_sliding_window(trainer, pl_module, batch, split="val", index=batch_idx)

    def on_test_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.stage != "test":
            return
        self._run_sliding_window(trainer, pl_module, batch, split="test", index=batch_idx)

    @torch.no_grad()
    def _run_sliding_window(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Dict[str, Any], split: str, index: int) -> None:
        pl_module.eval()
        device = pl_module.device

        image = batch.get("image", None) or batch.get("image_optical", None) or batch.get("image_sar", None)
        if image is None:
            return
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)
        image, (H, W) = _pad_to_multiple(image, self.multiple_of)
        coords, tiles = _tile_batch(image, self.tile_size, self.pad_size)

        out_acc: Optional[torch.Tensor] = None
        for (si, sj), tile in zip(coords, tiles):
            pred = pl_module(tile)
            if out_acc is None:
                Bp, C = pred.shape[0], pred.shape[1]
                out_acc = torch.zeros((Bp, C, image.shape[2], image.shape[3]), device=device)
            assert out_acc is not None
            h_tile = si.stop - si.start
            w_tile = sj.stop - sj.start
            ti = (tile.shape[2] - h_tile) // 2
            tj = (tile.shape[3] - w_tile) // 2
            pred_cropped = pred[:, :, ti:ti + h_tile, tj:tj + w_tile]
            out_acc[:, :, si, sj] = pred_cropped

        if out_acc is None:
            return
        out_acc = out_acc[:, :, :H, :W]

        if self.save_predictions:
            import os
            save_dir = self.save_dir or os.path.join(trainer.default_root_dir, f"wf2_{self.stage}_preds")
            os.makedirs(save_dir, exist_ok=True)
            meta = batch.get("meta", {}) or {}
            paths = meta.get("path") if isinstance(meta, dict) else None
            if isinstance(paths, str):
                paths = [paths]
            assert out_acc is not None
            Bsave = out_acc.shape[0]
            for b in range(Bsave):
                default_name = f"{split}_{index:05d}_{b:03d}"
                stem = default_name
                if isinstance(paths, list) and b < len(paths):
                    stem = os.path.splitext(os.path.basename(paths[b]))[0]
                torch.save(out_acc[b].detach().cpu(), os.path.join(save_dir, f"{stem}_pred.pt"))


def percentile_normalize_batch(batch: torch.Tensor, min_p: float = 2.0, max_p: float = 98.0) -> torch.Tensor:
    """逐样本逐通道百分位缩放至 [0,1]。

    输入 batch: [B, C, H, W] float32
    """
    if batch.numel() == 0:
        return batch
    b, c, h, w = batch.shape
    flat = batch.view(b * c, h * w)
    q_min = torch.quantile(flat, q=min_p / 100.0, dim=1, keepdim=True)
    q_max = torch.quantile(flat, q=max_p / 100.0, dim=1, keepdim=True)
    denom = q_max - q_min
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    norm = (flat - q_min) / denom
    norm = torch.clamp(norm, 0.0, 1.0)
    return norm.view(b, c, h, w)


class WorldFloodsv2DataModule(NonGeoDataModule):
    """Lightning DataModule for WorldFloodsv2.

    Args:
        batch_size: 批大小
        num_workers: DataLoader workers
        train_transforms/val_transforms/test_transforms: 几何增强
        normalization_mean/std: 可选 Z-Score 标准化参数
        其余 **kwargs 透传给 WorldFloodsv2 数据集（如 root/split/channels/target_type 等）。
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transforms: Optional[Callable[[dict], dict]] = None,
        val_transforms: Optional[Callable[[dict], dict]] = None,
        test_transforms: Optional[Callable[[dict], dict]] = None,
        normalization_mean: Optional[list[float]] = None,
        normalization_std: Optional[list[float]] = None,
        official_normalization: bool = True,
        random_crop_size: Optional[tuple[int, int] | list[int]] = None,
        window_size: Optional[list[int] | tuple[int, int]] = (256, 256),
        # 测试阶段：使用 tiles 代替整图，避免 OOM 并与训练流程一致
        test_use_tiles: bool = True,
        # 验证/测试：跳过整块为忽略标签(如0/3)的 tiles，避免无有效像元导致指标跳过
        skip_all_ignore_tiles: bool = True,
        # 仅保留官方 filter_windows（v1/v2），移除手工阈值过滤
        # 可选：追加 MNDWI 输入通道（需包含 B3/B11 原始波段）
        add_mndwi_input: bool = False,
        # 官方 filter_windows 风格开关（v1/v2）
        filter_windows: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset_class=WorldFloodsv2,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

        self.train_aug = train_transforms
        self.val_aug = val_transforms
        self.test_aug = test_transforms

        self.official_normalization = official_normalization
        self.random_crop_size = tuple(random_crop_size) if random_crop_size is not None else None
        ws = tuple(window_size) if window_size is not None else (256, 256)
        if len(ws) != 2:
            raise ValueError(f"window_size 必须为 [H, W]，收到: {window_size}")
        self.window_size: tuple[int, int] = (int(ws[0]), int(ws[1]))
        # 过滤与 MNDWI
        self.add_mndwi_input = bool(add_mndwi_input)
        self.filter_windows_cfg = filter_windows
        self.test_use_tiles = bool(test_use_tiles)
        self.skip_all_ignore_tiles = bool(skip_all_ignore_tiles)

        self.mean = torch.tensor(normalization_mean) if normalization_mean is not None else None
        self.std = torch.tensor(normalization_std) if normalization_std is not None else None

        # val/test 阶段图片尺寸填充的对齐倍数：默认 32；若配置中提供 sliding_window.multiple_of，则使用之
        sw_cfg = self.kwargs.get("sliding_window", None)
        if isinstance(sw_cfg, dict) and isinstance(sw_cfg.get("multiple_of", None), int):
            self.multiple_of_pad = int(sw_cfg["multiple_of"]) if int(sw_cfg["multiple_of"]) > 0 else 32
        else:
            self.multiple_of_pad = 32

        # 如果用户未提供 transforms 且指定了 random_crop_size，按约定创建默认裁剪
        if self.random_crop_size is not None:
            if self.train_aug is None:
                self.train_aug = K.AugmentationSequential(
                    K.RandomCrop(self.random_crop_size, keepdim=True),
                    data_keys=["input", "mask"],
                )
            if self.val_aug is None:
                self.val_aug = K.AugmentationSequential(
                    K.CenterCrop(self.random_crop_size, keepdim=True),
                    data_keys=["input", "mask"],
                )
            if self.test_aug is None:
                self.test_aug = K.AugmentationSequential(
                    K.CenterCrop(self.random_crop_size, keepdim=True),
                    data_keys=["input", "mask"],
                )

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        # 统一键名：Dataset 输出为 {image, mask}
        image = batch["image"]
        mask = batch["mask"]

        # 确保图像为 float32（避免 pad 对 Long 报错）
        if not torch.is_floating_point(image):
            image = image.float()

        # 正式对齐：默认关闭百分位归一化
        if not self.official_normalization:
            image = percentile_normalize_batch(image)

        # 2) 几何增强
        stage = "predict"
        if getattr(self, "trainer", None):
            if self.trainer.training:
                stage = "train"
            elif self.trainer.validating or self.trainer.sanity_checking:
                stage = "val"
            elif self.trainer.testing:
                stage = "test"

        aug = {"train": self.train_aug, "val": self.val_aug, "test": self.test_aug, "predict": None}[stage]
        if aug is not None:
            image, mask = aug(image, mask)
            if len(mask.shape) == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)

        # 3) 标准化
        # 官方模式：使用 SENTINEL2_NORMALIZATION（根据 channels 选择子集）
        # 自定义模式：使用外部传入的 mean/std
        if self.official_normalization:
            # 从 kwargs 中解析 channels（要求为字符串且能在 CHANNELS_CONFIGURATIONS 中找到）
            channels_cfg = self.kwargs.get("channels", "all")
            if isinstance(channels_cfg, str):
                if channels_cfg not in CHANNELS_CONFIGURATIONS:
                    raise ValueError(
                        f"官方标准化要求 channels 为命名配置之一，收到：{channels_cfg}. 可选：{list(CHANNELS_CONFIGURATIONS)}"
                    )
                idxs = CHANNELS_CONFIGURATIONS[channels_cfg]
            else:
                raise ValueError("官方标准化模式下，请将 channels 设置为字符串配置（如 'rgb'、'bgriswirs' 等）")

            # 来自第三方仓库 ml4floods 的常量（S2 各 band 的均值/方差），按官方顺序 13 个波段
            SENTINEL2_NORMALIZATION = torch.tensor([
                [3787.0604973, 2634.44474043],
                [3758.07467509, 2794.09579088],
                [3238.08247208, 2549.4940614],
                [3418.90147615, 2811.78109878],
                [3450.23315812, 2776.93269704],
                [4030.94700446, 2632.13814197],
                [4164.17468251, 2657.43035126],
                [3981.96268494, 2500.47885249],
                [4226.74862547, 2589.29159887],
                [1868.29658114, 1820.90184704],
                [ 399.3878948 ,  761.3640411 ],
                [2391.66101119, 1500.02533014],
                [1790.32497137, 1241.9817628 ],
            ], dtype=torch.float32)

            mean = SENTINEL2_NORMALIZATION[idxs, 0].to(image.device)
            std = SENTINEL2_NORMALIZATION[idxs, 1].to(image.device)
            normalizer = K.Normalize(mean=mean, std=std)
            image = normalizer(image)
        else:
            if self.mean is not None and self.std is not None:
                normalizer = K.Normalize(mean=self.mean.to(image.device), std=self.std.to(image.device))
                image = normalizer(image)

        batch["image"] = image
        batch["mask"] = mask
        # 额外：为分层/窗口编码器规避尺寸约束，验证/测试阶段对输入进行 multiple_of 倍数填充（默认32）
        # 仅填充 image，不改变 mask；模型输出应在上游对齐回掩膜尺寸
        if stage in ("val", "test", "predict"):
            h, w = batch["image"].shape[-2], batch["image"].shape[-1]
            m = int(getattr(self, "multiple_of_pad", 32))
            if (h % m != 0) or (w % m != 0):
                padded, _ = _pad_to_multiple(batch["image"], m)
                batch["image"] = padded
        return batch

    # 供推理脚本直接调用，复用与 on_after_batch_transfer 相同的处理逻辑
    def process_batch(self, batch: dict, stage: str = "test") -> dict:
        image = batch["image"]
        mask = batch["mask"]

        # 确保图像为 float32（避免 pad 对 Long 报错）
        if not torch.is_floating_point(image):
            image = image.float()

        if not self.official_normalization:
            image = percentile_normalize_batch(image)

        aug = {"train": self.train_aug, "val": self.val_aug, "test": self.test_aug, "predict": None}.get(stage, None)
        if aug is not None:
            image, mask = aug(image, mask)
            if len(mask.shape) == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)

        if self.official_normalization:
            channels_cfg = self.kwargs.get("channels", "all")
            if isinstance(channels_cfg, str):
                if channels_cfg not in CHANNELS_CONFIGURATIONS:
                    raise ValueError(
                        f"官方标准化要求 channels 为命名配置之一，收到：{channels_cfg}. 可选：{list(CHANNELS_CONFIGURATIONS)}"
                    )
                idxs = CHANNELS_CONFIGURATIONS[channels_cfg]
            else:
                raise ValueError("官方标准化模式下，请将 channels 设置为字符串配置（如 'rgb'、'bgriswirs' 等）")

            SENTINEL2_NORMALIZATION = torch.tensor([
                [3787.0604973, 2634.44474043],
                [3758.07467509, 2794.09579088],
                [3238.08247208, 2549.4940614],
                [3418.90147615, 2811.78109878],
                [3450.23315812, 2776.93269704],
                [4030.94700446, 2632.13814197],
                [4164.17468251, 2657.43035126],
                [3981.96268494, 2500.47885249],
                [4226.74862547, 2589.29159887],
                [1868.29658114, 1820.90184704],
                [ 399.3878948 ,  761.3640411 ],
                [2391.66101119, 1500.02533014],
                [1790.32497137, 1241.9817628 ],
            ], dtype=torch.float32)

            mean = SENTINEL2_NORMALIZATION[idxs, 0].to(image.device)
            std = SENTINEL2_NORMALIZATION[idxs, 1].to(image.device)
            normalizer = K.Normalize(mean=mean, std=std)
            image = normalizer(image)
        else:
            if self.mean is not None and self.std is not None:
                normalizer = K.Normalize(mean=self.mean.to(image.device), std=self.std.to(image.device))
                image = normalizer(image)

        out = {
            "image": image,
            "mask": mask,
        }
        if stage in ("val", "test", "predict"):
            h, w = out["image"].shape[-2], out["image"].shape[-1]
            mlt = int(getattr(self, "multiple_of_pad", 32))
            if (h % mlt != 0) or (w % mlt != 0):
                padded, _ = _pad_to_multiple(out["image"], mlt)
                out["image"] = padded
        return out

    def setup(self, stage: str) -> None:
        # 官方对齐：
        # - 训练/验证：在线切片（无重叠网格窗口），使用 Tiled 数据集
        # - 测试：整图逐张
        root = self.kwargs.get("root")
        # 仅向 Dataset 透传其支持的参数，避免传入如 sliding_window/filter_windows 等无关键
        allowed_ds_keys = {"root", "channels", "target_type", "water_values", "ignore_index", "ignore_values", "add_mndwi_input"}
        base_kwargs = {k: v for k, v in self.kwargs.items() if k in allowed_ds_keys}

        if stage in ["fit", "validate"]:
            # 构造 train 基础数据集以扫描文件对
            ds_train = self.dataset_class(split="train", **base_kwargs)
            ds_val: Optional[WorldFloodsv2] = None
            try:
                ds_val = self.dataset_class(split="val", **base_kwargs)
            except Exception:
                ds_val = None

            # 在线切片窗口
            train_windows = get_list_of_window_slices(ds_train.samples, self.window_size)
            # 优先：官方 filter_windows（v1/v2）
            if self.filter_windows_cfg and self.filter_windows_cfg.get("apply", False):
                version = str(self.filter_windows_cfg.get("version", "v1")).lower()
                thr = float(self.filter_windows_cfg.get("threshold_clouds", 0.5))
                filtered: List[Dict[str, Any]] = []
                for info in train_windows:
                    msk_path, win = info["msk"], info["window"]
                    try:
                        with rio.open(msk_path) as src_m:
                            label = src_m.read(window=win, boundless=True, fill_value=0)
                        # 严格对齐官方：
                        # v1: 使用单通道(0/3) 比例；
                        # v2: 假定第1通道为云，第2通道为水 → 使用 (label[1]==0) 比例；
                        # 若不足2通道则回退至 v1 判据。
                        if label.ndim == 2:
                            # 单通道
                            H, W = label.shape
                            if version == "v2":
                                # 不足2通道，回退 v1
                                frac = (np.logical_or(label == 0, label == 3).sum()) / float(H * W)
                            else:
                                frac = (np.logical_or(label == 0, label == 3).sum()) / float(H * W)
                        else:
                            # 多通道 [C,H,W]
                            C, H, W = label.shape[0], label.shape[1], label.shape[2]
                            if version == "v2" and C >= 2:
                                frac = ((label[1] == 0).sum()) / float(H * W)
                            elif version == "v1" and C >= 1:
                                base = label[0]
                                frac = (np.logical_or(base == 0, base == 3).sum()) / float(H * W)
                            else:
                                # 其它情况回退 v1
                                base = label[0]
                                frac = (np.logical_or(base == 0, base == 3).sum()) / float(H * W)
                        if frac < thr:
                            filtered.append(info)
                    except Exception:
                        # 读失败则丢弃
                        pass
                train_windows = filtered
            # 不启用手工阈值过滤
            self.train_dataset = WorldFloodsv2Tiled(
                list_of_windows=train_windows,
                channels=self.kwargs.get("channels", "all"),
                target_type=self.kwargs.get("target_type", "binary"),
                water_values=self.kwargs.get("water_values", (2,)),
                ignore_index=self.kwargs.get("ignore_index", -1),
                ignore_values=self.kwargs.get("ignore_values", (0, 3)),
                transforms=None,
                add_mndwi_input=self.add_mndwi_input,
            )

            if ds_val is not None:
                val_windows = get_list_of_window_slices(ds_val.samples, self.window_size)
                # 评估阶段可选：跳过整块为忽略标签(0/3)的 tiles，稳定指标
                if self.skip_all_ignore_tiles:
                    ign_vals = set(int(v) for v in self.kwargs.get("ignore_values", (0, 3)))
                    filtered_val: List[Dict[str, Any]] = []
                    for info in val_windows:
                        msk_path, win = info["msk"], info["window"]
                        try:
                            with rio.open(msk_path) as src_m:
                                if src_m.count == 1:
                                    base = src_m.read(1, window=win, boundless=True, fill_value=0)
                                else:
                                    bands = src_m.read(window=win, boundless=True, fill_value=0)
                                    base = bands[1] if bands.shape[0] >= 2 else bands[0]
                            # 若存在任一非忽略值，则保留该 tile
                            if not np.isin(base, list(ign_vals)).all():
                                filtered_val.append(info)
                        except Exception:
                            # 读失败则保留，避免过度删减
                            filtered_val.append(info)
                    val_windows = filtered_val
                self.val_dataset = WorldFloodsv2Tiled(
                    list_of_windows=val_windows,
                    channels=self.kwargs.get("channels", "all"),
                    target_type=self.kwargs.get("target_type", "binary"),
                    water_values=self.kwargs.get("water_values", (2,)),
                    ignore_index=self.kwargs.get("ignore_index", -1),
                    ignore_values=self.kwargs.get("ignore_values", (0, 3)),
                    transforms=None,
                    add_mndwi_input=self.add_mndwi_input,
                )
            # 若无 val split，则不构建 val_dataset（保持 None）

        if stage in ["test"]:
            if not hasattr(self, "test_dataset") or self.test_dataset is None:
                if self.test_use_tiles:
                    ds_test = self.dataset_class(split="test", **base_kwargs)
                    test_windows = get_list_of_window_slices(ds_test.samples, self.window_size)
                    if self.skip_all_ignore_tiles:
                        ign_vals = set(int(v) for v in self.kwargs.get("ignore_values", (0, 3)))
                        filtered_t: List[Dict[str, Any]] = []
                        for info in test_windows:
                            msk_path, win = info["msk"], info["window"]
                            try:
                                with rio.open(msk_path) as src_m:
                                    if src_m.count == 1:
                                        base = src_m.read(1, window=win, boundless=True, fill_value=0)
                                    else:
                                        bands = src_m.read(window=win, boundless=True, fill_value=0)
                                        base = bands[1] if bands.shape[0] >= 2 else bands[0]
                                if not np.isin(base, list(ign_vals)).all():
                                    filtered_t.append(info)
                            except Exception:
                                filtered_t.append(info)
                        test_windows = filtered_t
                    self.test_dataset = WorldFloodsv2Tiled(
                        list_of_windows=test_windows,
                        channels=self.kwargs.get("channels", "all"),
                        target_type=self.kwargs.get("target_type", "binary"),
                        water_values=self.kwargs.get("water_values", (2,)),
                        ignore_index=self.kwargs.get("ignore_index", -1),
                        ignore_values=self.kwargs.get("ignore_values", (0, 3)),
                        transforms=None,
                        add_mndwi_input=self.add_mndwi_input,
                    )
                else:
                    self.test_dataset = self.dataset_class(split="test", **base_kwargs)

    @property
    def num_classes(self) -> int:
        # Dataset 默认是 binary；若在实例化数据集时 target_type='multiclass'，此属性可被上层覆盖
        try:
            ds = self.dataset if hasattr(self, "dataset") else self.dataset_class(**self.kwargs)
            return 2 if getattr(ds, "target_type", "binary") == "binary" else int(torch.max(torch.tensor([3])))
        except Exception:
            return 2

    # 官方对齐：test batch_size 固定 1
    def test_dataloader(self) -> DataLoader:
        # 兼容某些 Lightning 版本在 fit 之后不再次调用 setup("test") 的情况
        if not hasattr(self, "test_dataset") or (self.test_dataset is None):
            self.setup("test")
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    # 自定义 collate，避免默认行为触发不可调整存储的 resize_
    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "train_dataset"):
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=_safe_collate,
        )

    def val_dataloader(self) -> DataLoader:
        if not hasattr(self, "val_dataset"):
            self.setup("validate")
        if getattr(self, "val_dataset", None) is None:
            class _EmptyDataset(TorchDataset):
                def __len__(self) -> int:
                    return 0
                def __getitem__(self, index: int):
                    raise IndexError("Empty dataset")
            return DataLoader(_EmptyDataset())  # 占位，兼容无 val 的场景
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=_safe_collate,
        )

    # 供 predict 流程使用：与 test 复用同一数据集，避免 TorchGeo 的 MisconfigurationException
    def predict_dataloader(self) -> DataLoader:
        if not hasattr(self, "test_dataset") or (self.test_dataset is None):
            self.setup("test")
        # 为安全起见维持 batch_size=1，整图滑窗由 Writer 负责并行与内存管理
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)
