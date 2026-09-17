"""KuroSiwo洪水后图像分割的Lightning DataModule。

用于处理KuroSiwo洪水后图像分割的训练、验证和测试数据加载，
包括数据增强、标准化和批处理。

职责分离：
- Dataset 负责 I/O 与数据读取
- 本 DataModule 负责批处理级别的标准化与几何增强
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from collections.abc import Sequence
import json
import time

import kornia.augmentation as K
import torch
from torch.utils.data import DataLoader
import lightning as L
import os
import sys

# 修复 Kornia 0.8.1 掩码变换 bug（与其它模块相同补丁）
import kornia.augmentation._2d.geometric.base as kornia_base
from kornia.constants import Resample

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
if not hasattr(base_class, "_kurosiwo_patch_applied"):
    base_class._original_apply_transform_mask = base_class.apply_transform_mask
    base_class.apply_transform_mask = _fixed_apply_transform_mask
    base_class._kurosiwo_patch_applied = True

from ..datasets.kurosiwo import KuroSiwoDataset


class KuroSiwoDataModule(L.LightningDataModule):
    """KuroSiwo洪水后图像分割的Lightning DataModule

    仅使用洪水后的SAR图像进行洪水分段任务。

    Args:
        root: 数据根目录路径
        batch_size: 批大小
        num_workers: DataLoader workers数量
        train_transforms/val_transforms/test_transforms: 几何增强函数
        channels: 通道配置，如 ["vv", "vh"]
        scale_input: 标准化方式 ("normalize", "min-max", "custom")
        data_mean: 数据均值（用于标准化）
        data_std: 数据标准差（用于标准化）
        dem: 是否使用DEM数据作为辅助特征
        train_acts: 训练集激活ID列表
        val_acts: 验证集激活ID列表
        test_acts: 测试集激活ID列表
        **kwargs: 传递给Dataset的其他参数
    """

    def __init__(
        self,
        root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transforms: Optional[Callable[[Dict], Dict]] = None,
        val_transforms: Optional[Callable[[Dict], Dict]] = None,
        test_transforms: Optional[Callable[[Dict], Dict]] = None,
        channels: List[str] = None,
        scale_input: str = "normalize",
        data_mean: List[float] = None,
        data_std: List[float] = None,
        dem: bool = False,
        train_acts: List[int] = None,
        val_acts: List[int] = None,
        test_acts: List[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 数据增强
        self.train_aug = train_transforms
        self.val_aug = val_transforms
        self.test_aug = test_transforms

        # 数据集参数
        self.channels = channels or ["vv", "vh"]
        self.scale_input = scale_input
        # 不再使用“或”回退，允许在log模式下传入None以仅做log1p
        self.data_mean = data_mean
        self.data_std = data_std
        self.dem = dem
        self.train_acts = train_acts
        self.val_acts = val_acts
        self.test_acts = test_acts

        self.dataset_kwargs = kwargs

        self.train_dataset: Optional[KuroSiwoDataset] = None
        self.val_dataset: Optional[KuroSiwoDataset] = None
        self.test_dataset: Optional[KuroSiwoDataset] = None

        self._dbg_count = {"train": 0, "val": 0, "test": 0}

    def prepare_data(self) -> None:  # type: ignore[override]
        return None

    def on_exception(self, exception: BaseException) -> None:  # type: ignore[override]
        return None

    def setup(self, stage: str) -> None:  # type: ignore[override]
        dataset_kwargs = {
            "root": self.root,
            "channels": self.channels,
            "scale_input": self.scale_input,
            "data_mean": self.data_mean,
            "data_std": self.data_std,
            "dem": self.dem,
            "train_acts": self.train_acts,
            "val_acts": self.val_acts,
            "test_acts": self.test_acts,
            **self.dataset_kwargs
        }

        if stage in ["fit", "validate"]:
            self.train_dataset = KuroSiwoDataset(split="train", transforms=None, **dataset_kwargs)
            try:
                self.val_dataset = KuroSiwoDataset(split="val", transforms=None, **dataset_kwargs)
            except Exception as e:
                print(f"Warning: Could not create validation dataset: {e}")
                self.val_dataset = None

            # 使用配置中的全局均值/方差；不再自动统计

            # 打印各阶段事件样本统计
            try:
                from collections import Counter
                trn_counts = Counter([s["activation"] for s in getattr(self.train_dataset, "samples", [])])
                print(f"[KURO-DM] train: total_samples={len(self.train_dataset)}, events={len(trn_counts)}", file=sys.stderr)
                print(f"[KURO-DM] train events: {dict(trn_counts.most_common())}", file=sys.stderr)
                
                if self.val_dataset is not None:
                    val_counts = Counter([s["activation"] for s in getattr(self.val_dataset, "samples", [])])
                    print(f"[KURO-DM] val: total_samples={len(self.val_dataset)}, events={len(val_counts)}", file=sys.stderr)
                    print(f"[KURO-DM] val events: {dict(val_counts.most_common())}", file=sys.stderr)
            except Exception:
                pass

        if stage in ["test"]:
            self.test_dataset = KuroSiwoDataset(split="test", transforms=None, **dataset_kwargs)
            # 打印测试集事件样本统计
            try:
                from collections import Counter
                test_counts = Counter([s["activation"] for s in getattr(self.test_dataset, "samples", [])])
                print(f"[KURO-DM] test: total_samples={len(self.test_dataset)}, events={len(test_counts)}", file=sys.stderr)
                print(f"[KURO-DM] test events: {dict(test_counts.most_common())}", file=sys.stderr)
            except Exception:
                pass

    def train_dataloader(self) -> DataLoader:  # type: ignore[override]
        if self.train_dataset is None:
            self.setup("fit")
        assert self.train_dataset is not None
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> Optional[DataLoader]:  # type: ignore[override]
        if self.val_dataset is None:
            self.setup("fit")
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:  # type: ignore[override]
        if self.test_dataset is None:
            self.setup("test")
        assert self.test_dataset is not None
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self) -> DataLoader:  # type: ignore[override]
        return self.test_dataloader()

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def classes(self) -> List[str]:
        return ["background", "water"]

    # ---- helpers ----

    def on_before_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:  # type: ignore[override]
        return batch

    def on_after_batch_transfer(self, batch: Dict[str, Any], dataloader_idx: int) -> Dict[str, Any]:  # type: ignore[override]
        stage = "predict"
        if hasattr(self, "trainer") and self.trainer is not None:
            if getattr(self.trainer, "training", False):
                stage = "train"
            elif getattr(self.trainer, "validating", False) or getattr(self.trainer, "sanity_checking", False):
                stage = "val"
            elif getattr(self.trainer, "testing", False):
                stage = "test"

        aug = {"train": self.train_aug, "val": self.val_aug, "test": self.test_aug, "predict": None}[stage]
        image = batch["image"]
        # DEM 早期融合已在 Dataset 中完成，此处不再处理 dem 键
        mask = batch["mask"]
        if aug is not None:
            image, mask = aug(image, mask)
            if len(mask.shape) == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            batch["image"], batch["mask"] = image, mask

        # 一致性校验：通道数 = SAR通道数(len(channels)) + DEM通道数(若启用则+1)
        base_c = len(self.channels) if isinstance(self.channels, Sequence) else 2
        expected_c = base_c + (1 if self.dem else 0)

        if isinstance(batch["image"], torch.Tensor) and batch["image"].dim() == 4:
            c = batch["image"].shape[1]
            if c != expected_c:
                raise ValueError(
                    f"KuroSiwo: 输入通道数不一致，得到 {c} 通道，期望 {expected_c} 通道。"
                    f" 当前 channels={self.channels}, dem={self.dem}, scale_input={self.scale_input}。"
                    f" 请检查 configs/data/kurosiwo.yaml 以及提交脚本中的 data.channels / data.sar_channels 是否一致。"
                )

        # 移除详细调试信息，保持简洁
        return batch

    def transfer_batch_to_device(self, batch: Dict[str, Any], device: torch.device, dataloader_idx: int) -> Dict[str, Any]:  # type: ignore[override]
        batch_on_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_on_device[key] = value.to(device)
            else:
                batch_on_device[key] = value
        return batch_on_device
