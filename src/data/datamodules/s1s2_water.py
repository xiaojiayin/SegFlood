"""S1S2-Water DataModule。

简化设计：
- modal_type + add_dem/add_slope 控制
- DEM/SLOPE 总是加在光学后
- 支持几何增强和数值增强
- 不做归一化
"""

from typing import Any, Callable
import os

import torch
import kornia.augmentation as K  # type: ignore[import]
from torchgeo.datamodules import NonGeoDataModule  # type: ignore[import]
from torch.utils.data import DataLoader  # type: ignore[import]

from ..datasets.s1s2_water import S1S2Water


class S1S2WaterDataModule(NonGeoDataModule):
    """LightningDataModule for S1S2-Water。

    Args:
        modal_type: "sar" | "optical" | "dual"
        add_dem: 是否添加 DEM 到光学分支，默认 False
        add_slope: 是否添加 SLOPE 到光学分支，默认 False
        root: 统一多通道根目录
        train_transforms: 训练时几何增强
        val_transforms: 验证时几何增强  
        test_transforms: 测试时几何增强
        optical_value_transforms: 光学数值增强（ColorJitter等）
        sar_value_transforms: SAR数值增强
        enable_sar_speckle: 是否启用SAR散斑噪声
        sar_speckle_sigma_range: 散斑噪声强度范围
        sar_speckle_prob: 散斑噪声概率
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        # 模态控制
        modal_type: str = "dual",
        add_dem: bool = False,
        add_slope: bool = False,
        # --- 🎯 新增：早期融合支持 ---
        output_image_key: bool = False,  # True时强制输出单键image（早期融合baseline）
        # 数据路径
        root: str | None = None,
        split_train: str = "train",
        split_val: str = "val", 
        split_test: str = "test",
        # 几何增强
        train_transforms: Callable[[dict], dict] | None = None,
        val_transforms: Callable[[dict], dict] | None = None,
        test_transforms: Callable[[dict], dict] | None = None,
        # 数值增强
        optical_value_transforms: Callable[[dict], dict] | None = None,
        sar_value_transforms: Callable[[dict], dict] | None = None,
        enable_sar_speckle: bool = False,
        sar_speckle_sigma_range: list[float] | None = None,
        sar_speckle_prob: float = 0.0,
        # 评估阶段缺模态测试：dual | optical_only | sar_only
        eval_modal_mode: str = "dual",
        **kwargs: Any,
    ) -> None:
        if not root:
            raise ValueError("需要提供统一 tiles 根目录 root")

        # 自动 fallback: root 不含 train/images 结构时尝试 root/S1S2-Water
        def _has_required(r: str) -> bool:
            return all(
                os.path.isdir(os.path.join(r, sp, "images")) and os.path.isdir(os.path.join(r, sp, "masks"))
                for sp in [split_train, split_val]
            )
        
        if not _has_required(root):
            candidate = os.path.join(root, "S1S2-Water")
            if os.path.isdir(candidate) and _has_required(candidate):
                root = candidate

        # 严格校验
        for sp in [split_train, split_val]:
            img_dir = os.path.join(root, sp, "images")
            msk_dir = os.path.join(root, sp, "masks")
            if not (os.path.isdir(img_dir) and os.path.isdir(msk_dir)):
                raise FileNotFoundError(f"缺失目录: {img_dir} 或 {msk_dir}")

        self.modal_type = modal_type
        self.add_dem = add_dem
        self.add_slope = add_slope
        self.output_image_key = output_image_key
        if eval_modal_mode not in {"dual", "optical_only", "sar_only"}:
            raise ValueError("eval_modal_mode must be one of: dual, optical_only, sar_only")
        self.eval_modal_mode = eval_modal_mode

        # 计算通道数
        self.optical_channels = 4  # RGB+NIR
        if add_dem:
            self.optical_channels += 1
        if add_slope:
            self.optical_channels += 1
        self.sar_channels = 2  # VV+VH

        dataset_kwargs: dict[str, Any] = dict(
            split=split_train,
            modal_type=modal_type,
            root=root,
            add_dem=add_dem,
            add_slope=add_slope,
        )

        super().__init__(
            dataset_class=S1S2Water,
            batch_size=batch_size,
            num_workers=num_workers,
            **dataset_kwargs,
        )

        # DataLoader 性能参数
        self.pin_memory = True
        self.persistent_workers = True

        # 增强
        self.train_aug = train_transforms
        self.val_aug = val_transforms
        self.test_aug = test_transforms
        self.optical_value_aug = optical_value_transforms
        self.sar_value_aug = sar_value_transforms
        self.enable_sar_speckle = enable_sar_speckle
        self.speckle_sigma_range = sar_speckle_sigma_range or [0.03, 0.10]
        self.speckle_prob = sar_speckle_prob

        # 记录路径与 splits
        self._paths = dict(root=root)
        self._splits = dict(train=split_train, val=split_val, test=split_test)

        print(
            "[S1S2WaterDataModule] 初始化:"
            f" modal_type={modal_type}, add_dem={add_dem}, add_slope={add_slope},"
            f" optical_channels={self.optical_channels}, sar_channels={self.sar_channels}, root={root}"
        )

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        stage = "predict"
        if getattr(self, "trainer", None):
            if self.trainer.training:
                stage = "train"
            elif self.trainer.validating or self.trainer.sanity_checking:
                stage = "val"
            elif self.trainer.testing:
                stage = "test"
        return self.process_batch(batch, stage)

    def process_batch(self, batch: dict, stage: str) -> dict:
        mask = batch["mask"]
        
        # 判定是否双模态
        has_opt = "image_optical" in batch
        has_sar = "image_sar" in batch
        is_dual = has_opt and has_sar
        
        if is_dual:
            image = torch.cat([batch["image_optical"], batch["image_sar"]], dim=1)
        else:
            if has_opt:
                image = batch["image_optical"]
            elif has_sar:
                image = batch["image_sar"]
            else:
                raise KeyError("batch 中缺少 image_optical 或 image_sar 键")

        # 几何增强
        transform = None
        if stage == "train":
            transform = self.train_aug
        elif stage == "val":
            transform = self.val_aug
        elif stage == "test":
            transform = self.test_aug
            
        if transform is not None:
            if hasattr(transform, "data_keys") and getattr(transform, "data_keys"):
                try:
                    transform.data_keys = None  # type: ignore
                except Exception:
                    pass
            sample_tmp = {"input": image, "mask": mask}
            sample_tmp = transform(sample_tmp)
            image = sample_tmp.get("input", sample_tmp.get("image", image))
            mask = sample_tmp.get("mask", mask)
            if isinstance(mask, torch.Tensor) and mask.dim() == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)

        # 数值增强（仅训练阶段）
        if stage == "train":
            # 切分为 optical/sar 部分
            if is_dual:
                opt_part = image[:, :self.optical_channels]
                sar_part = image[:, self.optical_channels:]
            elif has_opt:
                opt_part = image
                sar_part = None
            else:
                opt_part = None
                sar_part = image

            # 光学数值增强：仅对 RGB+NIR（前4个通道）
            if opt_part is not None and self.optical_value_aug is not None:
                opt_base = opt_part[:, :4]  # RGB+NIR
                opt_extra = opt_part[:, 4:] if opt_part.shape[1] > 4 else None
                try:
                    sample_opt = {"input": opt_base}
                    sample_opt = self.optical_value_aug(sample_opt)
                    opt_base = sample_opt.get("input", sample_opt.get("image", opt_base))
                except Exception:
                    pass
                if opt_extra is not None:
                    opt_part = torch.cat([opt_base, opt_extra], dim=1)
                else:
                    opt_part = opt_base

            # SAR数值增强
            if sar_part is not None and self.sar_value_aug is not None:
                try:
                    sample_sar = {"input": sar_part}
                    sample_sar = self.sar_value_aug(sample_sar)
                    sar_part = sample_sar.get("input", sample_sar.get("image", sar_part))
                except Exception:
                    pass

            # SAR散斑噪声：仅对VV/VH（前2个通道）
            if sar_part is not None and self.enable_sar_speckle and self.speckle_prob > 0.0:
                if torch.rand(1).item() < self.speckle_prob:
                    vv_vh = sar_part[:, :2]  # VV/VH
                    sigma = torch.empty(1).uniform_(self.speckle_sigma_range[0], self.speckle_sigma_range[1]).item()
                    noise = torch.randn_like(vv_vh) * sigma
                    sar_part[:, :2] = vv_vh * (1.0 + noise)

            # 合并回 image
            if is_dual and opt_part is not None and sar_part is not None:
                image = torch.cat([opt_part, sar_part], dim=1)
            elif opt_part is not None:
                image = opt_part
            elif sar_part is not None:
                image = sar_part

        # 还原格式
        if self.output_image_key:
            # 早期融合模式：强制输出单键image（用于单编码器baseline）
            batch["image"] = image
            # 移除可能存在的双模态键
            batch.pop("image_optical", None)
            batch.pop("image_sar", None)
        elif self.modal_type == "dual" and stage in {"val", "test", "predict"} and self.eval_modal_mode != "dual":
            if self.eval_modal_mode == "optical_only":
                batch["image_optical"] = image[:, :self.optical_channels]
                batch.pop("image_sar", None)
            else:
                batch["image_sar"] = image[:, self.optical_channels:]
                batch.pop("image_optical", None)
        elif self.modal_type == "dual":
            batch["image_optical"] = image[:, :self.optical_channels]
            batch["image_sar"] = image[:, self.optical_channels:]
        elif self.modal_type == "optical":
            batch["image_optical"] = image
            batch.pop("image_sar", None)
        else:  # sar
            batch["image_sar"] = image
            batch.pop("image_optical", None)
            
        batch["mask"] = mask
        return batch

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            self.train_dataset = self._build_dataset(self._splits["train"]) 
            self.val_dataset = self._build_dataset(self._splits["val"]) 
        if stage in ["test"]:
            self.test_dataset = self._build_dataset(self._splits["test"]) 

    def _build_dataset(self, split: str) -> S1S2Water:
        return S1S2Water(
            root=self._paths["root"],
            split=split,
            modal_type=self.modal_type,
            add_dem=self.add_dem,
            add_slope=self.add_slope,
        )

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def num_channels(self) -> int:
        if self.modal_type == "dual":
            return self.optical_channels + self.sar_channels
        elif self.modal_type == "optical":
            return self.optical_channels
        else:
            return self.sar_channels

    # 供 predict 流程使用：与 test 共用一套数据
    def predict_dataloader(self) -> DataLoader:
        if not hasattr(self, "test_dataset") or self.test_dataset is None:
            self.setup("test")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)