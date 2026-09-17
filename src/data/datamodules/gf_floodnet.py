"""GF-FloodNet DataModule following TorchGeo standards."""

import fnmatch
import os
from typing import Any, Callable

import torch
import kornia.augmentation as K
from torch.utils.data import random_split, DataLoader, Subset

# 🔧 Kornia 0.8.1 关键Bug修复: GeometricAugmentationBase2D.apply_transform_mask
# 问题: resample_method变量未初始化导致UnboundLocalError
# 解决: 确保resample_method始终被初始化为None
import kornia.augmentation._2d.geometric.base as kornia_base
from kornia.constants import Resample

def _fixed_apply_transform_mask(self, input, params, flags, transform=None):
    """修复Kornia 0.8.1中resample_method UnboundLocalError的关键bug"""
    resample_method = None  # 🎯 关键修复：确保变量始终被初始化
    if "resample" in flags:
        resample_method = flags["resample"]
        flags["resample"] = Resample.get("nearest")
    output = self.apply_transform(input, params, flags, transform)
    if resample_method is not None:
        flags["resample"] = resample_method
    return output

# 应用补丁到正确的基类
base_class = kornia_base.GeometricAugmentationBase2D
base_class._original_apply_transform_mask = base_class.apply_transform_mask
base_class.apply_transform_mask = _fixed_apply_transform_mask

from torchgeo.datamodules import NonGeoDataModule

from ..datasets.gf_floodnet import GFFloodNet


def percentile_normalize_batch(batch: torch.Tensor, min_p: float = 2.0, max_p: float = 98.0) -> torch.Tensor:
    """对一个 batch 的图像执行逐样本、逐通道的百分位归一化到 [0, 1]。

    Args:
        batch: 输入张量，形状 [B, C, H, W] 且为浮点数类型。
        min_p: 最小百分位（默认 2.0）。
        max_p: 最大百分位（默认 98.0）。

    Returns:
        归一化后的张量，形状同输入，数值范围在 [0, 1]。
    """
    if batch.numel() == 0:
        return batch

    b, c, h, w = batch.shape
    flat = batch.view(b * c, h * w)

    # 计算每个样本每个通道的分位数
    q_min = torch.quantile(flat, q=min_p / 100.0, dim=1, keepdim=True)
    q_max = torch.quantile(flat, q=max_p / 100.0, dim=1, keepdim=True)

    # 避免除零：当 q_max == q_min 时使用 1.0 作为分母占位
    denom = (q_max - q_min)
    safe_denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    norm_flat = (flat - q_min) / safe_denom
    norm_flat = torch.clamp(norm_flat, 0.0, 1.0)
    return norm_flat.view(b, c, h, w)


class GFFloodNetDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the GF-FloodNet dataset.
    
    📋 Dataset Info (Based on original paper Section 2.3):
    Task: Semantic segmentation for flood area extraction
    Labels: 1=flood/water areas (all water bodies), 255=non-flood areas (background)
    
    This DataModule follows TorchGeo standards and best practices for satellite
    data processing. It includes proper normalization, data augmentation, and
    follows the NonGeoDataModule pattern.
    
    Band Organization (5 channels):
    - Band 0: Blue (GF-2 optical)
    - Band 1: Green (GF-2 optical)  
    - Band 2: Red (GF-2 optical)
    - Band 3: Near-Infrared (NIR, GF-2 optical)
    - Band 4: SAR Intensity (GF-3 C-band)
    
    Args:
        batch_size: Size of each mini-batch
        val_split_pct: Percentage of data to use for validation  
        test_split_pct: Percentage of data to use for testing
        num_workers: Number of workers for parallel data loading
        **kwargs: Additional keyword arguments passed to GFFloodNet dataset
    """

    # Pre-calculated channel statistics for 5-band GF-FloodNet data
    # Based on percentile normalization analysis (Job 110660, 2025-08-09)
    # Data correctly normalized to [0,1] using 2-98th percentile clipping
    mean = torch.tensor([0.666135, 0.404481, 0.247136, 0.350982, 0.371458])  # [Blue, Green, Red, NIR, SAR]
    std = torch.tensor([0.245092, 0.194524, 0.200037, 0.232912, 0.301114])   # Standard deviations from GPU analysis

    def __init__(
        self,
        batch_size: int = 32,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.1,
        num_workers: int = 4,
        modal_type: str = "dual",  # "optical", "sar", "dual"
        # --- 🎯 新增：早期融合支持 ---
        output_image_key: bool = False,  # True时强制输出单键image（早期融合baseline）
        train_transforms: Callable[[dict], dict] | None = None,
        val_transforms: Callable[[dict], dict] | None = None,
        test_transforms: Callable[[dict], dict] | None = None,
        # --- 新增：可配置的标准化参数 ---
        normalization_mean: list[float] | None = None,
        normalization_std: list[float] | None = None,
        # --- 新增：预测阶段是否使用完整数据集而不是 test split ---
        predict_use_full: bool = False,
        # --- 评估阶段缺模态测试：dual | optical_only | sar_only ---
        eval_modal_mode: str = "dual",
        # --- Cross-event / cross-region holdout：按 images 文件名匹配 ---
        holdout_patterns: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new GFFloodNetDataModule instance.
        
        Args:
            batch_size: Size of each mini-batch
            val_split_pct: Percentage of the dataset to use as a validation set
            test_split_pct: Percentage of the dataset to use as a test set  
            num_workers: Number of workers for parallel data loading
            modal_type: Type of modality ("optical", "sar", "dual")
            output_image_key: If True, always output single 'image' key (for early fusion baseline)
            train_transforms: Transforms for training data (from config)
            val_transforms: Transforms for validation data (from config)
            test_transforms: Transforms for test data (from config)
            normalization_mean: Mean values for normalization
            normalization_std: Std values for normalization
            **kwargs: Additional keyword arguments passed to GFFloodNet
        """
        self.modal_type = modal_type
        self.output_image_key = output_image_key
        if eval_modal_mode not in {"dual", "optical_only", "sar_only"}:
            raise ValueError("eval_modal_mode must be one of: dual, optical_only, sar_only")
        self.eval_modal_mode = eval_modal_mode
        self.holdout_patterns = [str(p) for p in (holdout_patterns or []) if str(p)]
        
        # 提取通道配置参数，避免传递给数据集类
        self.optical_channels = kwargs.pop('optical_channels', 4)
        self.sar_channels = kwargs.pop('sar_channels', 1)
        
        # 🎯 向数据集传递modal_type参数
        kwargs['modal_type'] = modal_type
        
        super().__init__(
            dataset_class=GFFloodNet,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
        
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.predict_use_full = predict_use_full

        # 配置标准化参数
        self.mean = torch.tensor(normalization_mean or [0.666135, 0.404481, 0.247136, 0.350982, 0.371458])
        self.std = torch.tensor(normalization_std or [0.245092, 0.194524, 0.200037, 0.232912, 0.301114])
        
        # 根据模态类型调整标准化参数
        if self.modal_type == "optical":
            self.mean = self.mean[:4]
            self.std = self.std[:4]
        elif self.modal_type == "sar":
            self.mean = self.mean[4:5]
            self.std = self.std[4:5]
        
        # 存储变换（将在 on_after_batch_transfer 中应用）
        self.train_aug = train_transforms
        self.val_aug = val_transforms
        self.test_aug = test_transforms
        self.predict_aug = None  # 预测时通常不增强

    def _split_by_holdout_patterns(self, dataset):
        """Return train/val/test subsets using filename patterns as held-out events/regions."""
        if not self.holdout_patterns:
            return None

        files = list(getattr(dataset, "files", []))
        holdout_indices = []
        trainval_indices = []
        for idx, path in enumerate(files):
            basename = os.path.basename(path)
            is_holdout = any(fnmatch.fnmatch(basename, pattern) for pattern in self.holdout_patterns)
            (holdout_indices if is_holdout else trainval_indices).append(idx)

        if len(holdout_indices) == 0:
            raise ValueError(
                f"holdout_patterns did not match any GF-FloodNet images: {self.holdout_patterns}"
            )
        if len(trainval_indices) == 0:
            raise ValueError(
                f"holdout_patterns matched all GF-FloodNet images: {self.holdout_patterns}"
            )

        val_samples = max(1, int(len(trainval_indices) * self.val_split_pct))
        generator = torch.Generator().manual_seed(42)
        perm = torch.randperm(len(trainval_indices), generator=generator).tolist()
        shuffled = [trainval_indices[i] for i in perm]
        val_indices = shuffled[:val_samples]
        train_indices = shuffled[val_samples:]

        print("📊 GF-FloodNet holdout split:")
        print(f"   holdout_patterns: {self.holdout_patterns}")
        print(f"   train samples: {len(train_indices)}")
        print(f"   val samples: {len(val_indices)}")
        print(f"   test holdout samples: {len(holdout_indices)}")

        return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, holdout_indices)
    
    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """
        PyTorch Lightning的钩子，用于在数据移动到设备后应用变换。
        这个方法现在会从self.trainer中推断stage。
        """
        # 从trainer推断当前阶段
        stage = "predict"  # 默认值
        if getattr(self, "trainer", None):
            if self.trainer.training:
                stage = "train"
            elif self.trainer.validating or self.trainer.sanity_checking:
                stage = "val"
            elif self.trainer.testing:
                stage = "test"

        return self.process_batch(batch, stage)

    def process_batch(self, batch: dict, stage: str) -> dict:
        """
        核心数据处理逻辑，与Trainer解耦。
        可被`on_after_batch_transfer`自动调用，也可被外部脚本手动调用。

        Args:
            batch: 输入的数据批次。
            stage: 当前的阶段 ('train', 'val', 'test', 'predict')。
        """
        mask = batch['mask']
        
        # 🎯 处理不同的输入格式
        if 'image' in batch:
            # 单模态格式: {"image": tensor, "mask": mask}
            image = batch['image']
            is_dual_modal = False
        elif 'image_optical' in batch and 'image_sar' in batch:
            # 双模态格式: {"image_optical": tensor, "image_sar": tensor, "mask": mask}
            # 合并为统一格式进行处理
            image_optical = batch['image_optical']
            image_sar = batch['image_sar']
            image = torch.cat([image_optical, image_sar], dim=1)  # [B, 5, H, W]
            is_dual_modal = True
        else:
            raise ValueError(f"不支持的batch格式，期望 'image' 或 'image_optical'+'image_sar'，实际: {list(batch.keys())}")
        
        # 步骤一：百分位归一化到 [0, 1]（逐样本逐通道），与GF-FloodNet官方流程一致
        image = percentile_normalize_batch(image)

        # 根据stage选择几何变换
        transform = None
        if stage == 'train':
            transform = self.train_aug
        elif stage == 'val':
            transform = self.val_aug
        elif stage == 'test':
            transform = self.test_aug
        elif stage == 'predict':
            transform = self.predict_aug
            
        # 几何变换（同步image和mask）
        if transform is not None:
            try:
                image, mask = transform(image, mask)
                # 修复Kornia可能添加的额外维度
                if len(mask.shape) == 4 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)
            except Exception as e:
                raise RuntimeError(f"变换配置错误，需要 data_keys=['input', 'mask']: {e}")
        
        # 标准化（只对image）
        normalizer = K.Normalize(mean=self.mean.to(image.device), std=self.std.to(image.device))
        image = normalizer(image)
        
        # 🎯 根据原始格式和配置返回对应结果
        if self.output_image_key:
            # 早期融合模式：强制输出单键image（用于单编码器baseline）
            batch['image'] = image
            # 移除可能存在的双模态键
            batch.pop('image_optical', None)
            batch.pop('image_sar', None)
        elif is_dual_modal and stage in {"val", "test", "predict"} and self.eval_modal_mode != "dual":
            if self.eval_modal_mode == "optical_only":
                batch['image_optical'] = image[:, :self.optical_channels, :, :]
                batch.pop('image_sar', None)
            else:
                batch['image_sar'] = image[:, self.optical_channels:, :, :]
                batch.pop('image_optical', None)
        elif is_dual_modal:
            # 分离回双模态格式
            batch['image_optical'] = image[:, :self.optical_channels, :, :]
            batch['image_sar'] = image[:, self.optical_channels:, :, :]
        else:
            # 保持单模态格式
            batch['image'] = image
            
        batch['mask'] = mask
        return batch

    def setup(self, stage: str) -> None:
        """Set up datasets for different stages.
        
        注意：fit/validate阶段都需要创建train/val/test三个分割，
        这是因为Lightning的训练和验证都需要访问验证集。
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage in ['fit', 'validate']:
            # Create the full dataset
            self.dataset = self.dataset_class(**self.kwargs)
            holdout_split = self._split_by_holdout_patterns(self.dataset)
            if holdout_split is not None:
                self.train_dataset, self.val_dataset, self.test_dataset = holdout_split
                return
            
            # Split into train/val/test
            total_samples = len(self.dataset)
            val_samples = int(total_samples * self.val_split_pct)
            test_samples = int(total_samples * self.test_split_pct)
            train_samples = total_samples - val_samples - test_samples
            
            # Use fixed seed for reproducible splits
            generator = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_samples, val_samples, test_samples],
                generator=generator
            )
            
        if stage in ['test']:
            if not hasattr(self, 'test_dataset') or self.test_dataset is None:
                # If test dataset not created yet, create full dataset and split
                self.dataset = self.dataset_class(**self.kwargs)
                holdout_split = self._split_by_holdout_patterns(self.dataset)
                if holdout_split is not None:
                    _, _, self.test_dataset = holdout_split
                    return
                total_samples = len(self.dataset)
                val_samples = int(total_samples * self.val_split_pct)
                test_samples = int(total_samples * self.test_split_pct)
                train_samples = total_samples - val_samples - test_samples
                
                generator = torch.Generator().manual_seed(42)
                _, _, self.test_dataset = random_split(
                    self.dataset,
                    [train_samples, val_samples, test_samples], 
                    generator=generator
                )

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return 2  # Binary flood segmentation

    @property 
    def num_channels(self) -> int:
        """Return the number of input channels."""
        return self.optical_channels + self.sar_channels
    
    # 供 Trainer.predict 使用：
    # - 默认沿用 test split（与评估口径一致）
    # - 当 predict_use_full=True 时，使用完整数据集进行推理（覆盖所有 tiles）
    def predict_dataloader(self) -> DataLoader:
        if getattr(self, "predict_use_full", False):
            # 完整数据集，无 train/val/test 切分
            if not hasattr(self, "predict_dataset") or self.predict_dataset is None:
                self.predict_dataset = self.dataset_class(**self.kwargs)
            dataset = self.predict_dataset
        else:
            if not hasattr(self, 'test_dataset') or self.test_dataset is None:
                self.setup('test')
            dataset = self.test_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
