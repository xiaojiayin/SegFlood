"""Sen1Floods11 DataModule following TorchGeo standards."""

import random
import logging
from typing import Any, Callable

import kornia.augmentation as K
import torch

logger = logging.getLogger(__name__)

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

from ..datasets.sen1floods11 import Sen1Floods11


class Sen1Floods11DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Sen1Floods11 dataset.
    
    This DataModule follows TorchGeo standards and best practices for multi-modal
    satellite data processing. It supports both hand-labeled and weakly-labeled data.
    
    Band Organization (15 channels total):
    SAR data (2 channels):
    - Band 0: VV polarization (dB)
    - Band 1: VH polarization (dB)
    
    Optical data (13 channels):
    - Band 2: B1 (Coastal, 443nm)
    - Band 3: B2 (Blue, 490nm)
    - Band 4: B3 (Green, 560nm)
    - Band 5: B4 (Red, 665nm)
    - Band 6: B5 (Red Edge 1, 705nm)
    - Band 7: B6 (Red Edge 2, 740nm)
    - Band 8: B7 (Red Edge 3, 783nm)
    - Band 9: B8 (NIR, 842nm)
    - Band 10: B8A (Narrow NIR, 865nm)
    - Band 11: B9 (Water Vapor, 945nm)
    - Band 12: B10 (Cirrus, 1375nm)
    - Band 13: B11 (SWIR 1, 1610nm)
    - Band 14: B12 (SWIR 2, 2190nm)
    
    Args:
        batch_size: Size of each mini-batch
        num_workers: Number of workers for parallel data loading
        use_weak_labels: Whether to include weakly-labeled data in training
        **kwargs: Additional keyword arguments passed to Sen1Floods11 dataset
    """

    # Sen1Floods11归一化参数
    # SAR数据通常在dB域，光学数据为TOA反射率乘以10000
    # 基于数据集特性的经验值，可能需要根据实际数据统计调整
    # 🎯 输出通道顺序: [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12, VV, VH]
    # 光学数据范围：0到10000 (TOA reflectance * 10000)
    # SAR数据范围大约：-30到+5 dB
    # 🎯 修正后的标准化参数（确保尺度一致）
    # 光学数据：TOA反射率×10000 → 归一化到合理范围
    # SAR数据：已预处理到[0,1] → 使用官方统计值
    mean = torch.tensor([
        # Optical (TOA reflectance * 10000 / 10000 = [0,1]) - B1到B12
        0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,  # B1-B7 (1500/10000=0.15)
        0.15, 0.15, 0.15, 0.15, 0.15, 0.15,        # B8-B12
        # SAR (官方统计值，预处理后[0,1]范围) - VV, VH
        0.6851, 0.5235  # VV, VH (官方Train.ipynb统计值)
    ])
    
    std = torch.tensor([
        # Optical (TOA reflectance * 10000 / 10000 = [0,1]) - B1到B12
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08,  # B1-B7 (800/10000=0.08)
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08,        # B8-B12
        # SAR (官方统计值，预处理后[0,1]范围) - VV, VH
        0.0820, 0.1102  # VV, VH (官方Train.ipynb统计值)
    ])

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        use_weak_labels: bool = False,
        # --- 接收变换对象，而不是自己创建 ---
        train_transforms: Callable[[dict], dict] | None = None,
        val_transforms: Callable[[dict], dict] | None = None,
        test_transforms: Callable[[dict], dict] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Sen1Floods11DataModule instance.
        
        Args:
            batch_size: Size of each mini-batch
            num_workers: Number of workers for parallel data loading
            use_weak_labels: Whether to include weakly-labeled data in training
            train_transforms: Transforms for training data (from config)
            val_transforms: Transforms for validation data (from config)
            test_transforms: Transforms for test data (from config)
            **kwargs: Additional keyword arguments passed to Sen1Floods11
        """
        # 提取通道配置参数，避免传递给数据集类
        self.optical_channels = kwargs.pop('optical_channels', 13)
        self.sar_channels = kwargs.pop('sar_channels', 2)
        
        super().__init__(
            dataset_class=Sen1Floods11,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )

        # 保存配置参数
        self.use_weak_labels = use_weak_labels
        
        logger.info(f"📊 Sen1Floods11DataModule: {self.num_channels}通道 (光学:{self.optical_channels} + SAR:{self.sar_channels})")

        # 🎯 分模态标准化参数（匹配数据集的分模态归一化）
        # 光学：百分位归一化后 [0,1]，SAR：官方方法归一化后 [0,1]
        mixed_mean = torch.tensor([
            # 光学通道：百分位归一化后的经验均值
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,  # 13个光学通道
            # SAR通道：官方统计值（预处理后[0,1]范围）
            0.6851, 0.5235  # VV, VH
        ])
        mixed_std = torch.tensor([
            # 光学通道：百分位归一化后的经验标准差
            0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,  # 13个光学通道
            # SAR通道：官方统计值（预处理后[0,1]范围）
            0.0820, 0.1102  # VV, VH
        ])
        
        self.base_transforms = K.AugmentationSequential(
            K.Normalize(mean=mixed_mean, std=mixed_std),
            data_keys=None,
            keepdim=True,
            extra_args={
                "mask": {"resample": "nearest", "align_corners": None}
            },
        )

        # 组合变换：基础变换 + 来自配置文件的可选变换
        self.train_aug = self._compose_transforms(train_transforms)
        self.val_aug = self._compose_transforms(val_transforms)
        self.test_aug = self._compose_transforms(test_transforms)
    

    
    def _generate_normalization_params(self, optical_channels: int, sar_channels: int):
        """简洁的标准化参数生成：基于通道数量动态截取
        
        Args:
            optical_channels: 光学通道数量
            sar_channels: SAR通道数量
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 均值和标准差张量
        """
        # 从预定义的15通道参数中选择对应数量的通道
        # Sen1Floods11数据顺序：[VV, VH, B1, B2, ..., B12]
        
        # SAR部分：取前sar_channels个SAR参数
        sar_mean = self.mean[:sar_channels]  # VV, VH
        sar_std = self.std[:sar_channels]
        
        # 光学部分：取前optical_channels个光学参数
        optical_mean = self.mean[2:2+optical_channels]  # B1, B2, ..., Bn
        optical_std = self.std[2:2+optical_channels]
        
        # 组合：SAR + 光学（保持Sen1Floods11原始数据顺序）
        combined_mean = torch.cat([sar_mean, optical_mean])
        combined_std = torch.cat([sar_std, optical_std])
        
        print(f"📊 标准化参数: SAR({sar_channels}) + 光学({optical_channels}) = {len(combined_mean)}通道")
        
        return combined_mean, combined_std

    def _compose_transforms(self, augs: Callable | None) -> Callable:
        """辅助函数，将基础变换与可选的增强变换组合起来。
        
        采用GitHub Copilot建议的方法，避免标准化重复和AugmentationSequential嵌套问题。
        """
        def has_normalize(seq):
            """检查增强序列中是否已包含标准化变换"""
            if isinstance(seq, K.AugmentationSequential):
                for module in seq.children():
                    if isinstance(module, K.Normalize):
                        return True
            return False

        transform_list = []
        
        # 只有在增强中没有标准化时才添加标准化
        if not (augs and has_normalize(augs)):
            # 使用分模态标准化参数
            mixed_mean = torch.tensor([
                # 光学：百分位归一化后
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                # SAR：官方统计值
                0.6851, 0.5235
            ])
            mixed_std = torch.tensor([
                # 光学：百分位归一化后
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                # SAR：官方统计值
                0.0820, 0.1102
            ])
            transform_list.append(K.Normalize(mean=mixed_mean, std=mixed_std))

        if augs is not None:
            if isinstance(augs, K.AugmentationSequential):
                # 展开AugmentationSequential，避免嵌套
                for module in augs.children():
                    transform_list.append(module)
            else:
                # 单个变换直接添加
                transform_list.append(augs)
                
            return K.AugmentationSequential(
                *transform_list,
                data_keys=None,
                keepdim=True,
                extra_args={
                    "mask": {"resample": "nearest", "align_corners": None}
                },
            )
        
        # 如果没有增强，返回只包含标准化的基础变换
        return self.base_transforms

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """TorchGeo数据增强应用点 - Kornia变换在此处执行"""
        return super().on_after_batch_transfer(batch, dataloader_idx)

    def setup(self, stage: str) -> None:
        """Set up datasets for different stages.
        
        Sen1Floods11 dataset: 使用官方预定义的train/val/test分割。
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage in ['fit', 'validate']:
            # 创建训练数据集（可选择包含弱标注）
            self.train_dataset = self.dataset_class(
                split='train', 
                use_weak_labels=self.use_weak_labels,

                **self.kwargs
            )
            
            # 创建验证数据集（只使用手工标注）
            self.val_dataset = self.dataset_class(
                split='val', 
                use_weak_labels=False,  # 验证时不使用弱标注

                **self.kwargs
            )
            
            weak_info = f" (+{len([s for s in self.train_dataset.samples if s['is_weak']])} weak)" if self.use_weak_labels else ""
            print(f"📊 Sen1Floods11数据划分:")
            print(f"   训练样本: {len(self.train_dataset)}{weak_info}")
            print(f"   验证样本: {len(self.val_dataset)}")
            print(f"   弱标注: {'启用' if self.use_weak_labels else '禁用'}")
            
        if stage in ['test']:
            # 创建独立的test数据集用于最终评估
            self.test_dataset = self.dataset_class(
                split='test', 
                use_weak_labels=False,  # 测试时不使用弱标注

                **self.kwargs
            )
            print(f"📊 Sen1Floods11测试集: {len(self.test_dataset)} 样本")

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return 2  # Binary flood segmentation

    @property 
    def num_channels(self) -> int:
        """Return the number of input channels."""
        return self.optical_channels + self.sar_channels
