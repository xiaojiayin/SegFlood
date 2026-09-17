"""CAU-Flood DataModule following TorchGeo standards."""

import random
from typing import Any, Callable

import torch
import kornia.augmentation as K
from torch.utils.data import DataLoader

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

from ..datasets.cau_flood import CAUFlood


class CAUFloodDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the CAU-Flood dataset.
    
    🎯 IMPORTANT: CAU-Flood is a FLOOD CHANGE DETECTION dataset!
    
    Label Semantics (Critical Understanding):
    - 0 = Background (including permanent water bodies, land, buildings, etc.)
    - 1 = Newly flooded areas (flood-induced inundation areas only)
    
    This means permanent rivers, lakes, and other pre-existing water bodies 
    are labeled as background (0), NOT as flood (1). Only areas that became 
    newly flooded due to the flood event are labeled as foreground (1).
    
    This DataModule follows TorchGeo standards and best practices for satellite
    data processing. It includes proper normalization, data augmentation, and
    follows the NonGeoDataModule pattern.
    
    Band Organization (5 channels):
    - Band 0: Red (Pre-event optical)
    - Band 1: Green (Pre-event optical)  
    - Band 2: Blue (Pre-event optical)
    - Band 3: Near-Infrared/NIR (Pre-event optical)
    - Band 4: VV Polarization (Post-event SAR)
    
    Args:
        batch_size: Size of each mini-batch
        num_workers: Number of workers for parallel data loading
        **kwargs: Additional keyword arguments passed to CAUFlood dataset
    """

    # 默认归一化参数 - 可通过配置文件覆盖
    # 通道顺序: [Red, Green, Blue, NIR, VV] - 灾前光学(RGBNIR) + 灾后SAR(VV)
    DEFAULT_MEAN = [123.675, 116.28, 103.53, 123.675, 120.0]  # [R, G, B, NIR, VV]
    DEFAULT_STD = [58.395, 57.12, 57.375, 58.395, 60.0]       # CMCDNet标准方差

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        # --- 🎯 新增：模态选择支持 ---
        modal_type: str = "dual",  # "optical", "sar", "dual"
        # --- 🎯 新增：早期融合支持 ---
        output_image_key: bool = False,  # True时强制输出单键image（早期融合baseline）
        # --- 核心修改：接收变换对象，而不是自己创建 ---
        train_transforms: Callable[[dict], dict] | None = None,
        val_transforms: Callable[[dict], dict] | None = None,
        test_transforms: Callable[[dict], dict] | None = None,
        # --- 新增：数据划分参数 ---
        val_split: float = 0.125,  # 1/(7+1) = 0.125，即7:1的比例
        random_seed: int = 42,
        # --- 新增：可配置的标准化参数 ---
        normalization_mean: list[float] | None = None,
        normalization_std: list[float] | None = None,
        # --- 评估阶段缺模态测试：dual | optical_only | sar_only ---
        eval_modal_mode: str = "dual",
        **kwargs: Any,
    ) -> None:
        """Initialize a new CAUFloodDataModule instance.
        
        Args:
            batch_size: Size of each mini-batch
            num_workers: Number of workers for parallel data loading
            modal_type: Type of modality ("optical", "sar", "dual")
            output_image_key: If True, always output single 'image' key (for early fusion baseline)
            train_transforms: Transforms for training data (from config)
            val_transforms: Transforms for validation data (from config)
            test_transforms: Transforms for test data (from config)
            val_split: Fraction of train data to use for validation (default: 0.125 for 7:1 ratio)
            random_seed: Random seed for reproducible train/val splits
            normalization_mean: Mean values for normalization (defaults to CMCDNet standard)
            normalization_std: Std values for normalization (defaults to CMCDNet standard)
            **kwargs: Additional keyword arguments passed to CAUFlood
        """
        self.modal_type = modal_type
        self.output_image_key = output_image_key
        if eval_modal_mode not in {"dual", "optical_only", "sar_only"}:
            raise ValueError("eval_modal_mode must be one of: dual, optical_only, sar_only")
        self.eval_modal_mode = eval_modal_mode
        
        # 提取通道配置参数，避免传递给数据集类
        self.optical_channels = kwargs.pop('optical_channels', 4)
        self.sar_channels = kwargs.pop('sar_channels', 1)
        
        # 🎯 向数据集传递modal_type参数
        kwargs['modal_type'] = modal_type
        
        super().__init__(
            dataset_class=CAUFlood,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )

        # 保存划分参数
        self.val_split = val_split
        self.random_seed = random_seed
        
        # 🎯 配置标准化参数 - 支持从配置文件覆盖
        if normalization_mean is not None:
            self.mean = torch.tensor(normalization_mean)
        else:
            self.mean = torch.tensor(self.DEFAULT_MEAN)
            
        if normalization_std is not None:
            self.std = torch.tensor(normalization_std)
        else:
            self.std = torch.tensor(self.DEFAULT_STD)
        
        # 根据modal_type调整标准化参数
        if self.modal_type == "optical":
            self.mean = self.mean[:4]  # 只取前4个通道
            self.std = self.std[:4]
        elif self.modal_type == "sar":
            self.mean = self.mean[4:5]  # 只取第5个通道
            self.std = self.std[4:5]
        # dual模式使用全部5个通道

        # 🎯 架构分离：标准化和增强在on_after_batch_transfer中分步处理
        self.train_aug = train_transforms
        self.val_aug = val_transforms  
        self.test_aug = test_transforms
        self.predict_aug = None  # 预测时通常不增强
    
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

        # 1. 根据stage选择几何变换
        transform = None
        if stage == 'train':
            transform = self.train_aug
        elif stage == 'val':
            transform = self.val_aug
        elif stage == 'test':
            transform = self.test_aug
        elif stage == 'predict':
            transform = self.predict_aug
        
        # 2. 应用几何变换
        if transform:
            try:
                image, mask = transform(image, mask)
                # 修复Kornia可能添加的额外维度
                if len(mask.shape) == 4 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)
            except Exception as e:
                raise RuntimeError(
                    "Kornia AugmentationSequential 配置错误：请确保 data_keys=['input','mask'] "
                    f"并传入 (image, mask)。原始错误: {e}"
                )

        # 3. 应用标准化
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
        
        CAU-Flood dataset: 从train split按7:1比例划分出train/val，保留test split用于最终测试。
        
        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage in ['fit', 'validate']:
            # 创建完整的train数据集以获取样本总数
            full_train_dataset = self.dataset_class(split='train', **self.kwargs)
            total_samples = len(full_train_dataset)
            
            # 计算验证集样本数
            val_size = int(total_samples * self.val_split)
            train_size = total_samples - val_size
            
            # 设置随机种子确保可重复性
            random.seed(self.random_seed)
            all_indices = list(range(total_samples))
            random.shuffle(all_indices)
            
            # 划分索引
            train_indices = all_indices[:train_size]
            val_indices = all_indices[train_size:]
            
            # 创建train和validation数据集，使用索引子集
            self.train_dataset = self.dataset_class(split='train', indices=train_indices, **self.kwargs)
            self.val_dataset = self.dataset_class(split='train', indices=val_indices, **self.kwargs)
            
            print(f"📊 CAU-Flood数据划分:")
            print(f"   训练样本: {len(self.train_dataset)} ({train_size}/{total_samples})")
            print(f"   验证样本: {len(self.val_dataset)} ({val_size}/{total_samples})")
            # 计算简化的比例显示
            train_ratio = round((1-self.val_split) / self.val_split)
            print(f"   划分比例: {train_ratio}:1 (训练:{1-self.val_split:.1%}, 验证:{self.val_split:.1%})")
            
        if stage in ['test']:
            # 创建独立的test数据集用于最终评估
            self.test_dataset = self.dataset_class(split='test', **self.kwargs)

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return 2  # Binary flood change detection: 0=background(incl. permanent water), 1=newly flooded

    @property 
    def num_channels(self) -> int:
        """Return the number of input channels based on modal_type."""
        if self.modal_type == "optical":
            return self.optical_channels  # 4 channels
        elif self.modal_type == "sar":
            return self.sar_channels  # 1 channel
        else:  # "dual"
            return self.optical_channels + self.sar_channels  # 5 channels
    
    # 供 Trainer.predict 使用：沿用 test 数据加载器，确保与评估口径一致
    def predict_dataloader(self) -> DataLoader:
        if not hasattr(self, 'test_dataset') or self.test_dataset is None:
            self.setup('test')
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
