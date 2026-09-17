"""CAU-Flood dataset following TorchGeo multi-modal best practices."""

import os
from collections.abc import Callable
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch import Tensor

# Import TorchGeo's standardization tools and base classes
try:
    from torchgeo.datasets import NonGeoDataset
    from torchgeo.transforms.functional import percentile_normalization
    BaseDataset = NonGeoDataset
    TORCHGEO_AVAILABLE = True
except ImportError:
    # Fallback for environments without TorchGeo
    from torch.utils.data import Dataset
    BaseDataset = Dataset
    # Provide a simple fallback for percentile_normalization
    def percentile_normalization(array, **kwargs):
        min_val, max_val = np.percentile(array, (2, 98))
        return np.clip((array - min_val) / (max_val - min_val), 0, 1)
    TORCHGEO_AVAILABLE = False


class CAUFlood(BaseDataset):
    """
    CAU-Flood dataset for flood change detection following TorchGeo best practices.
    
    🎯 IMPORTANT: This is a FLOOD CHANGE DETECTION dataset, NOT water body segmentation!
    
    Label Semantics:
    - 0 = Background (including permanent water bodies, land, etc.)
    - 1 = Newly flooded areas (flood-induced inundation areas only)
    
    Therefore, permanent rivers, lakes, and other water bodies are labeled as background (0),
    only newly flooded areas caused by flood events are labeled as foreground (1).
    
    SRP: Dataset 只负责数据加载与掩码二值化（0/1）。
    归一化与几何增强全部在 DataModule 中执行：
    - CAU-Flood 标准化使用 CMCDNet 官方统计（基于 8-bit PNG 的 [0,255] 分布），
      不在 Dataset 中进行任何百分位归一化；
    - DataModule 先进行几何增强（同步 image 与 mask），再按通道做 Z-Score 标准化。
    
    Band Organization (5 channels):
    - Band 0: Red (Pre-event optical)
    - Band 1: Green (Pre-event optical)  
    - Band 2: Blue (Pre-event optical)
    - Band 3: Near-Infrared/NIR (Pre-event optical)
    - Band 4: VV Polarization (Post-event SAR)
    
    注：为保持职责清晰与可配置性，任何归一化（含百分位/标准化）均不在 Dataset 内执行。
    
    The dataset structure should be:
    root/
    ├── train/
    │   ├── flood_vv/     # Ground truth flood masks
    │   │   ├── 00001.png # Binary: 0=non-flood, 1=flood
    │   │   └── ...
    │   ├── opt/          # Pre-event optical images
    │   │   ├── 00001.png # 4-channel: R, G, B, NIR
    │   │   └── ...
    │   └── vv/           # Post-event SAR images
    │       ├── 00001.png # 1-channel: VV polarization
    │       └── ...
    └── test/
        ├── flood_vv/     # Test set flood masks
        ├── opt/          # Test set optical images  
        └── vv/           # Test set SAR images
    
    Args:
        root: Root directory of the dataset.
        split: Dataset split to use ('train' or 'test').
        transforms: Optional transforms to apply to the samples.
    """
    
    classes = ['background', 'newly_flooded']  # 0=background(including permanent water), 1=newly flooded areas
    
    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        indices: list[int] | None = None,
        modal_type: str = "dual",  # "optical", "sar", "dual"
    ) -> None:
        """Initialize CAU-Flood dataset.
        
        Args:
            root: Root directory containing 'train' and 'test' folders.
            split: Dataset split to use ('train' or 'test').
            transforms: A function/transform that takes a sample and returns a
                transformed version.
            indices: Optional list of indices to use as a subset of the full dataset.
                If None, uses all samples in the split.
            modal_type: Type of modality to return ("optical", "sar", "dual")
        """
        super().__init__()
        
        self.root = root
        self.split = split
        self.transforms = transforms
        self.indices = indices
        self.modal_type = modal_type
        
        # 验证分割参数
        if split not in ['train', 'test']:
            raise ValueError(f"Split must be 'train' or 'test', got '{split}'")
        
        # 检查目录结构
        split_dir = os.path.join(self.root, self.split)
        self.optical_dir = os.path.join(split_dir, 'opt')
        self.sar_dir = os.path.join(split_dir, 'vv')
        self.mask_dir = os.path.join(split_dir, 'flood_vv')
        
        # 验证所有必需目录存在
        for dir_path, dir_name in [
            (self.optical_dir, f"{split}/opt"),
            (self.sar_dir, f"{split}/vv"),
            (self.mask_dir, f"{split}/flood_vv")
        ]:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        # 扫描所有样本，使用光学图像目录作为基准
        all_sample_ids = []
        for filename in sorted(os.listdir(self.optical_dir)):
            if filename.endswith('.png'):
                sample_id = filename[:-4]  # 移除.png扩展名
                all_sample_ids.append(sample_id)
        
        if not all_sample_ids:
            raise FileNotFoundError(f"No PNG files found in {self.optical_dir}")
        
        # 应用索引过滤（如果提供的话）
        if self.indices is not None:
            self.sample_ids = [all_sample_ids[i] for i in self.indices if i < len(all_sample_ids)]
        else:
            self.sample_ids = all_sample_ids
        
        # 验证所有样本都有对应的SAR和标注文件
        self._validate_files()
    
    def _validate_files(self) -> None:
        """验证所有样本文件都完整存在。"""
        missing_files = []
        
        for sample_id in self.sample_ids:
            # 检查SAR文件
            sar_path = os.path.join(self.sar_dir, f"{sample_id}.png")
            if not os.path.exists(sar_path):
                missing_files.append(f"SAR: {sample_id}.png")
            
            # 检查标注文件
            mask_path = os.path.join(self.mask_dir, f"{sample_id}.png")
            if not os.path.exists(mask_path):
                missing_files.append(f"Mask: {sample_id}.png")
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing files for split '{self.split}': {missing_files[:5]}"
                + (f" and {len(missing_files) - 5} more" if len(missing_files) > 5 else "")
            )
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.sample_ids)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return a sample from the dataset with proper multi-modal normalization.
        
        Uses TorchGeo percentile normalization to handle different dynamic ranges
        between optical (bands 0-3) and SAR (band 4) modalities.
        
        Args:
            index: Index of the sample to return.
            
        Returns:
            Dictionary containing:
                - image: Tensor of shape [C, H, W] with normalized values (5 channels)
                - mask: Tensor of shape [H, W] with binary flood labels
        """
        sample_id = self.sample_ids[index]
        
        # 构建文件路径
        optical_path = os.path.join(self.optical_dir, f"{sample_id}.png")
        sar_path = os.path.join(self.sar_dir, f"{sample_id}.png")
        mask_path = os.path.join(self.mask_dir, f"{sample_id}.png")
        
        # 加载光学图像 (4通道: R, G, B, NIR)
        optical_img = Image.open(optical_path)
        optical_array = np.array(optical_img)
        
        # 处理单通道图像
        if len(optical_array.shape) == 2:
            optical_array = np.stack([optical_array] * 3, axis=-1)
        
        # 加载SAR图像 (1通道: VV)
        sar_img = Image.open(sar_path)
        sar_array = np.array(sar_img)
        if len(sar_array.shape) == 3:
            sar_array = sar_array[:, :, 0]
        
        # 加载标注掩码并二值化
        mask_img = Image.open(mask_path)
        mask_array = np.array(mask_img)
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        
        # 二值化mask：0/255 → 0/1
        mask_array = (mask_array > 0).astype(np.uint8)
        
        # 转换数据类型
        optical_array = optical_array.astype(np.float32)
        sar_array = sar_array.astype(np.float32)
        
        # 转换为张量
        optical_tensor = torch.from_numpy(optical_array.transpose(2, 0, 1)).float()  # [4, H, W]
        sar_tensor = torch.from_numpy(sar_array).unsqueeze(0).float()  # [1, H, W]
        mask = torch.from_numpy(mask_array).long()  # [H, W]
        
        # 🎯 职责分离：Dataset只负责数据加载，标准化由DataModule处理
        
        # 根据模态类型返回不同格式
        if self.modal_type == "optical":
            sample = {"image": optical_tensor, "mask": mask}
        elif self.modal_type == "sar":
            sample = {"image": sar_tensor, "mask": mask}
        else:  # "dual"
            # 🎯 关键修复：双模态分离通道，与GF-FloodNet保持一致
            sample = {"image_optical": optical_tensor, "image_sar": sar_tensor, "mask": mask}
        
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample


