"""GF-FloodNet dataset following TorchGeo multi-modal best practices."""

import glob
import os
from collections.abc import Callable
from typing import Any, Dict

import numpy as np
import rasterio as rio
import torch
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


class GFFloodNet(BaseDataset):
    """
    GF-FloodNet dataset for flood area semantic segmentation following TorchGeo best practices.
    
    📋 Label Semantics (Based on original paper Section 2.3):
    - Pixel value 1: Flood area (all water bodies including permanent + flood water)
    - Pixel value 255: Non-flood area (background: land, buildings, etc.)
    
    Task Type: Semantic Segmentation (flood area extraction)
    Goal: Segment all water bodies in post-disaster imagery, including both 
          permanent water bodies and newly flooded areas.
    
    SRP: Dataset 只负责数据加载与掩码二值化（0/1）。
    归一化与几何增强全部在 DataModule 中执行，以保持训练/验证/测试路径一致：
    - GF-FloodNet 流程为：先对图像做逐样本逐通道的 2-98 百分位缩放至 [0,1]，
      再按照官方统计做 Z-Score 标准化（均在 DataModule 中完成）。
    
    Band Organization (5 channels):
    - Band 0: Blue (GF-2 optical)
    - Band 1: Green (GF-2 optical)  
    - Band 2: Red (GF-2 optical)
    - Band 3: Near-Infrared (NIR, GF-2 optical)
    - Band 4: SAR Intensity (GF-3 C-band)
    
    注：为保持职责清晰与可配置性，任何归一化（含百分位/标准化）均不在 Dataset 内执行。
    
    The dataset structure should be:
    root/
    ├── images/
    │   ├── image_001.tif (5-channel: Blue, Green, Red, NIR, SAR)
    │   ├── image_002.tif
    │   └── ...
    └── annotations/
        ├── label_001.tif (Binary: 0=non-flood, 1=flood)
        ├── label_002.tif
        └── ...
    
    Args:
        root: Root directory of the dataset.
        transforms: Optional transforms to apply to the samples.
    """
    
    classes = ['non-flood', 'flood']  # 0=background, 1=flood/water areas (all water bodies)
    
    def __init__(
        self,
        root: str = 'data',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        modal_type: str = "dual",  # "optical", "sar", "dual"
    ) -> None:
        """Initialize GF-FloodNet dataset.
        
        Args:
            root: Root directory containing 'images' and 'annotations' folders.
            transforms: A function/transform that takes a sample and returns a
                transformed version.
            modal_type: Type of modality to return ("optical", "sar", "dual")
        """
        super().__init__()
        
        self.root = root
        self.transforms = transforms
        self.modal_type = modal_type
        
        # 设置目录路径
        images_dir = os.path.join(self.root, 'images')
        annotations_dir = os.path.join(self.root, 'annotations')
        
        # 扫描图像文件
        image_pattern = os.path.join(images_dir, '*.tif')
        self.files = sorted(glob.glob(image_pattern))
    

    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.files)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a sample based on modal_type configuration."""
        image_path = self.files[index]
        mask_path = image_path.replace('images', 'annotations')
        
        # Load full 5-channel image and mask
        image = torch.from_numpy(rio.open(image_path).read().astype(np.float32))
        mask = torch.from_numpy(rio.open(mask_path).read(1).astype(np.int64))
        
        # Convert mask: [1, 255] → [1, 0] 
        # 修正：1=洪水, 255=背景 (2025-08-26 修复)
        mask = (mask == 1).long()
        
        # 🎯 职责分离：Dataset只负责数据加载，标准化由DataModule处理
        # 移除percentile normalization，改为原始数据输出
        
        # Select channels based on modal_type
        if self.modal_type == "optical":
            # Channels 0-3: Blue, Green, Red, NIR
            selected_image = image[:4]
            sample = {"image": selected_image, "mask": mask}
        elif self.modal_type == "sar":
            # Channel 4: SAR
            selected_image = image[4:5]
            sample = {"image": selected_image, "mask": mask}
        else:  # "dual"
            # 🎯 关键修复：双模态分离通道，与CAU-Flood保持一致
            optical_image = image[:4]
            sar_image = image[4:5]
            sample = {"image_optical": optical_image, "image_sar": sar_image, "mask": mask}
        
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample

