"""Sen1Floods11 dataset following TorchGeo multi-modal best practices."""

import os
import pandas as pd
import logging
from collections.abc import Callable
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torch import Tensor

logger = logging.getLogger(__name__)

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


class Sen1Floods11(BaseDataset):
    """
    Sen1Floods11 dataset for flood segmentation following TorchGeo best practices.
    
    This dataset combines Sentinel-1 SAR data and Sentinel-2 optical data for multi-modal
    flood detection. It supports both hand-labeled and weakly-labeled data splits.
    
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
    
    The dataset structure should be:
    root/
    └── v1.1/
        ├── data/
        │   └── flood_events/
        │       ├── HandLabeled/         # 446 hand-labeled samples
        │       │   ├── S1Hand/          # Sentinel-1 SAR data
        │       │   ├── S2Hand/          # Sentinel-2 optical data
        │       │   └── LabelHand/       # Hand-labeled flood masks
        │       └── WeaklyLabeled/       # 4,384 weakly-labeled samples
        │           ├── S1Weak/          # Sentinel-1 SAR data
        │           ├── S2Weak/          # Sentinel-2 optical data (if available)
        │           └── S1OtsuLabelWeak/ # Otsu-threshold labels
        └── splits/
            └── flood_handlabeled/
                ├── flood_train_data.csv     # 252 samples
                ├── flood_valid_data.csv     # 89 samples
                └── flood_test_data.csv      # 90 samples
    
    Args:
        root: Root directory of the dataset.
        split: Dataset split to use ('train', 'val', 'test').
        use_weak_labels: Whether to include weakly-labeled data (only for 'train').
        transforms: Optional transforms to apply to the samples.
    """
    
    classes = ['non-flood', 'flood']
    
    def __init__(
        self,
        root: str = 'data/Sen1Floods11',
        split: str = 'train',
        use_weak_labels: bool = False,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        indices: list[int] | None = None,
    ) -> None:
        """Initialize Sen1Floods11 dataset.
        
        Args:
            root: Root directory containing Sen1Floods11 v1.1 data.
            split: Dataset split to use ('train', 'val', 'test').
            use_weak_labels: Whether to include weakly-labeled data (only for 'train').
            transforms: A function/transform that takes a sample and returns a
                transformed version.
            indices: Optional list of indices to use as a subset of the full dataset.
                If None, uses all samples in the split.
                
        Note:
            输出通道顺序为：[光学B1-B12, SAR_VV, SAR_VH] (15通道)
            与其他数据集保持一致（光学在前，SAR在后）
        """
        # 调用基类的构造函数（如果可用的话）
        if hasattr(super(), '__init__'):
            try:
                super().__init__()
            except TypeError:
                # 如果基类不接受参数，就直接调用
                pass
        
        self.root = root
        self.split = split
        self.use_weak_labels = use_weak_labels
        self.transforms = transforms
        self.indices = indices
        

        
        # 验证分割参数
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got '{split}'")
        
        # 弱标注只在训练时可用
        if use_weak_labels and split != 'train':
            raise ValueError("Weak labels are only available for training split")
        
        # 设置数据路径
        self.data_dir = os.path.join(self.root, 'v1.1', 'data', 'flood_events')
        self.splits_dir = os.path.join(self.root, 'v1.1', 'splits', 'flood_handlabeled')
        
        # 检查目录结构
        self._validate_directories()
        
        # 加载数据样本
        self.samples = self._load_samples()
        
        # 应用索引过滤（如果提供的话）
        if self.indices is not None:
            self.samples = [self.samples[i] for i in self.indices if i < len(self.samples)]
    

    def _validate_directories(self) -> None:
        """验证所有必需目录存在。"""
        required_dirs = [
            self.data_dir,
            self.splits_dir,
            os.path.join(self.data_dir, 'HandLabeled', 'S1Hand'),
            os.path.join(self.data_dir, 'HandLabeled', 'S2Hand'),
            os.path.join(self.data_dir, 'HandLabeled', 'LabelHand'),
        ]
        
        if self.use_weak_labels:
            required_dirs.extend([
                os.path.join(self.data_dir, 'WeaklyLabeled', 'S1Weak'),
                os.path.join(self.data_dir, 'WeaklyLabeled', 'S1OtsuLabelWeak'),
            ])
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    def _load_samples(self) -> list[dict]:
        """加载样本列表。"""
        samples = []
        
        # 加载手工标注数据
        hand_labeled_samples = self._load_hand_labeled_samples()
        samples.extend(hand_labeled_samples)
        
        # 如果需要，加载弱标注数据
        if self.use_weak_labels:
            weak_labeled_samples = self._load_weak_labeled_samples()
            samples.extend(weak_labeled_samples)
        
        return samples
    
    def _load_hand_labeled_samples(self) -> list[dict]:
        """加载手工标注样本。"""
        # 根据split选择对应的CSV文件
        split_map = {
            'train': 'flood_train_data.csv',
            'val': 'flood_valid_data.csv', 
            'test': 'flood_test_data.csv'
        }
        
        csv_file = os.path.join(self.splits_dir, split_map[self.split])
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Split file not found: {csv_file}")
        
        # 读取CSV文件（没有header）
        df = pd.read_csv(csv_file, header=None, names=['s1_file', 'label_file'])
        
        samples = []
        for _, row in df.iterrows():
            s1_filename = row['s1_file']
            label_filename = row['label_file']
            
            # 从文件名中提取sample_id（去掉后缀）
            sample_id = s1_filename.replace('_S1Hand.tif', '')
            
            # 构建文件路径
            s1_path = os.path.join(self.data_dir, 'HandLabeled', 'S1Hand', s1_filename)
            s2_path = os.path.join(self.data_dir, 'HandLabeled', 'S2Hand', f"{sample_id}_S2Hand.tif")
            label_path = os.path.join(self.data_dir, 'HandLabeled', 'LabelHand', label_filename)
            
            # 验证文件存在
            for path in [s1_path, s2_path, label_path]:
                if not os.path.exists(path):
                    print(f"Warning: File not found: {path}")
                    continue
            
            samples.append({
                'sample_id': sample_id,
                's1_path': s1_path,
                's2_path': s2_path,
                'label_path': label_path,
                'is_weak': False
            })
        
        return samples
    
    def _load_weak_labeled_samples(self) -> list[dict]:
        """加载弱标注样本。"""
        weak_s1_dir = os.path.join(self.data_dir, 'WeaklyLabeled', 'S1Weak')
        weak_label_dir = os.path.join(self.data_dir, 'WeaklyLabeled', 'S1OtsuLabelWeak')
        
        samples = []
        for filename in sorted(os.listdir(weak_s1_dir)):
            if filename.endswith('_S1Weak.tif'):
                sample_id = filename.replace('_S1Weak.tif', '')
                
                s1_path = os.path.join(weak_s1_dir, filename)
                label_path = os.path.join(weak_label_dir, f"{sample_id}_S1OtsuLabelWeak.tif")
                
                # 弱标注数据可能没有S2数据
                s2_path = None
                weak_s2_dir = os.path.join(self.data_dir, 'WeaklyLabeled', 'S2Weak')
                if os.path.exists(weak_s2_dir):
                    s2_candidate = os.path.join(weak_s2_dir, f"{sample_id}_S2Weak.tif")
                    if os.path.exists(s2_candidate):
                        s2_path = s2_candidate
                
                if os.path.exists(s1_path) and os.path.exists(label_path):
                    samples.append({
                        'sample_id': sample_id,
                        's1_path': s1_path,
                        's2_path': s2_path,
                        'label_path': label_path,
                        'is_weak': True
                    })
        
        return samples
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Return a sample from the dataset with proper multi-modal normalization.
        
        Args:
            index: Index of the sample to return.
            
        Returns:
            Dictionary containing:
                - image: Tensor of shape [C, H, W] with selected channels
                - mask: Tensor of shape [H, W] with binary flood labels
        """
        sample_info = self.samples[index]
        
        # 加载SAR数据 (2通道: VV, VH)
        s1_data = self._load_geotiff(sample_info['s1_path'])  # [2, H, W]
        s1_data = np.nan_to_num(s1_data, nan=-50.0)  # 处理NaN值
        
        # 加载光学数据 (13通道: B1-B12)
        if sample_info['s2_path'] is not None:
            s2_data = self._load_geotiff(sample_info['s2_path'])  # [13, H, W]
            s2_data = np.nan_to_num(s2_data, nan=0.0)  # 处理NaN值
        else:
            # 如果没有S2数据，创建零填充
            s2_data = np.zeros((13, s1_data.shape[1], s1_data.shape[2]), dtype=np.float32)
        
        # 加载标签数据
        label_data = self._load_geotiff(sample_info['label_path'])  # [1, H, W]
        if len(label_data.shape) == 3:
            label_data = label_data[0]  # 移除通道维度 -> [H, W]
        
        # 组合原始多模态数据 [SAR(2) + Optical(13) = 15 channels]
        original_data = np.concatenate([s1_data, s2_data], axis=0)  # [15, H, W]
        
        # 🎯 分模态归一化（充分利用两种数据的特性）
        
        # SAR数据：使用官方验证的预处理方法
        s1_data = np.clip(s1_data, -50, 1)  # 裁剪到[-50, 1]范围
        s1_data = (s1_data + 50) / 51       # 归一化到[0, 1]
        s1_tensor = torch.from_numpy(s1_data).float()
        
        # 光学数据：使用TorchGeo百分位归一化（对多光谱数据更鲁棒）
        s2_tensor = torch.from_numpy(s2_data).float()
        try:
            # TorchGeo percentile normalization for optical data
            s2_normalized = percentile_normalization(s2_tensor, min_percentile=2, max_percentile=98)
        except NameError:
            # Fallback: Manual per-channel percentile normalization
            s2_normalized = torch.zeros_like(s2_tensor)
            for c in range(s2_tensor.shape[0]):
                channel = s2_tensor[c]
                p2, p98 = torch.quantile(channel.flatten(), torch.tensor([0.02, 0.98]))
                if p98 > p2:
                    s2_normalized[c] = torch.clamp((channel - p2) / (p98 - p2), 0, 1)
                else:
                    s2_normalized[c] = torch.zeros_like(channel)
        
        # 重排序通道：[光学, SAR] (与其他数据集保持一致)
        image = torch.cat([s2_normalized, s1_tensor], dim=0)  # [15, H, W]
        mask = torch.from_numpy(label_data).long()  # [H, W]
        
        # 数据清理：确保数值稳定性
        image = torch.nan_to_num(image, nan=0.0)
        image = torch.clamp(image, min=-10.0, max=10.0)
        mask = torch.clamp(mask, min=-1, max=1)
        
        # 创建样本字典（与其他数据集保持一致的简洁格式）
        sample = {"image": image, "mask": mask}
        
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample
    
    def _load_geotiff(self, path: str) -> np.ndarray:
        """加载GeoTIFF文件。"""
        try:
            import rasterio
            with rasterio.open(path) as src:
                data = src.read()  # [C, H, W]
                return data.astype(np.float32)
        except ImportError:
            # 如果没有rasterio，尝试用PIL加载（可能有限制）
            img = Image.open(path)
            data = np.array(img)
            
            # 确保数据是3维的
            if len(data.shape) == 2:
                data = data[np.newaxis, :, :]  # [1, H, W]
            elif len(data.shape) == 3 and data.shape[-1] > data.shape[0]:
                data = data.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
            return data.astype(np.float32)

    def plot(self, sample: dict[str, Tensor], show_titles: bool = True, suptitle: str | None = None) -> None:
        """Plot a sample from the dataset.
        
        Args:
            sample: A sample dict containing 'image' and 'mask'.
            show_titles: Whether to show subplot titles.
            suptitle: Super title for the entire figure.
        """
        try:
            import matplotlib.pyplot as plt
            
            image = sample['image']
            mask = sample['mask']
            
            # 如果是torch tensor，转换为numpy
            if hasattr(image, 'numpy'):
                image = image.numpy()
            if hasattr(mask, 'numpy'):
                mask = mask.numpy()
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # 显示SAR VV波段
            axes[0, 0].imshow(image[0], cmap='gray')
            if show_titles:
                axes[0, 0].set_title('SAR VV (dB)')
            axes[0, 0].axis('off')
            
            # 显示SAR VH波段
            axes[0, 1].imshow(image[1], cmap='gray')
            if show_titles:
                axes[0, 1].set_title('SAR VH (dB)')
            axes[0, 1].axis('off')
            
            # 显示RGB合成图像 (S2 B4-B3-B2: 红-绿-蓝)
            if image.shape[0] >= 15:  # 确保有S2数据
                rgb_indices = [5, 4, 3]  # B4(红), B3(绿), B2(蓝) in combined array
                rgb_image = image[rgb_indices].transpose(1, 2, 0)  # (H, W, 3)
                # 标准化到0-1范围
                rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())
                rgb_image = np.clip(rgb_image, 0, 1)
                axes[0, 2].imshow(rgb_image)
                if show_titles:
                    axes[0, 2].set_title('S2 RGB Composite')
            else:
                axes[0, 2].text(0.5, 0.5, 'No S2 Data', ha='center', va='center')
                if show_titles:
                    axes[0, 2].set_title('S2 RGB (N/A)')
            axes[0, 2].axis('off')
            
            # 显示洪水掩码
            axes[1, 0].imshow(mask, cmap='RdYlBu', vmin=0, vmax=1)
            if show_titles:
                axes[1, 0].set_title('Flood Mask')
            axes[1, 0].axis('off')
            
            # 显示SAR VV + 掩码叠加
            sar_norm = (image[0] - image[0].min()) / (image[0].max() - image[0].min())
            sar_rgb = np.stack([sar_norm, sar_norm, sar_norm], axis=-1)
            flood_pixels = mask == 1
            sar_rgb[flood_pixels] = [1, 0, 0]  # 红色表示洪水
            axes[1, 1].imshow(sar_rgb)
            if show_titles:
                axes[1, 1].set_title('SAR VV + Flood Overlay')
            axes[1, 1].axis('off')
            
            # 显示RGB + 掩码叠加（如果有S2数据）
            if image.shape[0] >= 15:
                rgb_with_mask = rgb_image.copy()
                rgb_with_mask[flood_pixels] = [1, 0, 0]  # 红色表示洪水
                axes[1, 2].imshow(rgb_with_mask)
                if show_titles:
                    axes[1, 2].set_title('RGB + Flood Overlay')
            else:
                axes[1, 2].text(0.5, 0.5, 'No S2 Data', ha='center', va='center')
                if show_titles:
                    axes[1, 2].set_title('RGB Overlay (N/A)')
            axes[1, 2].axis('off')
            
            if suptitle:
                fig.suptitle(suptitle)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting.")
