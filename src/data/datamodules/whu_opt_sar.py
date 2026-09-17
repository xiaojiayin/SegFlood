"""WHU-OPT-SAR DataModule following TorchGeo standards.

Based on detailed dataset analysis:
- 100 samples total, resolution 5556×3704 pixels  
- Optical: 4-channel RGBA, uint8, range [1,254]
- SAR: Single-channel, uint8, range [1,248]
- Labels: 8 classes {0,10,20,30,40,50,60,70} → indices {0,1,2,3,4,5,6,7}
- Class distribution: forest(40.7%), farmland(40.4%), water(6.4%), village(5.7%), city(3.8%), others(1.9%), road(1.0%), background(0.0%)

Follows the pattern used by CAUFloodDataModule with adaptations for:
- Multi-class segmentation (8 classes vs 2)
- RGBA+SAR channels (5 total: R,G,B,A,SAR)
- Larger image resolution (5556×3704 vs 256×256)
"""

from __future__ import annotations

import random
from typing import Any, Callable, Tuple

import torch
import kornia.augmentation as K
from torchgeo.datamodules import NonGeoDataModule

from ..datasets.whu_opt_sar import WHUOptSAR


class WHUOptSARDataModule(NonGeoDataModule):
    """LightningDataModule for WHU-OPT-SAR multi-class segmentation.
    
    Dataset characteristics based on analysis:
    - 100 samples, 5556×3704 pixels each
    - 5 channels: [R, G, B, Alpha, SAR] 
    - 8 classes: background, farmland, city, village, water, forest, road, others
    - Imbalanced distribution: forest(40.7%) and farmland(40.4%) dominate
    """

    # Channel statistics for [R, G, B, Alpha, SAR] based on analysis
    # Optical RGBA channels: mean ~[99.5, 91.8, 79.5, 90.6], range [1,254]
    # SAR channel: range [1,248], estimated mean ~124
    # Scaled to [0,255] range for consistency with existing framework
    mean = torch.tensor([99.5, 91.8, 79.5, 90.6, 124.0])  # [R, G, B, A, SAR]
    std = torch.tensor([60.0, 60.0, 60.0, 60.0, 60.0])    # Conservative std estimates

    def __init__(
        self,
        batch_size: int = 8,  # 增加批次大小，适应512×512裁剪
        num_workers: int = 4,
        train_transforms: Callable[[dict], dict] | None = None,
        val_transforms: Callable[[dict], dict] | None = None,
        test_transforms: Callable[[dict], dict] | None = None,
        val_split: float = 0.2,  # 80:20 split (80 train, 20 val) for 100 samples
        random_seed: int = 42,
        # --- cropping ---
        crop_size: Tuple[int, int] | None = (512, 512),
        val_crop_size: Tuple[int, int] | None = (512, 512),
        # 每图训练采样次数（提升batch内方差）
        train_samples_per_image: int = 1,
        # 测试滑窗
        test_tile_size: Tuple[int, int] | None = (512, 512),
        test_tile_stride: Tuple[int, int] | None = (384, 384),
        **kwargs: Any,
    ) -> None:
        """Initialize WHU-OPT-SAR DataModule.
        
        Args:
            batch_size: 8 for 512×512 cropped images
            val_split: 0.2 for 80:20 split (recommended for 100 samples)
            **kwargs: Passed to WHUOptSAR dataset
        """
        # 保存原始参数
        self.val_split = val_split
        self.random_seed = random_seed
        self.crop_size = crop_size
        self.val_crop_size = val_crop_size
        self.train_samples_per_image = train_samples_per_image
        self.test_tile_size = test_tile_size
        self.test_tile_stride = test_tile_stride
        
        # 提取通道配置参数，避免传递给数据集类
        self.optical_channels = kwargs.pop('optical_channels', 4)
        self.sar_channels = kwargs.pop('sar_channels', 1)
        
        # 🚨 重要：保存来自Hydra的transforms（将在on_after_batch_transfer中应用）
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        
        # 🐛 调试：打印接收到的transforms
        print(f"🔍 DataModule收到的transforms:")
        print(f"   train_transforms: {type(train_transforms)} - {train_transforms}")
        print(f"   val_transforms: {type(val_transforms)} - {val_transforms}")
        if train_transforms is not None:
            print(f"   train_transforms类型: {type(train_transforms)}")
            if hasattr(train_transforms, 'children'):
                print(f"   train_transforms子模块: {list(train_transforms.children())}")
        
        # 创建基础标准化变换（仅标准化，不包含数据增强）
        self.base_transforms = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            data_keys=None,
            keepdim=True,
        )
        
        # 初始化基类，不传递transforms（我们将在on_after_batch_transfer中手动应用）
        super().__init__(
            dataset_class=WHUOptSAR,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """🎯 关键：在此处应用Kornia数据增强
        
        根据GitHub Copilot建议，这是应用batch-level transforms的正确位置
        """
        # � 调试：打印初始batch信息
        print(f"🔍 on_after_batch_transfer 调用:")
        print(f"   训练模式: {self.trainer.training}")
        print(f"   batch keys: {list(batch.keys())}")
        if "image" in batch:
            print(f"   原始image形状: {batch['image'].shape}")
            print(f"   原始mask形状: {batch['mask'].shape}")
        
        # �🚨 关键：转换batch格式从{"image": ..., "mask": ...} 到 {"input": ..., "mask": ...}
        # 这是Kornia AugmentationSequential期望的格式
        if "image" in batch:
            batch = {"input": batch["image"], "mask": batch["mask"]}
            print(f"   转换后batch keys: {list(batch.keys())}")
        
        # 应用数据增强
        original_shape = batch["input"].shape if "input" in batch else "unknown"
        
        if self.trainer.training and self.train_transforms is not None:
            print(f"   🎯 应用训练transforms: {type(self.train_transforms)}")
            try:
                batch = self.train_transforms(batch)
                print(f"   ✅ 训练transforms应用成功!")
                if "input" in batch:
                    print(f"   训练后image形状: {batch['input'].shape}")
                    print(f"   训练后mask形状: {batch['mask'].shape}")
            except Exception as e:
                print(f"   ❌ 训练transforms应用失败: {e}")
                import traceback
                traceback.print_exc()
        elif not self.trainer.training and self.val_transforms is not None:
            print(f"   🎯 应用验证transforms: {type(self.val_transforms)}")
            try:
                batch = self.val_transforms(batch)
                print(f"   ✅ 验证transforms应用成功!")
                if "input" in batch:
                    print(f"   验证后image形状: {batch['input'].shape}")
                    print(f"   验证后mask形状: {batch['mask'].shape}")
            except Exception as e:
                print(f"   ❌ 验证transforms应用失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"   ⚠️  没有可应用的transforms")
            print(f"      训练模式: {self.trainer.training}")
            print(f"      train_transforms is None: {self.train_transforms is None}")
            print(f"      val_transforms is None: {self.val_transforms is None}")
        
        # 🚨 重要：转换回原始格式 {"input": ..., "mask": ...} → {"image": ..., "mask": ...}  
        # 因为Lightning期望"image"键名
        if "input" in batch:
            batch = {"image": batch["input"], "mask": batch["mask"]}
            print(f"   转换回image格式，最终形状: image={batch['image'].shape}, mask={batch['mask'].shape}")
        
        # 应用基础标准化（总是应用）
        if "image" in batch:
            batch_for_norm = {"input": batch["image"], "mask": batch["mask"]}
            try:
                batch_for_norm = self.base_transforms(batch_for_norm)
                batch = {"image": batch_for_norm["input"], "mask": batch_for_norm["mask"]}
                print(f"   ✅ 基础标准化应用成功!")
            except Exception as e:
                print(f"   ❌ 基础标准化失败: {e}")
        
        print(f"   🏁 最终输出形状: image={batch['image'].shape}, mask={batch['mask'].shape}")
        return batch

    def setup(self, stage: str) -> None:
        if stage in ["fit", "validate"]:
            full = self.dataset_class(**self.kwargs)
            n = len(full)
            val_n = int(n * self.val_split)
            train_n = n - val_n
            random.seed(self.random_seed)
            idxs = list(range(n))
            random.shuffle(idxs)
            train_idx = idxs[:train_n]
            val_idx = idxs[train_n:]
            # 训练集：随机裁剪
            self.train_dataset = self.dataset_class(
                indices=train_idx,
                crop_size=self.crop_size,
                crop_type="random",
                samples_per_image=self.train_samples_per_image,
                **self.kwargs,
            )
            # 验证集：中心裁剪
            self.val_dataset = self.dataset_class(
                indices=val_idx,
                crop_size=self.val_crop_size,
                crop_type="center",
                **self.kwargs,
            )
            print(f"📊 WHU-OPT-SAR数据划分 (100样本总计):")
            print(f"   训练样本: {len(self.train_dataset)} ({train_n}/{n})")
            print(f"   验证样本: {len(self.val_dataset)} ({val_n}/{n})")
            print(f"   划分比例: {train_n}:{val_n} ({(1-self.val_split)*100:.0f}:{self.val_split*100:.0f})")
            print(f"   随机种子: {self.random_seed}")
        if stage in ["test"]:
            # 测试集：中心裁剪
            self.test_dataset = self.dataset_class(
                # 使用滑窗测试
                crop_size=None,
                crop_type=None,
                tile_size=self.test_tile_size,
                tile_stride=self.test_tile_stride,
                **self.kwargs,
            )
            print(f"📊 WHU-OPT-SAR测试集: {len(self.test_dataset)} 样本")

    @property
    def num_classes(self) -> int:
        return 8

    @property
    def num_channels(self) -> int:
        return self.optical_channels + self.sar_channels
