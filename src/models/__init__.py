"""
模型模块：重构后的模块化多模态分割框架

新的架构设计：
1. encoders/: 统一的编码器模块（TorchGeo + timm）
2. fusion/: 中期融合策略（concat, add, attention）
3. decoders/: 多种解码器架构（UNet等）
4. lightning_module.py: 简化的Lightning训练模块

使用示例：
```python
from src.models import DualStreamEncoder, FeatureFusion, UNetDecoder, MultiModalSegmentationModule

# 创建组件
encoder = DualStreamEncoder(
    model_name="resnet50",
    optical_channels=4,
    sar_channels=1,
    optical_weights="SENTINEL2_ALL_MOCO",
    sar_weights="SENTINEL1_GRD_MOCO"
)

fusion = FeatureFusion(
    feature_channels=encoder.feature_channels,
    fusion_type="concat"
)

decoder = UNetDecoder(
    feature_channels=encoder.feature_channels,
    num_classes=2,
    fusion_type="concat"
)

# 创建Lightning模块
model = MultiModalSegmentationModule(
    encoder=encoder,
    fusion=fusion, 
    decoder=decoder,
    loss_fn=loss_fn
)
```
"""

# 编码器相关
from .encoders import DualStreamEncoder, create_encoder

# 融合策略
from .fusion import FeatureFusion

# 解码器
from .decoders import UNetDecoder, SegmentationHead

# Lightning模块
from .lightning_module import MultiModalSegmentationModule

__all__ = [
    # 编码器
    'DualStreamEncoder', 'create_encoder',
    
    # 融合策略
    'FeatureFusion',
    
    # 解码器
    'UNetDecoder', 'SegmentationHead',
    
    # Lightning模块
    'MultiModalSegmentationModule'
]
