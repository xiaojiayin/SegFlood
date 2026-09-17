"""
特征融合模块：策略模式重构入口
"""

from .fusion import FeatureFusion
from .blocks import (
    ChannelAttBlock,
    DWResidualProjection,
    SpatialAttBlock,
    XATTN_AFFINITY_ORDERS,
    XATTN_ATTENTION_TYPES,
    XATTN_COMPONENTS,
)
from .strategies import (
    BaseFusionStrategy,
    ConcatFusionStrategy,
    SEFusionStrategy,
    CBAMFusionStrategy,
    AddFusionStrategy,
    GatedFusionStrategy,
    XAttnFusionStrategy,
    CanonicalAddFusionStrategy,
    CanonicalCrossAttentionFusionStrategy,
    CanonicalGatedFusionStrategy,
    IdentityFusionStrategy,
)

__all__ = [
    'FeatureFusion',
    'DWResidualProjection', 'SpatialAttBlock', 'ChannelAttBlock',
    'XATTN_COMPONENTS', 'XATTN_AFFINITY_ORDERS', 'XATTN_ATTENTION_TYPES',
    'BaseFusionStrategy', 'ConcatFusionStrategy', 'SEFusionStrategy', 'CBAMFusionStrategy',
    'AddFusionStrategy', 'GatedFusionStrategy', 'XAttnFusionStrategy',
    'CanonicalAddFusionStrategy', 'CanonicalCrossAttentionFusionStrategy',
    'CanonicalGatedFusionStrategy', 'IdentityFusionStrategy'
]
