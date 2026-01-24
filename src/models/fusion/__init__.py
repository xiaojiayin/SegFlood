"""Fusion package exports."""

from .fusion import FeatureFusion
from .blocks import DWResidualProjection, SpatialAttBlock, ChannelAttBlock
from .strategies import (
    BaseFusionStrategy,
    ConcatFusionStrategy,
    AddFusionStrategy,
    XAttnFusionStrategy,
)

__all__ = [
    "FeatureFusion",
    "DWResidualProjection",
    "SpatialAttBlock",
    "ChannelAttBlock",
    "BaseFusionStrategy",
    "ConcatFusionStrategy",
    "AddFusionStrategy",
    "XAttnFusionStrategy",
]
