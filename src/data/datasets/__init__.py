"""Dataset implementations for each flood/water dataset."""

from .gf_floodnet import GFFloodNetDataset
from .cau_flood import CAUFloodDataset
from .kurosiwo import KuroSiwoDataset
from .worldfloodsv2 import WorldFloodsv2Dataset, CHANNELS_CONFIGURATIONS
from .s1s2_water import S1S2WaterDataset

__all__ = [
    "GFFloodNetDataset",
    "CAUFloodDataset",
    "KuroSiwoDataset",
    "WorldFloodsv2Dataset",
    "CHANNELS_CONFIGURATIONS",
    "S1S2WaterDataset",
]
