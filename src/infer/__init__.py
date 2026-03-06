"""Inference utilities: dataset-specific predict writers and helpers."""

from .cauflood import CAUFloodPredictWriter
from .gffloodnet import GFFloodNetPredictWriter
from .s1s2_water import S1S2WaterPredictWriter
from .worldfloodsv2 import WorldFloodsPredictWriter

__all__ = [
    "CAUFloodPredictWriter",
    "GFFloodNetPredictWriter",
    "S1S2WaterPredictWriter",
    "WorldFloodsPredictWriter",
]
