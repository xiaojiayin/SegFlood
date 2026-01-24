from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFusionStrategy


class ConcatFusionStrategy(BaseFusionStrategy):
    def _calculate_output_channels(self) -> List[int]:
        return [opt + sar for opt, sar in zip(self.optical_channels, self.sar_channels)]

    def forward(self, optical_features: List[torch.Tensor], sar_features: List[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        for fo, fs in zip(optical_features, sar_features):
            if fo.shape[2:] != fs.shape[2:]:
                fs = F.interpolate(fs, size=fo.shape[2:], mode='bilinear', align_corners=False)
            fused.append(torch.cat([fo, fs], dim=1))
        return fused


