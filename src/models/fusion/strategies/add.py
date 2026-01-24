from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFusionStrategy
from ..blocks import DWResidualProjection


class AddFusionStrategy(BaseFusionStrategy):
    def __init__(self, optical_channels: List[int], sar_channels: List[int], proj_norm: str = "gn"):
        super().__init__(optical_channels, sar_channels)
        self.project_opt = nn.ModuleList([
            DWResidualProjection(in_ch, out_ch, norm=proj_norm)
            for in_ch, out_ch in zip(self.optical_channels, self.output_channels)
        ])
        self.project_sar = nn.ModuleList([
            DWResidualProjection(in_ch, out_ch, norm=proj_norm)
            for in_ch, out_ch in zip(self.sar_channels, self.output_channels)
        ])

    def _calculate_output_channels(self) -> List[int]:
        return [max(opt, sar) for opt, sar in zip(self.optical_channels, self.sar_channels)]

    def forward(self, optical_features: List[torch.Tensor], sar_features: List[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        for i, (fo, fs) in enumerate(zip(optical_features, sar_features)):
            if fo.shape[2:] != fs.shape[2:]:
                fs = F.interpolate(fs, size=fo.shape[2:], mode='bilinear', align_corners=False)
            fo_aligned = self.project_opt[i](fo)
            fs_aligned = self.project_sar[i](fs)
            fused.append(fo_aligned + fs_aligned)
        return fused


