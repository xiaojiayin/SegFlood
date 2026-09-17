from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFusionStrategy
from ..blocks import DWResidualProjection


class GatedFusionStrategy(BaseFusionStrategy):
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
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2 * out_ch, out_ch, 1, bias=False), nn.Sigmoid())
            for out_ch in self.output_channels
        ])
        # 残差缩放
        self.res_proj = nn.ModuleList([
            nn.Conv2d(self.optical_channels[i] + self.sar_channels[i], self.output_channels[i], 1, bias=False)
            for i in range(self.num_levels)
        ])
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, dtype=torch.float32)) for _ in range(self.num_levels)
        ])

    def _calculate_output_channels(self) -> List[int]:
        return [max(opt, sar) for opt, sar in zip(self.optical_channels, self.sar_channels)]

    def forward(self, optical_features: List[torch.Tensor], sar_features: List[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        for i, (fo, fs) in enumerate(zip(optical_features, sar_features)):
            if fo.shape[2:] != fs.shape[2:]:
                fs = F.interpolate(fs, size=fo.shape[2:], mode='bilinear', align_corners=False)
            fo_a = self.project_opt[i](fo)
            fs_a = self.project_sar[i](fs)
            g = self.gates[i](torch.cat([fo_a, fs_a], dim=1))
            out = fo_a * g + fs_a * (1.0 - g)
            res = self.res_proj[i](torch.cat([fo, fs], dim=1))
            out = out + self.gamma[i] * res
            fused.append(out)
        return fused


