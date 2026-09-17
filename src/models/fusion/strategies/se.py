from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFusionStrategy


class SEFusionStrategy(BaseFusionStrategy):
    def __init__(self, optical_channels: List[int], sar_channels: List[int], reduction: int = 16, gamma_init: float = 0.0, apply_levels: List[int] = None):
        super().__init__(optical_channels, sar_channels)
        self.apply_levels = set(apply_levels or [-1])
        self.blocks = nn.ModuleList()
        self.gamma = nn.ParameterList()
        for i in range(self.num_levels):
            att_in = self.optical_channels[i] + self.sar_channels[i]
            red = max(1, att_in // reduction)
            self.blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(att_in, red, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(red, att_in, 1, bias=False),
                nn.Sigmoid(),
            ))
            self.gamma.append(nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32)))

    def _calculate_output_channels(self) -> List[int]:
        return [opt + sar for opt, sar in zip(self.optical_channels, self.sar_channels)]

    def forward(self, optical_features: List[torch.Tensor], sar_features: List[torch.Tensor]) -> List[torch.Tensor]:
        fused = []
        for i, (fo, fs) in enumerate(zip(optical_features, sar_features)):
            if fo.shape[2:] != fs.shape[2:]:
                fs = F.interpolate(fs, size=fo.shape[2:], mode='bilinear', align_corners=False)
            x_cat = torch.cat([fo, fs], dim=1)
            if i not in self.apply_levels:
                fused.append(x_cat)
                continue
            w = self.blocks[i](x_cat)
            out = x_cat * w
            out = out + self.gamma[i] * x_cat
            fused.append(out)
        return fused


