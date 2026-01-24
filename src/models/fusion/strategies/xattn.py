from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFusionStrategy
from ..blocks import DWResidualProjection, SpatialAttBlock, ChannelAttBlock, _norm2d


class XAttnFusionStrategy(BaseFusionStrategy):
    def __init__(self, optical_channels: List[int], sar_channels: List[int], proj_norm: str = "gn", reduction: int = 8, gamma_init: float = 0.0, apply_levels: List[int] = None,
                 align_bias: bool = False, align_bias_scale: float = 1.0):
        super().__init__(optical_channels, sar_channels)
        # Apply levels support negative indices (-1 deepest, -2 second-deepest). If empty, fall back to deepest only.
        self.apply_levels = set(apply_levels or [-1])
        # Resolve to actual indices 0..num_levels-1
        self.apply_levels_idx = set()
        for _idx in (apply_levels or [-1]):
            idx = int(_idx)
            if idx < 0:
                idx = self.num_levels + idx
            if 0 <= idx < self.num_levels:
                self.apply_levels_idx.add(idx)
        if len(self.apply_levels_idx) == 0 and self.num_levels > 0:
            self.apply_levels_idx.add(self.num_levels - 1)
        self.align_bias = bool(align_bias)
        self.align_bias_scale = float(align_bias_scale)
        # Q/K modulation variant removed.
        # Shared projection
        self.project_opt = nn.ModuleList([
            DWResidualProjection(in_ch, out_ch, norm=proj_norm)
            for in_ch, out_ch in zip(self.optical_channels, self.output_channels)
        ])
        self.project_sar = nn.ModuleList([
            DWResidualProjection(in_ch, out_ch, norm=proj_norm)
            for in_ch, out_ch in zip(self.sar_channels, self.output_channels)
        ])
        # Attention blocks
        self.blocks = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        self.gamma = nn.ParameterList()
        for i in range(self.num_levels):
            c = self.output_channels[i]
            att_ch = max(1, c // reduction)
            self.blocks.append(nn.ModuleDict({
                "spatial": SpatialAttBlock(c, att_ch, c,
                                             enable_align_bias=self.align_bias,
                                             align_bias_scale=self.align_bias_scale),
                "channel": ChannelAttBlock(c, att_ch, c),
                "out": nn.Sequential(nn.Conv2d(c, c, 1, bias=False), _norm2d(proj_norm, c), nn.ReLU(inplace=True)),
            }))
            self.res_proj.append(nn.Conv2d(self.optical_channels[i] + self.sar_channels[i], c, 1, bias=False))
            self.gamma.append(nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32)))
        # Runtime info

    def _calculate_output_channels(self) -> List[int]:
        return [max(opt, sar) for opt, sar in zip(self.optical_channels, self.sar_channels)]

    

    def forward(self, optical_features: List[torch.Tensor], sar_features: List[torch.Tensor]) -> List[torch.Tensor]:
        of, sf = optical_features, sar_features

        fused = []
        for i, (fo, fs) in enumerate(zip(of, sf)):
            if fo.shape[2:] != fs.shape[2:]:
                fs = F.interpolate(fs, size=fo.shape[2:], mode='bilinear', align_corners=False)
            fo_a = self.project_opt[i](fo)
            fs_a = self.project_sar[i](fs)
            if i not in self.apply_levels_idx:
                fused.append(fo_a + fs_a)
                continue
            b = self.blocks[i]
            sp_out = b["spatial"](fo_a, fs_a)
            ch_out = b["channel"](fo_a, fs_a)
            core = sp_out + ch_out
            out = b["out"](core)
            res = self.res_proj[i](torch.cat([fo, fs], dim=1))
            out = out + self.gamma[i] * res
            fused.append(out)
        return fused


