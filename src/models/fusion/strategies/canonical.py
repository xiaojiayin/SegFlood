"""Canonical fusion baselines without MA-XAttn-specific enhancements."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import SpatialAttBlock
from .base import BaseFusionStrategy


def _linear_or_identity(in_channels: int, out_channels: int) -> nn.Module:
    if in_channels == out_channels:
        return nn.Identity()
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


class IdentityFusionStrategy(BaseFusionStrategy):
    """Single-stream passthrough used by input-level early fusion."""

    def _calculate_output_channels(self) -> List[int]:
        return [
            max(optical, sar)
            for optical, sar in zip(
                self.optical_channels, self.sar_channels
            )
        ]

    def forward(
        self,
        optical_features: List[torch.Tensor],
        sar_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        raise ValueError("identity fusion 仅适用于单编码器输入")


class CanonicalAddFusionStrategy(BaseFusionStrategy):
    """Direct dual-stream addition with only required 1x1 projections."""

    def __init__(
        self, optical_channels: List[int], sar_channels: List[int]
    ) -> None:
        super().__init__(optical_channels, sar_channels)
        self.project_opt = nn.ModuleList(
            [
                _linear_or_identity(in_ch, out_ch)
                for in_ch, out_ch in zip(
                    self.optical_channels, self.output_channels
                )
            ]
        )
        self.project_sar = nn.ModuleList(
            [
                _linear_or_identity(in_ch, out_ch)
                for in_ch, out_ch in zip(
                    self.sar_channels, self.output_channels
                )
            ]
        )

    def _calculate_output_channels(self) -> List[int]:
        return [
            max(optical, sar)
            for optical, sar in zip(
                self.optical_channels, self.sar_channels
            )
        ]

    def forward(
        self,
        optical_features: List[torch.Tensor],
        sar_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for index, (optical, sar) in enumerate(
            zip(optical_features, sar_features)
        ):
            if optical.shape[2:] != sar.shape[2:]:
                sar = F.interpolate(
                    sar,
                    size=optical.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            outputs.append(
                self.project_opt[index](optical)
                + self.project_sar[index](sar)
            )
        return outputs


class CanonicalGatedFusionStrategy(CanonicalAddFusionStrategy):
    """Standard sigmoid modality gate without MA residual branches."""

    def __init__(
        self, optical_channels: List[int], sar_channels: List[int]
    ) -> None:
        super().__init__(optical_channels, sar_channels)
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        2 * out_channels,
                        out_channels,
                        kernel_size=1,
                        bias=True,
                    ),
                    nn.Sigmoid(),
                )
                for out_channels in self.output_channels
            ]
        )

    def forward(
        self,
        optical_features: List[torch.Tensor],
        sar_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for index, (optical, sar) in enumerate(
            zip(optical_features, sar_features)
        ):
            if optical.shape[2:] != sar.shape[2:]:
                sar = F.interpolate(
                    sar,
                    size=optical.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            projected_optical = self.project_opt[index](optical)
            projected_sar = self.project_sar[index](sar)
            gate = self.gates[index](
                torch.cat([projected_optical, projected_sar], dim=1)
            )
            outputs.append(
                gate * projected_optical
                + (1.0 - gate) * projected_sar
            )
        return outputs


class CanonicalCrossAttentionFusionStrategy(CanonicalAddFusionStrategy):
    """Scaled bidirectional cross-attention without MA outer residuals."""

    def __init__(
        self,
        optical_channels: List[int],
        sar_channels: List[int],
        reduction: int = 8,
        apply_levels: Optional[List[int]] = None,
    ) -> None:
        if reduction <= 0:
            raise ValueError("reduction 必须是正整数")
        super().__init__(optical_channels, sar_channels)
        self.apply_levels = set()
        for raw_index in apply_levels or [-1]:
            index = int(raw_index)
            if index < 0:
                index += self.num_levels
            if 0 <= index < self.num_levels:
                self.apply_levels.add(index)
        self.blocks = nn.ModuleList(
            [
                SpatialAttBlock(
                    out_channels,
                    max(1, out_channels // reduction),
                    out_channels,
                    attention_type="standard",
                )
                for out_channels in self.output_channels
            ]
        )

    def forward(
        self,
        optical_features: List[torch.Tensor],
        sar_features: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for index, (optical, sar) in enumerate(
            zip(optical_features, sar_features)
        ):
            if optical.shape[2:] != sar.shape[2:]:
                sar = F.interpolate(
                    sar,
                    size=optical.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
            projected_optical = self.project_opt[index](optical)
            projected_sar = self.project_sar[index](sar)
            if index in self.apply_levels:
                outputs.append(
                    self.blocks[index](projected_optical, projected_sar)
                )
            else:
                outputs.append(projected_optical + projected_sar)
        return outputs
