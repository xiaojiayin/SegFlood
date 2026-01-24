"""Feature fusion (strategy pattern) for multi-modal / single-modal feature pyramids."""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .strategies import (
    ConcatFusionStrategy,
    AddFusionStrategy,
    XAttnFusionStrategy,
)

logger = logging.getLogger(__name__)


class FeatureFusion(nn.Module):
    """
    Fusion manager (strategy pattern).
    """
    def __init__(
        self,
        feature_channels: List[int],
        fusion_type: str = "concat",
        optical_channels: Optional[List[int]] = None,
        sar_channels: Optional[List[int]] = None,
        # xattn config
        xattn_apply_levels: Optional[List[int]] = None,
        xattn_reduction: int = 8,
        xattn_gamma_init: float = 0.0,
        # alignment guidance (alignment bias)
        xattn_align_bias: bool = False,
        xattn_align_bias_scale: float = 1.0,
        # projection normalization
        projection_norm: str = "gn",
        # DropBranch removed
    ):
        super().__init__()
        assert fusion_type in ("concat", "add", "xattn"), f"Unsupported fusion_type: {fusion_type}"

        # Metadata
        self.feature_channels = list(feature_channels)
        self.fusion_type = fusion_type
        self.num_levels = len(self.feature_channels)
        
        

        # Per-stream channels (from encoder)
        self.optical_channels = list(optical_channels) if optical_channels else None
        self.sar_channels = list(sar_channels) if sar_channels else None

        # Align list lengths (avoid silent truncation by zip).
        if self.optical_channels and len(self.optical_channels) != self.num_levels:
            N = min(self.num_levels, len(self.optical_channels))
            logger.warning(
                f"optical_channels length ({len(self.optical_channels)}) != feature_channels ({self.num_levels}); "
                f"truncate to {N}"
            )
            self.feature_channels = self.feature_channels[:N]
            self.optical_channels = self.optical_channels[:N]
            if self.sar_channels:
                self.sar_channels = self.sar_channels[:N]
            self.num_levels = N
        if self.sar_channels and len(self.sar_channels) != self.num_levels:
            N = min(self.num_levels, len(self.sar_channels))
            logger.warning(
                f"sar_channels length ({len(self.sar_channels)}) != feature_channels ({self.num_levels}); "
                f"truncate to {N}"
            )
            self.feature_channels = self.feature_channels[:N]
            if self.optical_channels:
                self.optical_channels = self.optical_channels[:N]
            self.sar_channels = self.sar_channels[:N]
            self.num_levels = N

        # Build strategy
        self.strategy = self._build_strategy(
            fusion_type,
            {
                "xattn_apply_levels": xattn_apply_levels,
                "xattn_reduction": xattn_reduction,
                "xattn_gamma_init": xattn_gamma_init,
                # alignment guidance
                "xattn_align_bias": xattn_align_bias,
                "xattn_align_bias_scale": xattn_align_bias_scale,
                "proj_norm": projection_norm,
            },
        )

        # Strategy decides output_channels
        self.output_channels = self.strategy.output_channels

        # Single-modal projection (concat uses concatenation width: opt+sar).
        if fusion_type == "concat":
            # If per-stream channels are missing, fall back to feature_channels.
            in_opt = self.optical_channels or self.feature_channels
            in_sar = self.sar_channels or self.feature_channels
            # Output width equals strategy output_channels (opt+sar).
            self.single_modal_proj_opt = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_opt[i], self.output_channels[i], 1, bias=False),
                    nn.BatchNorm2d(self.output_channels[i]),
                    nn.ReLU(inplace=True),
                )
                for i in range(self.num_levels)
            ])
            self.single_modal_proj_sar = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_sar[i], self.output_channels[i], 1, bias=False),
                    nn.BatchNorm2d(self.output_channels[i]),
                    nn.ReLU(inplace=True),
                )
                for i in range(self.num_levels)
            ])
        else:
            self.single_modal_proj_opt = None
            self.single_modal_proj_sar = None

        # Alias: xattn -> MA-XAttn
        self.alias = "MA-XAttn" if fusion_type == "xattn" else fusion_type
        logger.info(f"Feature fusion initialized: {self.alias}")
        if self.optical_channels and self.sar_channels:
            logger.info(f"  per-stream: optical={self.optical_channels}, sar={self.sar_channels}")
        logger.info(f"  output_channels={self.output_channels}")
        # Modality-dropout is removed from the fusion manager to keep behavior explicit.

    def _build_strategy(self, fusion_type: str, cfg: Dict[str, Any]):
        # If per-stream channels are missing, fall back to feature_channels.
        opt_ch = self.optical_channels or self.feature_channels
        sar_ch = self.sar_channels or self.feature_channels
        if fusion_type == "concat":
            return ConcatFusionStrategy(opt_ch, sar_ch)
        if fusion_type == "add":
            return AddFusionStrategy(opt_ch, sar_ch, proj_norm=cfg.get("proj_norm", "gn"))
        if fusion_type == "xattn":
            return XAttnFusionStrategy(
                opt_ch,
                sar_ch,
                proj_norm=cfg.get("proj_norm", "gn"),
                reduction=cfg.get("xattn_reduction", 8),
                gamma_init=cfg.get("xattn_gamma_init", 0.0),
                apply_levels=cfg.get("xattn_apply_levels") or [-1],
                align_bias=cfg.get("xattn_align_bias", False),
                align_bias_scale=cfg.get("xattn_align_bias_scale", 1.0),
            )
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

    # ------- Compatibility no-op (kept for older call sites) -------
    def set_epoch(self, epoch: int):
        return

    # ------- Forward -------
    def forward(self, optical_features: Optional[List[torch.Tensor]] = None, sar_features: Optional[List[torch.Tensor]] = None) -> List[torch.Tensor]:
        if optical_features is None and sar_features is None:
            raise ValueError("At least one modality features must be provided.")
        if optical_features is None:
            assert sar_features is not None
            return self._single_modal_forward(sar_features, modality="sar")
        if sar_features is None:
            assert optical_features is not None
            return self._single_modal_forward(optical_features, modality="optical")

        # Dual-modal: spatially align then delegate to strategy.
        for i in range(len(optical_features)):
            fo, fs = optical_features[i], sar_features[i]
            if fo.shape[2:] != fs.shape[2:]:
                sar_features[i] = F.interpolate(fs, size=fo.shape[2:], mode='bilinear', align_corners=False)
        out = self.strategy(optical_features, sar_features)
        return out

    

    def _single_modal_forward(self, feats: List[torch.Tensor], modality: str) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        if self.fusion_type == "concat":
            proj_list = self.single_modal_proj_opt if modality == "optical" else self.single_modal_proj_sar
            for i, f in enumerate(feats):
                outs.append(proj_list[i](f))
            return outs
        else:
            # add / xattn: project to max(opt, sar) width used by the strategy.
            proj_list = self.strategy.project_opt if modality == "optical" else self.strategy.project_sar
            for i, f in enumerate(feats):
                outs.append(proj_list[i](f))
            return outs