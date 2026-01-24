"""Decoder components (dynamic U-Net) driven by feature pyramid reductions."""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def conv_block(in_ch: int, out_ch: int, dropout: float = 0.0) -> nn.Module:
    """Standard conv block: Conv-BN-ReLU-(Dropout)-Conv-BN-ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SegmentationHead(nn.Sequential):
    """Segmentation head: map features to class logits."""
    def __init__(self, in_channels: int, num_classes: int, kernel_size: int = 3, dropout: float = 0.1):
        mid = max(16, in_channels // 2)
        super().__init__(
            nn.Conv2d(in_channels, mid, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(mid, num_classes, kernel_size=1),
        )


class UNetDecoder(nn.Module):
    """
    Dynamic U-Net decoder for arbitrary feature pyramids.

    - Builds the top-down path based on the provided feature pyramid length.
    - Uses interpolation to align spatial sizes (does not assume strict 2x scales).
    - Optionally upsamples logits back to input_size.
    """
    
    def __init__(
        self,
        feature_channels: List[int],             # fused channels per level (shallow -> deep)
        num_classes: int = 2,
        dropout: float = 0.1,
        feature_reductions: Optional[List[int]] = None,  # for logging/debugging only
        up_mode: str = "bilinear",               # 'bilinear' or 'convtrans'
        final_upsample_to_input: bool = True,    # upsample logits to input_size
    ):
        super().__init__()
        assert up_mode in ("bilinear", "convtrans"), f"up_mode must be 'bilinear' or 'convtrans', got {up_mode}"
        assert len(feature_channels) >= 1, "feature_channels must contain at least one level"
        
        self.num_classes = num_classes
        self.reductions = feature_reductions or []
        self.up_mode = up_mode
        self.final_upsample_to_input = final_upsample_to_input

        L = len(feature_channels)
        self.decoder_channels = feature_channels.copy()
        self.up_blocks = nn.ModuleList()

        # Build path from deep to shallow (L levels -> L-1 upsampling blocks).
        for i in range(L - 1, 0, -1):
            in_ch = self.decoder_channels[i]
            out_ch = self.decoder_channels[i - 1]
            
            if self.up_mode == "convtrans":
                # Transposed conv upsampling (fixed 2x).
                up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
                merge_in = out_ch + self.decoder_channels[i - 1]
            else:
                # Bilinear interpolation upsampling (recommended).
                up = nn.Identity()  # interpolation happens in forward()
                merge_in = in_ch + self.decoder_channels[i - 1]

            self.up_blocks.append(
                nn.ModuleDict({
                    'up': up, 
                    'conv': conv_block(merge_in, out_ch, dropout)
                })
            )

        # Segmentation head
        self.seg_head = SegmentationHead(self.decoder_channels[0], num_classes, dropout=dropout)

        logger.info("UNetDecoder initialized")
        logger.info(f"  feature_channels={feature_channels}")
        logger.info(f"  feature_reductions={feature_reductions}")
        logger.info(f"  up_mode={up_mode}")
        logger.info(f"  final_upsample_to_input={final_upsample_to_input}")
        logger.info(f"  num_classes={num_classes}")

    def forward(self, features: List[torch.Tensor], input_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward.

        Args:
            features: fused pyramid features (shallow -> deep)
            input_size: (H, W) input size for final upsampling

        Returns:
            logits: [B, num_classes, H, W]
        """
        assert isinstance(features, (list, tuple)) and len(features) >= 1
        
        # Start from the deepest level.
        x = features[-1]
        L = len(features)

        # Top-down decoding path.
        for k, blk in enumerate(self.up_blocks):
            i = L - 1 - k
            skip = features[i - 1]
            
            if self.up_mode == "convtrans":
                # Fixed 2x upsampling.
                x = blk["up"](x)
            else:
                # Interpolate to match skip spatial size.
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Skip connection + conv block
            x = torch.cat([x, skip], dim=1)
            x = blk["conv"](x)

        # Head
        logits = self.seg_head(x)

        # Final upsample to input size (optional).
        if self.final_upsample_to_input and input_size is not None and logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
            
        return logits