"""
解码器模块（Decoder）：Reductions驱动的动态U-Net解码器

职责（Single Responsibility）：
- 仅负责：接收融合后的金字塔特征，构建自顶向下的解码路径并输出语义分割logits

核心思想：
- Reductions-driven：依据实际降采样倍率与特征层数动态生成上采样链
- 尺寸精对齐：使用插值对齐到 skip 的空间尺寸，最终插值回输入尺寸

最佳实践：
- feature_channels 由“融合模块输出通道”提供（非编码器原始通道），保证通道一致性
- bilinear 上采样更加通用可靠；需要严格2x上采样时可切换 convtrans
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def conv_block(in_ch: int, out_ch: int, dropout: float = 0.0) -> nn.Module:
    """标准卷积块：Conv-BN-ReLU-Dropout-Conv-BN-ReLU"""
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
    """分割头：将特征映射到最终类别数"""
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
    Reductions驱动的动态U-Net解码器
    
    核心创新：
    - 根据实际feature_channels长度自动生成解码路径
    - 依据真实空间尺寸做插值对齐（不依赖固定2×倍率）
    - 支持任意骨干金字塔：CNN、ConvNeXt、ViT、SAM2等
    - 最后精确插值回input_size，确保输出与标签尺寸一致
    
    优势：
    - CNN标准金字塔：[4,8,16,32] ✅
    - MobileNet多层：[2,4,8,16,32] ✅ 
    - ViT单尺度：[16] ✅
    - SAM2/Hiera层级：[4,8,16,32] ✅
    """
    
    def __init__(
        self,
        feature_channels: List[int],             # 融合后的每层通道（浅->深）
        num_classes: int = 2,
        dropout: float = 0.1,
        feature_reductions: Optional[List[int]] = None,  # 用于日志记录
        up_mode: str = "bilinear",               # 'bilinear' 或 'convtrans'
        final_upsample_to_input: bool = True,    # 最终插值到原图大小
    ):
        super().__init__()
        assert up_mode in ("bilinear", "convtrans"), f"up_mode必须是bilinear或convtrans，得到{up_mode}"
        assert len(feature_channels) >= 1, "feature_channels至少包含一层特征"
        
        self.num_classes = num_classes
        self.reductions = feature_reductions or []
        self.up_mode = up_mode
        self.final_upsample_to_input = final_upsample_to_input

        L = len(feature_channels)
        self.decoder_channels = feature_channels.copy()
        self.up_blocks = nn.ModuleList()

        # 从深到浅构建路径（L层特征 => L-1个上采样块）
        for i in range(L - 1, 0, -1):
            in_ch = self.decoder_channels[i]        # 当前层通道
            out_ch = self.decoder_channels[i - 1]   # 目标层通道（更浅）
            
            if self.up_mode == "convtrans":
                # 转置卷积上采样（固定2倍，适用于标准金字塔）
                up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
                merge_in = out_ch + self.decoder_channels[i - 1]
            else:
                # 双线性插值上采样（适应任意尺寸，推荐）
                up = nn.Identity()  # 插值在forward中动态进行
                merge_in = in_ch + self.decoder_channels[i - 1]

            self.up_blocks.append(
                nn.ModuleDict({
                    'up': up, 
                    'conv': conv_block(merge_in, out_ch, dropout)
                })
            )

        # 分割头
        self.seg_head = SegmentationHead(self.decoder_channels[0], num_classes, dropout=dropout)

        logger.info(f"UNet解码器初始化完成:")
        logger.info(f"  特征通道: {feature_channels}")
        logger.info(f"  降采样倍率: {feature_reductions}")
        logger.info(f"  上采样模式: {up_mode}")
        logger.info(f"  最终插值到输入: {final_upsample_to_input}")
        logger.info(f"  输出类别: {num_classes}")

    def forward(self, features: List[torch.Tensor], input_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        前向推理
        
        Args:
            features: 浅->深的特征列表（融合后）
            input_size: (H, W) 原图尺寸，用于最终精确插值
            
        Returns:
            logits: [B, num_classes, H, W] 分割logits
        """
        assert isinstance(features, (list, tuple)) and len(features) >= 1
        
        # 从最深层开始
        x = features[-1]
        L = len(features)

        # 自顶向下的U-Net解码路径
        for k, blk in enumerate(self.up_blocks):
            i = L - 1 - k  # 当前深层索引
            skip = features[i - 1]  # skip connection特征
            
            if self.up_mode == "convtrans":
                # 转置卷积上采样（固定2倍）
                x = blk["up"](x)
            else:
                # 双线性插值上采样（适应真实尺寸）
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Skip connection + 卷积块
            x = torch.cat([x, skip], dim=1)
            x = blk["conv"](x)

        # 分割头
        logits = self.seg_head(x)

        # 最终精确插值到原图尺寸
        if self.final_upsample_to_input and input_size is not None and logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
            
        return logits