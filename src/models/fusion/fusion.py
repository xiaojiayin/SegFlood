"""
融合模块（Fusion）：多模态/单模态特征的统一融合接口（策略模式）

职责
- 作为“管理者”：实例化并委托具体融合策略（concat / se / cbam / add / gated / xattn）
- 处理单/双模态路由与空间对齐
- 维护输出通道元信息，供下游解码器使用

约定（与历史一致）
- concat / se / cbam：输出通道 = opt + sar（拼接）
- add / gated / xattn：先用统一投影对齐到 max(opt, sar) 再交互
- 单模态：
  - concat / se / cbam：单模态投影到等价双模态宽度（与历史一致）
  - add / gated / xattn：单模态投影到 max 宽度（与双模态一致）
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .strategies import (
    ConcatFusionStrategy,
    SEFusionStrategy,
    CBAMFusionStrategy,
    AddFusionStrategy,
    GatedFusionStrategy,
    XAttnFusionStrategy,
    CanonicalAddFusionStrategy,
    CanonicalCrossAttentionFusionStrategy,
    CanonicalGatedFusionStrategy,
    IdentityFusionStrategy,
)

logger = logging.getLogger(__name__)


class FeatureFusion(nn.Module):
    """
    特征融合管理者（策略模式）
    """
    def __init__(
        self,
        feature_channels: List[int],
        fusion_type: str = "concat",
        optical_channels: Optional[List[int]] = None,
        sar_channels: Optional[List[int]] = None,
        # se / cbam 配置
        se_apply_levels: Optional[List[int]] = None,
        se_reduction: int = 16,
        se_gamma_init: float = 0.0,
        cbam_apply_levels: Optional[List[int]] = None,
        cbam_reduction: int = 16,
        cbam_spatial_kernel: int = 7,
        cbam_gamma_init: float = 0.0,
        # xattn 配置
        xattn_apply_levels: Optional[List[int]] = None,
        xattn_reduction: int = 8,
        xattn_gamma_init: float = 0.0,
        # 对齐引导（对齐偏置）
        xattn_align_bias: bool = False,
        xattn_align_bias_scale: float = 1.0,
        xattn_align_bias_learnable: bool = False,
        xattn_align_bias_init: float = 0.0,
        xattn_align_bias_mode: str = "legacy",
        xattn_align_bias_radius: int = 1,
        # MA-XAttn 组件消融 / 标准 cross-attention baseline
        xattn_components: str = "spatial+channel",
        xattn_component_fusion: str = "sum",
        xattn_attention_residual_init: float = 0.0,
        xattn_affinity_order: str = "opt_sar",
        xattn_attention_type: str = "ma",
        xattn_reliability_gate: bool = False,
        xattn_reliability_gate_mode: str = "conv",
        xattn_qk_scale: bool = False,
        xattn_attention_norm: str = "none",
        # 统一投影归一化
        projection_norm: str = "gn",
        # DropBranch 已移除
    ):
        super().__init__()
        valid_fusion_types = (
            "concat",
            "add",
            "gated",
            "se",
            "cbam",
            "xattn",
            "identity",
            "canonical_add",
            "canonical_gated",
            "canonical_cross",
        )
        assert fusion_type in valid_fusion_types, f"不支持的融合类型: {fusion_type}"

        # 基础元信息
        self.feature_channels = list(feature_channels)
        self.fusion_type = fusion_type
        self.num_levels = len(self.feature_channels)
        
        

        # per-stream 通道（由编码器提供）
        self.optical_channels = list(optical_channels) if optical_channels else None
        self.sar_channels = list(sar_channels) if sar_channels else None

        # 对齐长度（防止 zip 截断）
        if self.optical_channels and len(self.optical_channels) != self.num_levels:
            N = min(self.num_levels, len(self.optical_channels))
            logger.warning(f"optical_channels 长度({len(self.optical_channels)})与feature_channels({self.num_levels})不一致，按最短({N})对齐")
            self.feature_channels = self.feature_channels[:N]
            self.optical_channels = self.optical_channels[:N]
            if self.sar_channels:
                self.sar_channels = self.sar_channels[:N]
            self.num_levels = N
        if self.sar_channels and len(self.sar_channels) != self.num_levels:
            N = min(self.num_levels, len(self.sar_channels))
            logger.warning(f"sar_channels 长度({len(self.sar_channels)})与feature_channels({self.num_levels})不一致，按最短({N})对齐")
            self.feature_channels = self.feature_channels[:N]
            if self.optical_channels:
                self.optical_channels = self.optical_channels[:N]
            self.sar_channels = self.sar_channels[:N]
            self.num_levels = N

        # 构建策略
        self.strategy = self._build_strategy(
            fusion_type,
            {
                "se_apply_levels": se_apply_levels,
                "se_reduction": se_reduction,
                "se_gamma_init": se_gamma_init,
                "cbam_apply_levels": cbam_apply_levels,
                "cbam_reduction": cbam_reduction,
                "cbam_spatial_kernel": cbam_spatial_kernel,
                "cbam_gamma_init": cbam_gamma_init,
                "xattn_apply_levels": xattn_apply_levels,
                "xattn_reduction": xattn_reduction,
                "xattn_gamma_init": xattn_gamma_init,
                # 对齐引导
                "xattn_align_bias": xattn_align_bias,
                "xattn_align_bias_scale": xattn_align_bias_scale,
                "xattn_align_bias_learnable": xattn_align_bias_learnable,
                "xattn_align_bias_init": xattn_align_bias_init,
                "xattn_align_bias_mode": xattn_align_bias_mode,
                "xattn_align_bias_radius": xattn_align_bias_radius,
                "xattn_components": xattn_components,
                "xattn_component_fusion": xattn_component_fusion,
                "xattn_attention_residual_init": xattn_attention_residual_init,
                "xattn_affinity_order": xattn_affinity_order,
                "xattn_attention_type": xattn_attention_type,
                "xattn_reliability_gate": xattn_reliability_gate,
                "xattn_reliability_gate_mode": xattn_reliability_gate_mode,
                "xattn_qk_scale": xattn_qk_scale,
                "xattn_attention_norm": xattn_attention_norm,
                "proj_norm": projection_norm,
            },
        )

        # 输出通道由策略决定
        self.output_channels = self.strategy.output_channels

        # 单模态投影（仅 concat / se / cbam 需要维持 2c 的统计一致性）
        if fusion_type in ("concat", "se", "cbam"):
            # 若未提供 per-stream 通道，则使用 feature_channels 近似（统一架构）
            in_opt = self.optical_channels or self.feature_channels
            in_sar = self.sar_channels or self.feature_channels
            # 输出宽度等于策略输出通道（opt+sar）
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

        # 别名：xattn → MA-XAttn
        self.alias = "MA-XAttn" if fusion_type == "xattn" else fusion_type
        logger.info(f"特征融合初始化: {self.alias}")
        if self.optical_channels and self.sar_channels:
            logger.info(f"  per-stream: optical={self.optical_channels}, sar={self.sar_channels}")
        logger.info(f"  output_channels={self.output_channels}")
        # 模态丢弃已从管理者中移除，保持简化、可控

    def _build_strategy(self, fusion_type: str, cfg: Dict[str, Any]):
        # 如果缺一路通道信息，用 feature_channels 占位，以便策略构建
        opt_ch = self.optical_channels or self.feature_channels
        sar_ch = self.sar_channels or self.feature_channels
        if fusion_type == "concat":
            return ConcatFusionStrategy(opt_ch, sar_ch)
        if fusion_type == "identity":
            if self.optical_channels and self.sar_channels:
                raise ValueError("identity fusion 仅适用于单编码器输入")
            return IdentityFusionStrategy(opt_ch, sar_ch)
        if fusion_type == "canonical_add":
            return CanonicalAddFusionStrategy(opt_ch, sar_ch)
        if fusion_type == "canonical_gated":
            return CanonicalGatedFusionStrategy(opt_ch, sar_ch)
        if fusion_type == "canonical_cross":
            return CanonicalCrossAttentionFusionStrategy(
                opt_ch,
                sar_ch,
                reduction=cfg.get("xattn_reduction", 8),
                apply_levels=cfg.get("xattn_apply_levels") or [-1],
            )
        if fusion_type == "se":
            return SEFusionStrategy(opt_ch, sar_ch, reduction=cfg.get("se_reduction", 16), gamma_init=cfg.get("se_gamma_init", 0.0), apply_levels=cfg.get("se_apply_levels"))
        if fusion_type == "cbam":
            return CBAMFusionStrategy(opt_ch, sar_ch, reduction=cfg.get("cbam_reduction", 16), spatial_kernel=cfg.get("cbam_spatial_kernel", 7), gamma_init=cfg.get("cbam_gamma_init", 0.0), apply_levels=cfg.get("cbam_apply_levels"))
        if fusion_type == "add":
            return AddFusionStrategy(opt_ch, sar_ch, proj_norm=cfg.get("proj_norm", "gn"))
        if fusion_type == "gated":
            return GatedFusionStrategy(opt_ch, sar_ch, proj_norm=cfg.get("proj_norm", "gn"))
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
                align_bias_learnable=cfg.get(
                    "xattn_align_bias_learnable", False
                ),
                align_bias_init=cfg.get("xattn_align_bias_init", 0.0),
                align_bias_mode=cfg.get("xattn_align_bias_mode", "legacy"),
                align_bias_radius=cfg.get("xattn_align_bias_radius", 1),
                components=cfg.get("xattn_components", "spatial+channel"),
                component_fusion=cfg.get("xattn_component_fusion", "sum"),
                attention_residual_init=cfg.get(
                    "xattn_attention_residual_init", 0.0
                ),
                affinity_order=cfg.get("xattn_affinity_order", "opt_sar"),
                attention_type=cfg.get("xattn_attention_type", "ma"),
                reliability_gate=bool(cfg.get("xattn_reliability_gate", False)),
                reliability_gate_mode=str(cfg.get("xattn_reliability_gate_mode", "conv")),
                qk_scale=bool(cfg.get("xattn_qk_scale", False)),
                attention_norm=str(cfg.get("xattn_attention_norm", "none")),
            )
        raise ValueError(f"未知融合类型: {fusion_type}")

    # ------- 通用控制（保留空实现以兼容旧调用） -------
    def set_epoch(self, epoch: int):
        return

    # ------- 前向 -------
    def optical_only(self, optical_features: List[torch.Tensor]) -> List[torch.Tensor]:
        if hasattr(self.strategy, "optical_only"):
            return self.strategy.optical_only(optical_features)
        return self._single_modal_forward(optical_features, modality="optical")

    def sar_only(self, sar_features: List[torch.Tensor]) -> List[torch.Tensor]:
        if hasattr(self.strategy, "sar_only"):
            return self.strategy.sar_only(sar_features)
        return self._single_modal_forward(sar_features, modality="sar")

    def forward(
        self,
        optical_features: Optional[List[torch.Tensor]] = None,
        sar_features: Optional[List[torch.Tensor]] = None,
        optical_valid_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        if optical_features is None and sar_features is None:
            raise ValueError("至少需要提供一种模态的特征")
        if optical_features is None:
            assert sar_features is not None
            return self._single_modal_forward(sar_features, modality="sar")
        if sar_features is None:
            assert optical_features is not None
            return self._single_modal_forward(optical_features, modality="optical")

        # 双模态：先空间对齐再委托策略
        for i in range(len(optical_features)):
            fo, fs = optical_features[i], sar_features[i]
            if fo.shape[2:] != fs.shape[2:]:
                sar_features[i] = F.interpolate(fs, size=fo.shape[2:], mode='bilinear', align_corners=False)
        if optical_valid_mask is not None and getattr(self.strategy, "reliability_gate", None) is not None:
            out = self.strategy(optical_features, sar_features, optical_valid_mask=optical_valid_mask)
        else:
            out = self.strategy(optical_features, sar_features)
        return out

    

    def _single_modal_forward(self, feats: List[torch.Tensor], modality: str) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        if self.fusion_type == "identity":
            return list(feats)
        if self.fusion_type in ("concat", "se", "cbam"):
            proj_list = self.single_modal_proj_opt if modality == "optical" else self.single_modal_proj_sar
            for i, f in enumerate(feats):
                outs.append(proj_list[i](f))
            return outs
        else:
            # add / gated / xattn：用策略投影到 max 宽度
            proj_list = self.strategy.project_opt if modality == "optical" else self.strategy.project_sar
            for i, f in enumerate(feats):
                outs.append(proj_list[i](f))
            return outs