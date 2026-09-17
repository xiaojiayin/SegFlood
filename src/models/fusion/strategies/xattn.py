from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFusionStrategy
from ..blocks import (
    ChannelAttBlock,
    DWResidualProjection,
    SpatialAttBlock,
    XATTN_AFFINITY_ORDERS,
    XATTN_ATTENTION_TYPES,
    XATTN_COMPONENTS,
    _norm2d,
)


class XAttnFusionStrategy(BaseFusionStrategy):
    def __init__(
        self,
        optical_channels: List[int],
        sar_channels: List[int],
        proj_norm: str = "gn",
        reduction: int = 8,
        gamma_init: float = 0.0,
        apply_levels: Optional[List[int]] = None,
        align_bias: bool = False,
        align_bias_scale: float = 1.0,
        align_bias_learnable: bool = False,
        align_bias_init: float = 0.0,
        align_bias_mode: str = "legacy",
        align_bias_radius: int = 1,
        components: str = "spatial+channel",
        component_fusion: str = "sum",
        attention_residual_init: float = 0.0,
        affinity_order: str = "opt_sar",
        attention_type: str = "ma",
        reliability_gate: bool = False,
        reliability_gate_mode: str = "conv",  # conv: sigmoid(1x1[F_s;F_o]) | affinity: sigmoid(a*cos(F_o,F_s)+b)
        qk_scale: bool = False,
        attention_norm: str = "none",
    ):
        if len(optical_channels) != len(sar_channels):
            raise ValueError("optical_channels 与 sar_channels 的层数必须一致")
        if len(optical_channels) == 0:
            raise ValueError("至少需要一个特征层")
        if reduction <= 0:
            raise ValueError("reduction 必须为正整数")
        if components not in XATTN_COMPONENTS:
            raise ValueError(
                f"不支持的 components: {components}; 可选值为 {XATTN_COMPONENTS}"
            )
        if component_fusion not in {"sum", "gated", "residual_gated"}:
            raise ValueError(
                "component_fusion 必须是 sum、gated 或 residual_gated，"
                f"当前为 {component_fusion!r}"
            )
        if (
            component_fusion in {"gated", "residual_gated"}
            and components != "spatial+channel"
        ):
            raise ValueError(
                "gated/residual_gated component fusion 仅适用于 spatial+channel"
            )
        if affinity_order not in XATTN_AFFINITY_ORDERS:
            raise ValueError(
                f"不支持的 affinity_order: {affinity_order}; "
                f"可选值为 {XATTN_AFFINITY_ORDERS}"
            )
        if attention_type not in XATTN_ATTENTION_TYPES:
            raise ValueError(
                f"不支持的 attention_type: {attention_type}; "
                f"可选值为 {XATTN_ATTENTION_TYPES}"
            )
        if attention_type == "standard" and components != "spatial":
            raise ValueError(
                "标准双向 QK^T cross-attention baseline 要求 components='spatial'"
            )
        if attention_type == "standard" and affinity_order != "opt_sar":
            raise ValueError(
                "标准双向 cross-attention 不使用 affinity 矩阵乘法顺序；"
                "请保留 affinity_order='opt_sar'"
            )
        if attention_type == "standard" and align_bias:
            raise ValueError(
                "标准 cross-attention baseline 不支持 align_bias，"
                "以免混入 MA-XAttn 的对齐增强"
            )
        super().__init__(optical_channels, sar_channels)
        self.components = components
        self.component_fusion = component_fusion
        self.affinity_order = affinity_order
        self.attention_type = attention_type
        # 解析允许的层级：支持负索引（-1 最深，-2 次深）。若解析后为空，则退回仅最深层。
        self.apply_levels = set(apply_levels or [-1])
        # 解析为实际索引 0..num_levels-1
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
        # 方案D（Q/K 调制）已移除
        # 统一投影
        self.project_opt = nn.ModuleList([
            DWResidualProjection(in_ch, out_ch, norm=proj_norm)
            for in_ch, out_ch in zip(self.optical_channels, self.output_channels)
        ])
        self.project_sar = nn.ModuleList([
            DWResidualProjection(in_ch, out_ch, norm=proj_norm)
            for in_ch, out_ch in zip(self.sar_channels, self.output_channels)
        ])
        # 注意力块
        self.blocks = nn.ModuleList()
        self.res_proj = nn.ModuleList()
        self.reliability_gate = None
        self.reliability_gate_mode = str(reliability_gate_mode).lower()
        if self.reliability_gate_mode not in {"conv", "affinity"}:
            raise ValueError(f"reliability_gate_mode 必须是 conv 或 affinity，当前为 {reliability_gate_mode!r}")
        if reliability_gate:
            if self.reliability_gate_mode == "conv":
                self.reliability_gate = nn.ModuleList([
                    nn.Conv2d(2 * c, 1, kernel_size=1) for c in self.output_channels
                ])
            else:
                # 亲和度驱动：G = sigmoid(a·cos(F_o, F_s) + b)，a 初值 5、b 初值 0（cos≈0 时 G≈0.5，cos→1 时 G→1）。
                # 可靠性完全来自光学–SAR 一致性，不依赖外部掩膜——这是 alignment-guided 的机制本体。
                self.reliability_gate = nn.ParameterList([
                    nn.Parameter(torch.tensor([5.0, 0.0])) for _ in self.output_channels
                ])
        self.gamma = nn.ParameterList()
        self.component_logits = (
            nn.ParameterList()
            if self.component_fusion in {"gated", "residual_gated"}
            else None
        )
        self.attention_residual_scale = (
            nn.ParameterList()
            if self.component_fusion == "residual_gated"
            else None
        )
        for i in range(self.num_levels):
            c = self.output_channels[i]
            att_ch = max(1, c // reduction)
            block = nn.ModuleDict()
            if self.components in ("spatial", "spatial+channel"):
                block["spatial"] = SpatialAttBlock(
                    c,
                    att_ch,
                    c,
                    enable_align_bias=self.align_bias,
                    align_bias_scale=self.align_bias_scale,
                    align_bias_learnable=align_bias_learnable,
                    align_bias_init=align_bias_init,
                    align_bias_mode=align_bias_mode,
                    align_bias_radius=align_bias_radius,
                    affinity_order=self.affinity_order,
                    attention_type=self.attention_type,
                    qk_scale=qk_scale,
                    attention_norm=attention_norm,
                )
            if self.components in ("channel", "spatial+channel"):
                block["channel"] = ChannelAttBlock(
                    c, att_ch, c, affinity_order=self.affinity_order, qk_scale=qk_scale,
                    attention_norm=attention_norm,
                )
            block["out"] = nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False),
                _norm2d(proj_norm, c),
                nn.ReLU(inplace=True),
            )
            self.blocks.append(block)
            self.res_proj.append(nn.Conv2d(self.optical_channels[i] + self.sar_channels[i], c, 1, bias=False))
            self.gamma.append(nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32)))
            if self.component_logits is not None:
                # 2*softmax([0,0])=[1,1]，初始化时严格复现原始求和。
                self.component_logits.append(nn.Parameter(torch.zeros(2)))
            if self.attention_residual_scale is not None:
                self.attention_residual_scale.append(
                    nn.Parameter(
                        torch.tensor(
                            float(attention_residual_init),
                            dtype=torch.float32,
                        )
                    )
                )
        # 运行态信息

    def _calculate_output_channels(self) -> List[int]:
        return [max(opt, sar) for opt, sar in zip(self.optical_channels, self.sar_channels)]

    

    def optical_only(self, optical_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Optical-only 路径（对称双路径辅助头）：每层只用光学投影。"""
        return [self.project_opt[i](fo) for i, fo in enumerate(optical_features)]

    def sar_only(self, sar_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """SAR-only 路径（SMAGNet 式双路径辅助头）：每层只用 SAR 投影，特征宽度与融合输出一致。"""
        return [self.project_sar[i](fs) for i, fs in enumerate(sar_features)]

    def forward(
        self,
        optical_features: List[torch.Tensor],
        sar_features: List[torch.Tensor],
        optical_valid_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
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
            component_outputs = []
            if "spatial" in b:
                component_outputs.append(b["spatial"](fo_a, fs_a))
            if "channel" in b:
                component_outputs.append(b["channel"](fo_a, fs_a))
            if self.component_logits is not None:
                weights = 2.0 * torch.softmax(self.component_logits[i], dim=0)
                core = (
                    weights[0] * component_outputs[0]
                    + weights[1] * component_outputs[1]
                )
            else:
                core = sum(component_outputs[1:], component_outputs[0])
            out = b["out"](core)
            if self.attention_residual_scale is not None:
                # 以强 dual-add 为基础，只学习 attention 的增量贡献。
                base = fo_a + fs_a
                out = base + self.attention_residual_scale[i] * out
            res = self.res_proj[i](torch.cat([fo, fs], dim=1))
            out = out + self.gamma[i] * res
            if self.reliability_gate is not None:
                # SMAGNet 式互补门：G = sigmoid(conv1x1([F_s; F_o]))；SM = 光学有效掩膜（缺测/云处为 0）。
                # F* = SM·G·out_MA + (1 − SM·G)·F_s，光学不可用处退化为纯 SAR。
                if self.reliability_gate_mode == "conv":
                    g = torch.sigmoid(self.reliability_gate[i](torch.cat([fs_a, fo_a], dim=1)))
                else:
                    cos = F.cosine_similarity(fo_a, fs_a, dim=1, eps=1e-6).unsqueeze(1)
                    ab = self.reliability_gate[i]
                    g = torch.sigmoid(ab[0] * cos + ab[1])
                if optical_valid_mask is not None:
                    sm = F.interpolate(optical_valid_mask.to(g.dtype), size=g.shape[-2:], mode="area").detach()
                    g = g * sm
                out = g * out + (1.0 - g) * fs_a
            fused.append(out)
        return fused


