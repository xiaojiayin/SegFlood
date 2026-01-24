from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm2d(kind: str, num_channels: int, groups: int = 32) -> nn.Module:
    if kind == "gn":
        # Choose the largest group count <= groups that divides num_channels.
        g_max = min(groups, num_channels)
        g = g_max
        while g > 1 and (num_channels % g) != 0:
            g -= 1
        # Fall back to 1 group if no divisor > 1 exists.
        if g <= 0:
            g = 1
        return nn.GroupNorm(g, num_channels)
    return nn.BatchNorm2d(num_channels)


class DWResidualProjection(nn.Module):
    """1x1 main branch + lightweight residual (DWConv3x3->Norm->ReLU->PWConv1x1) with learnable gamma.

    Used for add/xattn channel projection to a target width.
    """

    def __init__(self, in_ch: int, out_ch: int, norm: str = "gn", gn_groups: int = 32):
        super().__init__()
        self.shortcut = (
            nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        self.main_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.res_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            _norm2d(norm, out_ch, gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
        )
        self.gamma = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main_conv(x)
        y = self.res_conv(y)
        return self.shortcut(x) + self.gamma * y


class SpatialAttBlock(nn.Module):
    """Cross-modal spatial attention (optionally with alignment bias)."""

    def __init__(self, in_channels: int, att_channels: int, out_channels: int,
                 enable_align_bias: bool = False, align_bias_scale: float = 1.0):
        super().__init__()
        self.enable_align_bias = enable_align_bias
        self.align_bias_scale = float(align_bias_scale)
        self.q_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.k_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.v_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)

        self.q_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.k_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.v_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)

        self.proj_opt = nn.Conv2d(att_channels, out_channels, 1, bias=False)
        self.proj_sar = nn.Conv2d(att_channels, out_channels, 1, bias=False)

        

    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor):
        """
        Spatial attention forward (with optional alignment bias).
        """
        n, _, h, w = x_opt.shape
        

        q_opt_feat = self.q_opt(x_opt)
        k_opt_feat = self.k_opt(x_opt)
        q_sar_feat = self.q_sar(x_sar)
        k_sar_feat = self.k_sar(x_sar)

        

        q_opt = q_opt_feat.reshape(n, -1, h * w).permute(0, 2, 1).contiguous()
        k_opt = k_opt_feat.reshape(n, -1, h * w)
        v_opt = self.v_opt(x_opt).reshape(n, -1, h * w).permute(0, 2, 1).contiguous()

        q_sar = q_sar_feat.reshape(n, -1, h * w).permute(0, 2, 1).contiguous()
        k_sar = k_sar_feat.reshape(n, -1, h * w)
        v_sar = self.v_sar(x_sar).reshape(n, -1, h * w).permute(0, 2, 1).contiguous()

        logits_opt = torch.bmm(q_opt, k_opt)
        logits_sar = torch.bmm(q_sar, k_sar)
        if self.enable_align_bias:
            # Alignment bias: add a per-pixel cosine-similarity bias to the diagonal of attention logits.
            xo_n = F.normalize(x_opt, dim=1)
            xs_n = F.normalize(x_sar, dim=1)
            sim = (xo_n * xs_n).sum(dim=1).reshape(n, -1)  # [B, HW]
            length = float(h * w)
            norm = 1.0 / math.sqrt(max(1.0, length))
            bias = (self.align_bias_scale * norm) * sim
            logits_opt = logits_opt + torch.diag_embed(bias)
            logits_sar = logits_sar + torch.diag_embed(bias)  # [B, HW, HW]

        att_opt = torch.softmax(logits_opt, dim=-1)
        att_sar = torch.softmax(logits_sar, dim=-1)
        att_cross = torch.bmm(att_opt, att_sar)

        s_opt = torch.bmm(att_cross, v_opt).permute(0, 2, 1).contiguous().reshape(n, -1, h, w)
        s_opt = self.proj_opt(s_opt)
        s_sar = torch.bmm(att_cross, v_sar).permute(0, 2, 1).contiguous().reshape(n, -1, h, w)
        s_sar = self.proj_sar(s_sar)
        return s_opt + s_sar


class ChannelAttBlock(nn.Module):
    """Cross-modal channel attention."""

    def __init__(self, in_channels: int, att_channels: int, out_channels: int):
        super().__init__()
        self.q_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.k_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.v_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)

        self.q_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.k_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.v_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)

        self.proj_opt = nn.Conv2d(att_channels, out_channels, 1, bias=False)
        self.proj_sar = nn.Conv2d(att_channels, out_channels, 1, bias=False)

        

    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor):
        n, _, h, w = x_opt.shape
        

        q_opt_feat = self.q_opt(x_opt)
        k_opt_feat = self.k_opt(x_opt)
        q_sar_feat = self.q_sar(x_sar)
        k_sar_feat = self.k_sar(x_sar)

        

        q_opt = q_opt_feat.reshape(n, -1, h * w)
        k_opt = k_opt_feat.reshape(n, -1, h * w).permute(0, 2, 1).contiguous()
        v_opt = self.v_opt(x_opt).reshape(n, -1, h * w)

        q_sar = q_sar_feat.reshape(n, -1, h * w)
        k_sar = k_sar_feat.reshape(n, -1, h * w).permute(0, 2, 1).contiguous()
        v_sar = self.v_sar(x_sar).reshape(n, -1, h * w)

        att_opt = torch.softmax(torch.bmm(q_opt, k_opt), dim=-1)
        att_sar = torch.softmax(torch.bmm(q_sar, k_sar), dim=-1)
        att_cross = torch.bmm(att_opt, att_sar)

        s_opt = torch.bmm(att_cross, v_opt).reshape(n, -1, h, w)
        s_opt = self.proj_opt(s_opt)
        s_sar = torch.bmm(att_cross, v_sar).reshape(n, -1, h, w)
        s_sar = self.proj_sar(s_sar)
        return s_opt + s_sar


