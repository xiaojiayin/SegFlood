import math
import torch
import torch.nn as nn
import torch.nn.functional as F


XATTN_COMPONENTS = ("spatial", "channel", "spatial+channel")
XATTN_AFFINITY_ORDERS = ("opt_sar", "sar_opt")
XATTN_ATTENTION_TYPES = ("ma", "standard")


def _norm2d(kind: str, num_channels: int, groups: int = 32) -> nn.Module:
    if kind == "gn":
        # 选择不超过 groups 的最大公约数分组，确保可整除
        g_max = min(groups, num_channels)
        g = g_max
        while g > 1 and (num_channels % g) != 0:
            g -= 1
        # 若找不到>1的约数，则退化为每组=通道（等价 LayerNorm 按通道）
        if g <= 0:
            g = 1
        return nn.GroupNorm(g, num_channels)
    return nn.BatchNorm2d(num_channels)


class DWResidualProjection(nn.Module):
    """1x1 主分支 + 轻残差(Depthwise3x3→Norm→ReLU→Pointwise1x1) + 可学习 gamma。

    用于 add/gated/xattn 的统一通道对齐到目标宽度。
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
    """空间 MA-XAttn 或标准双向 token cross-attention。"""

    def __init__(self, in_channels: int, att_channels: int, out_channels: int,
                 enable_align_bias: bool = False, align_bias_scale: float = 1.0,
                 align_bias_learnable: bool = False,
                 align_bias_init: float = 0.0,
                 align_bias_mode: str = "legacy", align_bias_radius: int = 1,
                 affinity_order: str = "opt_sar", attention_type: str = "ma",
                 qk_scale: bool = False, attention_norm: str = "none",
                 logit_scale_init: float = 10.0):
        super().__init__()
        # P1 归一化注意力：cosine = q、k 沿特征维 L2 归一化，logits = cos · exp(logit_scale)（可学习温度，
        # Swin-V2 / CLIP 式，init 10 ⇔ τ=0.1，上限 100）。修复未归一化 logits 幅值数百导致的 softmax 饱和。
        self.attention_norm = str(attention_norm).lower()
        if self.attention_norm not in {"none", "cosine"}:
            raise ValueError(f"attention_norm 必须是 none 或 cosine，当前为 {attention_norm!r}")
        self.logit_scale = (
            nn.Parameter(torch.tensor(math.log(float(logit_scale_init))))
            if self.attention_norm == "cosine" else None
        )
        if att_channels <= 0:
            raise ValueError("att_channels 必须为正整数")
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
        if attention_type == "standard" and enable_align_bias:
            raise ValueError("标准 cross-attention baseline 不支持 align bias")
        if align_bias_mode not in {"legacy", "local_cross"}:
            raise ValueError(
                "align_bias_mode 必须是 legacy 或 local_cross，"
                f"当前为 {align_bias_mode!r}"
            )
        if align_bias_radius < 0:
            raise ValueError("align_bias_radius 不能为负数")
        self.enable_align_bias = enable_align_bias
        self.align_bias_scale = float(align_bias_scale)
        self.align_bias_weight = (
            nn.Parameter(torch.tensor(float(align_bias_init)))
            if align_bias_learnable
            else None
        )
        self.align_bias_mode = align_bias_mode
        self.align_bias_radius = int(align_bias_radius)
        self.affinity_order = affinity_order
        self.qk_scale = bool(qk_scale)  # True: MA 路 logits 乘 1/sqrt(d)（默认 False = 论文现行模型）
        self.attention_type = attention_type
        self.att_channels = att_channels
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
        空间注意力前向传播（含对齐偏置）
        
        张量流说明（假设输入为 [B, C, h, w]）：
        ============================================
        【输入】
        x_opt: [B, C, h, w] - 投影后的光学特征
        x_sar: [B, C, h, w] - 投影后的SAR特征
        
        【步骤1：计算Q/K/V】
        q_opt: [B, HW, att_ch] - 光学查询
        k_opt: [B, att_ch, HW] - 光学键
        v_opt: [B, HW, att_ch] - 光学值
        q_sar: [B, HW, att_ch] - SAR查询
        k_sar: [B, att_ch, HW] - SAR键
        v_sar: [B, HW, att_ch] - SAR值
        
        【步骤2：计算注意力logits】
        logits_opt: [B, HW, HW] - 光学自注意力分数矩阵
        logits_sar: [B, HW, HW] - SAR自注意力分数矩阵
        
        【步骤3：对齐偏置注入（如果启用）】⭐
        - 计算逐像素余弦相似度: sim[i] = cosine(x_opt[i], x_sar[i])
        - 归一化: bias = scale * (1/√HW) * sim
        - 注入: logits[i,i] += bias[i]  (仅对角位置)
        - 效果: 对齐好的位置自注意力权重增强
        
        【步骤4：注意力计算】
        att_opt: [B, HW, HW] - 光学注意力权重（softmax后）
        att_sar: [B, HW, HW] - SAR注意力权重（softmax后）
        att_cross: [B, HW, HW] - 跨模态注意力（att_opt @ att_sar）
        
        【步骤5：特征聚合】
        s_opt: [B, C, h, w] - 光学聚合特征
        s_sar: [B, C, h, w] - SAR聚合特征
        
        【输出】
        return: [B, C, h, w] - 融合后的空间特征
        """
        if x_opt.ndim != 4 or x_sar.ndim != 4:
            raise ValueError("x_opt 与 x_sar 必须是 [B, C, H, W] 四维张量")
        if x_opt.shape != x_sar.shape:
            raise ValueError(
                "SpatialAttBlock 要求两路输入 shape 完全一致，"
                f"实际为 {tuple(x_opt.shape)} 与 {tuple(x_sar.shape)}"
            )
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

        if self.attention_type == "standard":
            scale = 1.0 / math.sqrt(float(self.att_channels))
            # 光学 token 查询 SAR，SAR token 反向查询光学；这是标准 QK^T，
            # 与 MA-XAttn 的单模态 affinity 矩阵乘积不是同一运算。
            att_opt_to_sar = torch.softmax(
                torch.bmm(q_opt, k_sar) * scale, dim=-1
            )
            att_sar_to_opt = torch.softmax(
                torch.bmm(q_sar, k_opt) * scale, dim=-1
            )
            s_opt = torch.bmm(att_opt_to_sar, v_sar)
            s_sar = torch.bmm(att_sar_to_opt, v_opt)
            s_opt = s_opt.permute(0, 2, 1).contiguous().reshape(n, -1, h, w)
            s_sar = s_sar.permute(0, 2, 1).contiguous().reshape(n, -1, h, w)
            return self.proj_opt(s_opt) + self.proj_sar(s_sar)

        if self.attention_norm == "cosine":
            # q: [B, HW, d]（dim=-1 为特征），k: [B, d, HW]（dim=1 为特征）
            q_opt, q_sar = F.normalize(q_opt, dim=-1), F.normalize(q_sar, dim=-1)
            k_opt, k_sar = F.normalize(k_opt, dim=1), F.normalize(k_sar, dim=1)
            ls = self.logit_scale.clamp(max=math.log(100.0)).exp()
            logits_opt = torch.bmm(q_opt, k_opt) * ls
            logits_sar = torch.bmm(q_sar, k_sar) * ls
        else:
            logits_opt = torch.bmm(q_opt, k_opt)
            logits_sar = torch.bmm(q_sar, k_sar)
            if self.qk_scale:
                sc = 1.0 / math.sqrt(float(self.att_channels))
                logits_opt = logits_opt * sc
                logits_sar = logits_sar * sc
        if self.enable_align_bias and self.align_bias_mode == "legacy":
            # 【对齐偏置计算】
            # 以投影后特征的逐像素余弦相似度，给对角位置添加 bias
            # 这相当于"先对齐，再利用对齐信息指导融合"
            xo_n = F.normalize(x_opt, dim=1)  # [B, C, h, w] L2归一化
            xs_n = F.normalize(x_sar, dim=1)  # [B, C, h, w] L2归一化
            sim = (xo_n * xs_n).sum(dim=1).reshape(n, -1)  # [B, HW] 逐像素余弦相似度
            length = float(h * w)
            norm = 1.0 / math.sqrt(max(1.0, length))  # 归一化因子，避免偏置过大
            bias_scale = (
                self.align_bias_weight
                if self.align_bias_weight is not None
                else self.align_bias_scale
            )
            bias = (bias_scale * norm) * sim  # [B, HW] 对齐偏置
            # 注入到注意力logits的对角位置（仅自注意力位置）
            logits_opt = logits_opt + torch.diag_embed(bias)  # [B, HW, HW]
            logits_sar = logits_sar + torch.diag_embed(bias)  # [B, HW, HW]

        att_opt = torch.softmax(logits_opt, dim=-1)
        att_sar = torch.softmax(logits_sar, dim=-1)
        if self.affinity_order == "opt_sar":
            att_cross = torch.bmm(att_opt, att_sar)
        else:
            att_cross = torch.bmm(att_sar, att_opt)
        if self.enable_align_bias and self.align_bias_mode == "local_cross":
            # 直接在跨模态亲和上注入局部 correspondence prior。与旧实现只
            # 修改单模态自注意力对角线不同，该先验允许半径内的小范围错位。
            xo_tokens = F.normalize(
                x_opt.flatten(2).transpose(1, 2), dim=-1
            )
            xs_tokens = F.normalize(
                x_sar.flatten(2).transpose(1, 2), dim=-1
            )
            cross_similarity = torch.bmm(
                xo_tokens, xs_tokens.transpose(1, 2)
            )
            yy = torch.arange(h, device=x_opt.device).repeat_interleave(w)
            xx = torch.arange(w, device=x_opt.device).repeat(h)
            local = (
                torch.maximum(
                    (yy[:, None] - yy[None, :]).abs(),
                    (xx[:, None] - xx[None, :]).abs(),
                )
                <= self.align_bias_radius
            )
            local_prior = cross_similarity * local.to(cross_similarity.dtype)
            guided_logits = torch.log(att_cross.clamp_min(1e-8))
            bias_scale = (
                self.align_bias_weight
                if self.align_bias_weight is not None
                else self.align_bias_scale
            )
            guided_logits = guided_logits + bias_scale * local_prior
            att_cross = torch.softmax(guided_logits, dim=-1)

        s_opt = torch.bmm(att_cross, v_opt).permute(0, 2, 1).contiguous().reshape(n, -1, h, w)
        s_opt = self.proj_opt(s_opt)
        s_sar = torch.bmm(att_cross, v_sar).permute(0, 2, 1).contiguous().reshape(n, -1, h, w)
        s_sar = self.proj_sar(s_sar)
        return s_opt + s_sar


class ChannelAttBlock(nn.Module):
    """跨模态通道注意力。"""

    def __init__(self, in_channels: int, att_channels: int, out_channels: int,
                 affinity_order: str = "opt_sar", qk_scale: bool = False,
                 attention_norm: str = "none", logit_scale_init: float = 10.0):
        super().__init__()
        self.qk_scale = bool(qk_scale)
        self.attention_norm = str(attention_norm).lower()
        if self.attention_norm not in {"none", "cosine"}:
            raise ValueError(f"attention_norm 必须是 none 或 cosine，当前为 {attention_norm!r}")
        self.logit_scale = (
            nn.Parameter(torch.tensor(math.log(float(logit_scale_init))))
            if self.attention_norm == "cosine" else None
        )
        if att_channels <= 0:
            raise ValueError("att_channels 必须为正整数")
        if affinity_order not in XATTN_AFFINITY_ORDERS:
            raise ValueError(
                f"不支持的 affinity_order: {affinity_order}; "
                f"可选值为 {XATTN_AFFINITY_ORDERS}"
            )
        self.affinity_order = affinity_order
        self.q_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.k_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.v_opt = nn.Conv2d(in_channels, att_channels, 1, bias=False)

        self.q_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.k_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)
        self.v_sar = nn.Conv2d(in_channels, att_channels, 1, bias=False)

        self.proj_opt = nn.Conv2d(att_channels, out_channels, 1, bias=False)
        self.proj_sar = nn.Conv2d(att_channels, out_channels, 1, bias=False)

        

    def forward(self, x_opt: torch.Tensor, x_sar: torch.Tensor):
        if x_opt.ndim != 4 or x_sar.ndim != 4:
            raise ValueError("x_opt 与 x_sar 必须是 [B, C, H, W] 四维张量")
        if x_opt.shape != x_sar.shape:
            raise ValueError(
                "ChannelAttBlock 要求两路输入 shape 完全一致，"
                f"实际为 {tuple(x_opt.shape)} 与 {tuple(x_sar.shape)}"
            )
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

        if self.attention_norm == "cosine":
            # 通道注意力的内积维度是 HW：q [B, C', HW] 沿 dim=-1 归一化，k [B, HW, C'] 沿 dim=1 归一化
            q_opt, q_sar = F.normalize(q_opt, dim=-1), F.normalize(q_sar, dim=-1)
            k_opt, k_sar = F.normalize(k_opt, dim=1), F.normalize(k_sar, dim=1)
            sc = self.logit_scale.clamp(max=math.log(100.0)).exp()
        else:
            sc = (1.0 / math.sqrt(float(h * w))) if self.qk_scale else 1.0  # 通道图的内积维度是 HW
        att_opt = torch.softmax(torch.bmm(q_opt, k_opt) * sc, dim=-1)
        att_sar = torch.softmax(torch.bmm(q_sar, k_sar) * sc, dim=-1)
        if self.affinity_order == "opt_sar":
            att_cross = torch.bmm(att_opt, att_sar)
        else:
            att_cross = torch.bmm(att_sar, att_opt)

        s_opt = torch.bmm(att_cross, v_opt).reshape(n, -1, h, w)
        s_opt = self.proj_opt(s_opt)
        s_sar = torch.bmm(att_cross, v_sar).reshape(n, -1, h, w)
        s_sar = self.proj_sar(s_sar)
        return s_opt + s_sar


