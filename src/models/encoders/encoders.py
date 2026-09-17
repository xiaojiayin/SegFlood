"""
编码器模块（Encoders）：多模态特征提取的标准化实现

职责
- 创建光学与 SAR 两路编码器
- 导出按层的通道宽度与降采样倍率（feature_channels/reductions）

特性
- 通过 `timm.create_model(features_only=True, in_chans=...)` 创建编码器
- 从 `feature_info` 读取元数据；在缺失时做一次轻量前向探测
- 按降采样倍率进行“语义对齐”（reductions-driven alignment）

建议
- 避免硬编码层索引；优先用 feature_info.channels()/reduction()
- 对齐时选择“参考分支”为层数更少的一侧（若相等默认 optical）
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import logging

from .feature_meta import (
    probe_timm_feature_meta,
    probe_timm_default_out_indices,
    match_out_indices_by_reduction,
)
from .shallow import ShallowDetailEncoder

logger = logging.getLogger(__name__)

# 内部常量
_DEFAULT_FEATURE_SIZE = (256, 256)


def create_encoder(
    model_name: str,
    in_channels: int,
    pretrained: bool = True,
    name: str = "encoder",
    out_indices: Optional[Tuple[int, ...]] = None,
    output_stride: Optional[int] = None,
    drop_path_rate: Optional[float] = None,  # 仅保留这一项：stochastic depth
    pretrained_cfg_overlay: Optional[dict] = None,
) -> nn.Module:
    """
    创建编码器，使用timm库，features_only=True；只透传drop_path_rate
    
    Args:
        model_name: 模型架构名称
        in_channels: 输入通道数  
        pretrained: 是否使用预训练
        name: 编码器名称
        out_indices: 指定输出层级（None为默认）
        output_stride: 输出步长（仅支持部分CNN架构）
        drop_path_rate: stochastic depth率（所有模型支持，推荐0.1-0.3）
        
    Returns:
        配置好的编码器，附带feature_channels和feature_reductions属性
    """
    try:
        import timm
    except ImportError:
        raise ImportError("需要安装timm: pip install timm")

    kwargs = dict(
        pretrained=pretrained,
        features_only=True,
        in_chans=in_channels,
    )
    if out_indices is not None:
        kwargs["out_indices"] = out_indices
    if output_stride is not None:
        kwargs["output_stride"] = output_stride
    if drop_path_rate is not None:
        kwargs["drop_path_rate"] = float(drop_path_rate)
    # 若显式提供本地权重文件，则优先使用 checkpoint_path 进行本地加载，并避免走 Hub
    overlay_file: Optional[str] = None
    if isinstance(pretrained_cfg_overlay, dict):
        overlay_file = pretrained_cfg_overlay.get("file")  # type: ignore[assignment]
        if overlay_file:
            # 使用本地 checkpoint_path，并关闭预训练远程拉取
            kwargs["checkpoint_path"] = overlay_file
            kwargs["pretrained"] = False
        else:
            # 保留其它 overlay 键（如确有需要）
            kwargs["pretrained_cfg_overlay"] = dict(pretrained_cfg_overlay)

    try:
        model = timm.create_model(model_name, **kwargs)
        level_desc = f"{len(out_indices)}层" if out_indices is not None else "默认层级"
        logger.info(
            f"{name}创建成功({level_desc}): {model_name}, in_chans={in_channels}, "
            f"drop_path_rate={kwargs.get('drop_path_rate')}"
        )
    except Exception as e:
        # 精准回退：仅当为层级/stride 参数不被支持时才移除并重试，否则抛出
        msg = str(e).lower()
        is_param_issue = isinstance(e, (TypeError, ValueError)) and (
            ("out_indices" in msg) or ("output_stride" in msg)
        )
        if not is_param_issue:
            raise
        for k in ("output_stride", "out_indices"):
            if k in kwargs:
                kwargs.pop(k)
        logger.info(f"{name} 层级/stride 参数不被支持，已移除并回退创建: {e}")
        model = timm.create_model(model_name, **kwargs)
        logger.info(
            f"{name}创建成功(回退后): {model_name}, in_chans={in_channels}, "
            f"drop_path_rate={kwargs.get('drop_path_rate')}"
        )

    # 填充通道与降采样信息
    model.feature_channels = _get_feature_channels(model, model_name, in_channels)
    model.feature_reductions = _get_feature_reductions(model, model_name, in_channels)

    logger.info(f"{name}编码器: {model_name}, 通道{in_channels}, 特征{model.feature_channels}, 降采样{model.feature_reductions}")
    return model


def _get_feature_channels(encoder: nn.Module, model_name: str, in_channels: int) -> List[int]:
    """
    获取特征通道数（优先feature_info，失败则前向验证，都失败则fail fast）
    """
    # 1) feature_info优先 - 最可靠
    if hasattr(encoder, "feature_info"):
        try:
            ch = list(encoder.feature_info.channels())
            if all(isinstance(c, int) and c > 0 for c in ch):
                return ch
        except Exception:
            pass

    # 2) 前向验证备选 - 实际测试
    was_training = encoder.training
    try:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        encoder.eval()
        x = torch.randn(1, in_channels, *_DEFAULT_FEATURE_SIZE, device=device)
        with torch.no_grad():
            feats = encoder(x)
        if isinstance(feats, (list, tuple)) and len(feats) > 0:
            return [f.shape[1] for f in feats]
    finally:
        if was_training:
            encoder.train()

    # 3) 都失败则抛出异常（fail fast原则）
    raise RuntimeError(f"无法获取模型 {model_name} 的特征通道信息（features_only 必须可用）")


def _get_feature_reductions(encoder: nn.Module, model_name: str, in_channels: int) -> List[int]:
    """
    获取特征降采样倍率（优先feature_info，失败则前向估计）
    """
    # 1) feature_info优先
    if hasattr(encoder, "feature_info"):
        try:
            rd = list(encoder.feature_info.reduction())
            if all(isinstance(r, int) and r > 0 for r in rd):
                return rd[:len(getattr(encoder, "feature_channels", rd))]
        except Exception:
            pass

    # 2) 兜底：用一次前向估计（基于高度）
    was_training = encoder.training
    try:
        try:
            device = next(encoder.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        encoder.eval()
        H, W = _DEFAULT_FEATURE_SIZE
        x = torch.randn(1, in_channels, H, W, device=device)
        with torch.no_grad():
            feats = encoder(x)
        if isinstance(feats, (list, tuple)) and len(feats) > 0:
            rds = []
            for f in feats:
                if len(f.shape) >= 4:  # [B,C,H,W]
                    h = max(1, int(f.shape[2]))
                    r = max(1, round(H / h))
                    rds.append(r)
                else:
                    # 可能是ViT序列输出，需要特殊处理
                    logger.warning(f"特征形状异常: {f.shape}，可能需要reshape为2D")
                    rds.append(16)  # 默认值
            return rds
    finally:
        if was_training:
            encoder.train()
    
    raise RuntimeError(f"无法获取模型 {model_name} 的降采样信息（features_only 必须可用）")


class DualStreamEncoder(nn.Module):
    """
    双流编码器（按 reduction 智能对齐）
    
    特性：
    - 参考流：自动选择层数更少的一侧以减少无用计算
    - 保存每条流的 channels/reductions 以及对齐后的 reductions，供下游解码器使用
    - 智能对齐：按reduction语义对齐，避免无意义插值
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        optical_model_name: Optional[str] = None,
        sar_model_name: Optional[str] = None,
        optical_channels: int = 4,
        sar_channels: int = 1,
        optical_pretrained: bool = True,
        sar_pretrained: bool = True,
        output_stride: Optional[int] = None,  # 可选，对 CNN 有效
        drop_path_rate: Optional[float] = None,  # 可选：透传到各自的timm骨干
        pretrained_cfg_overlay: Optional[dict] = None,
        # 仅当主干为单尺度（例如 ViT/DINOv3）时，启用浅层细节编码器补充 4x/8x
        shallow_enabled: bool = False,
        shallow_norm: str = "bn",
        shallow_c2: int = 64,
        shallow_c4: int = 128,
        shallow_c8: int = 256,
    ):
        super().__init__()
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels

        optical_model = optical_model_name if optical_model_name else model_name
        sar_model = sar_model_name if sar_model_name else model_name

        # 轻量探测（不加载权重，带缓存）
        probe_opt_ch, probe_opt_rd = self._probe_if_needed(optical_model, optical_channels)
        probe_sar_ch, probe_sar_rd = self._probe_if_needed(sar_model, sar_channels)

        # 选择参考流：谁层数少选谁（若相等，默认 optical）
        len_opt = len(probe_opt_ch) if probe_opt_ch else 1e9
        len_sar = len(probe_sar_ch) if probe_sar_ch else 1e9
        # 参考分支：选择层数更少的一侧作为对齐参考（若相等，默认 optical）
        ref_stream = "sar" if len_sar < len_opt else "optical"
        logger.info(f"参考分支: {ref_stream} (optical:{len_opt if len_opt<1e9 else 0}层, sar:{len_sar if len_sar<1e9 else 0}层)")

        # 创建编码器
        self.encoder_optical = None
        self.encoder_sar = None
        self.shallow_opt = None
        self.shallow_sar = None
        
        if ref_stream == "optical":
            # 光学为参考流
            if optical_channels > 0:
                self.encoder_optical = create_encoder(
                    model_name=optical_model,
                    in_channels=optical_channels,
                    pretrained=optical_pretrained,
                    name="optical",
                    out_indices=None,  # 参考流保留默认层级
                    output_stride=output_stride,
                    drop_path_rate=drop_path_rate,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                )
            
            # 目标流对齐
            sar_out_indices = None
            if sar_channels > 0 and probe_opt_rd and probe_sar_rd:
                ref_rd = self.encoder_optical.feature_reductions
                sar_default_idx = probe_timm_default_out_indices(sar_model, sar_channels)
                sar_out_indices = match_out_indices_by_reduction(ref_rd, probe_sar_rd, sar_default_idx)
                logger.info(
                    f"SAR 按 reduction 对齐 indices: {sar_out_indices if sar_out_indices is not None else '默认层级'} "
                    f"(参考:{ref_rd}, 候选:{probe_sar_rd}, 候选默认out_indices:{sar_default_idx})"
                )
            
            if sar_channels > 0:
                self.encoder_sar = create_encoder(
                    model_name=sar_model,
                    in_channels=sar_channels,
                    pretrained=sar_pretrained,
                    name="sar",
                    out_indices=sar_out_indices,
                    output_stride=output_stride,
                    drop_path_rate=drop_path_rate,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                )
        else:
            # SAR为参考流
            if sar_channels > 0:
                self.encoder_sar = create_encoder(
                    model_name=sar_model,
                    in_channels=sar_channels,
                    pretrained=sar_pretrained,
                    name="sar",
                    out_indices=None,
                    output_stride=output_stride,
                    drop_path_rate=drop_path_rate,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                )
            
            optical_out_indices = None
            if optical_channels > 0 and probe_opt_rd and probe_sar_rd:
                ref_rd = self.encoder_sar.feature_reductions
                opt_default_idx = probe_timm_default_out_indices(optical_model, optical_channels)
                optical_out_indices = match_out_indices_by_reduction(ref_rd, probe_opt_rd, opt_default_idx)
                logger.info(
                    f"Optical 按 reduction 对齐 indices: {optical_out_indices if optical_out_indices is not None else '默认层级'} "
                    f"(参考:{ref_rd}, 候选:{probe_opt_rd}, 候选默认out_indices:{opt_default_idx})"
                )
            
            if optical_channels > 0:
                self.encoder_optical = create_encoder(
                    model_name=optical_model,
                    in_channels=optical_channels,
                    pretrained=optical_pretrained,
                    name="optical",
                    out_indices=optical_out_indices,
                    output_stride=output_stride,
                    drop_path_rate=drop_path_rate,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                )

        # 保存每条流的通道与降采样
        self.optical_feature_channels = getattr(self.encoder_optical, "feature_channels", [])
        self.sar_feature_channels = getattr(self.encoder_sar, "feature_channels", [])
        self.optical_feature_reductions = getattr(self.encoder_optical, "feature_reductions", [])
        self.sar_feature_reductions = getattr(self.encoder_sar, "feature_reductions", [])

        # 若启用浅层细节编码器且主干为单尺度（通常 reductions 仅含 [16] 或长度为1），则补齐 4x/8x
        def _maybe_build_shallow(channels: int, name: str):
            nonlocal shallow_c2, shallow_c4, shallow_c8
            return ShallowDetailEncoder(channels, shallow_c2, shallow_c4, shallow_c8, norm=shallow_norm)

        if shallow_enabled:
            def _is_single_scale(rd: List[int]) -> bool:
                try:
                    return len(set(rd)) == 1
                except Exception:
                    return False

            # optical：若所有层的 reduction 相同（如 [16,16,16]），视为单尺度
            if self.encoder_optical is not None and _is_single_scale(self.optical_feature_reductions):
                self.shallow_opt = _maybe_build_shallow(optical_channels, "opt")
                # 组合为 2/4/8 来自 shallow，16 来自主干
                self.optical_feature_channels = [shallow_c2, shallow_c4, shallow_c8] + (self.optical_feature_channels[-1:] or [])
                self.optical_feature_reductions = [2, 4, 8] + (self.optical_feature_reductions[-1:] or [16])
                logger.info(f"启用浅层细节编码器(optical): c2={shallow_c2}, c4={shallow_c4}, c8={shallow_c8}; 输出层级=[2,4,8,16]")
            # sar
            if self.encoder_sar is not None and (_is_single_scale(self.sar_feature_reductions) or len(self.sar_feature_reductions) <= 2):
                self.shallow_sar = _maybe_build_shallow(sar_channels, "sar")
                self.sar_feature_channels = [shallow_c2, shallow_c4, shallow_c8] + (self.sar_feature_channels[-1:] or [])
                self.sar_feature_reductions = [2, 4, 8] + (self.sar_feature_reductions[-1:] or [16])
                logger.info(f"启用浅层细节编码器(sar): c2={shallow_c2}, c4={shallow_c4}, c8={shallow_c8}; 输出层级=[2,4,8,16]")

        # 对齐长度（按最短），并保存对齐后的 reductions（取参考流的前 n 个）
        if self.encoder_optical and self.encoder_sar:
            n = min(len(self.optical_feature_channels), len(self.sar_feature_channels))
            self.optical_feature_channels = self.optical_feature_channels[:n]
            self.sar_feature_channels = self.sar_feature_channels[:n]
            self.feature_channels = [o + s for o, s in zip(self.optical_feature_channels, self.sar_feature_channels)]

            # reductions：优先参考流
            ref_rd = self.optical_feature_reductions if ref_stream == "optical" else self.sar_feature_reductions
            self.feature_reductions = ref_rd[:n]
        elif self.encoder_optical:
            self.feature_channels = self.optical_feature_channels
            self.feature_reductions = self.optical_feature_reductions
        elif self.encoder_sar:
            self.feature_channels = self.sar_feature_channels
            self.feature_reductions = self.sar_feature_reductions
        else:
            raise RuntimeError("至少需要一个有效的编码器")

        # 日志：输出对齐后的关键信息，便于下游构建解码器
        logger.info(f"编码器元数据: channels={self.feature_channels}, reductions={self.feature_reductions}")
        logger.info(f"双路编码器就绪: optical={optical_channels}ch({optical_model}), sar={sar_channels}ch({sar_model})")

    def _probe_if_needed(self, model_name: str, channels: int) -> Tuple[List[int], List[int]]:
        """探测模型meta（如果需要，带缓存）"""
        return probe_timm_feature_meta(model_name, channels, out_indices=None) if channels > 0 else ([], [])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """前向推理，支持单模态和双模态输入"""
        result = {"optical": [], "sar": []}
        
        if 'image' in batch:
            # 单模态输入
            x = batch['image']
            if self.encoder_optical is not None:
                feats = list(self.encoder_optical(x))
                if self.shallow_opt is not None:
                    sh = self.shallow_opt(x)
                    feats = sh + feats[-1:]
                result["optical"] = feats[:len(self.feature_channels)]
            elif self.encoder_sar is not None:
                feats = list(self.encoder_sar(x))
                if self.shallow_sar is not None:
                    sh = self.shallow_sar(x)
                    feats = sh + feats[-1:]
                result["sar"] = feats[:len(self.feature_channels)]
            return result

        # 双模态输入
        if 'image_optical' in batch and self.encoder_optical is not None:
            x_opt = batch['image_optical']
            feats = list(self.encoder_optical(x_opt))
            if self.shallow_opt is not None:
                sh = self.shallow_opt(x_opt)
                feats = sh + feats[-1:]
            result["optical"] = feats[:len(self.feature_channels)]
        if 'image_sar' in batch and self.encoder_sar is not None:
            x_sar = batch['image_sar']
            feats = list(self.encoder_sar(x_sar))
            if self.shallow_sar is not None:
                sh = self.shallow_sar(x_sar)
                feats = sh + feats[-1:]
            result["sar"] = feats[:len(self.feature_channels)]
        
        return result