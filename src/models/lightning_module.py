"""
LightningModule（Training Orchestrator）：统一的多模态分割训练编排

职责
- 训练/验证/测试的编排，损失与指标管理
- 组件实例化与超参数记录

约定
- 组件使用 Hydra（_target_）在 __init__ 内实例化
- 实例化顺序：encoder → fusion → decoder（decoder 依赖 fusion 的 output_channels）
"""

from typing import Any, Dict, Optional, Union, Tuple, List
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
import hydra
from omegaconf import DictConfig
from torchmetrics import Accuracy, JaccardIndex, F1Score, Precision, Recall, Specificity
import logging

logger = logging.getLogger(__name__)


class MultiModalSegmentationModule(LightningModule):
    """
    多模态分割Lightning模块
    
    采用配置驱动设计，所有组件参数必须是配置字典(含_target_)，
    确保checkpoint完全自包含，支持一行代码加载。
    """
    
    def __init__(
        self,
        encoder: Dict[str, Any],
        fusion: Dict[str, Any],
        decoder: Dict[str, Any],
        loss_fn: Dict[str, Any],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        aux_loss_fn: Optional[Dict[str, Any]] = None,
        aux_loss_weight: float = 0.0,
        scheduler_config: Optional[Dict[str, Any]] = None,
        num_classes: int = 2,
        # 对齐正则（共享投影空间 + 统计/对比式对齐）
        alignment_enabled: bool = False,
        alignment_type: str = "nce",  # only nce retained
        alignment_target_weight: float = 0.02,
        alignment_layers: Optional[List[int]] = None,
        alignment_temperature: float = 0.1,
        alignment_num_samples: int = 256,
        # PatchNCE：正样本为同图像的局部邻域（半径 r，r=0 等价于原 InfoNCE）
        alignment_patch_radius: int = 1,
        # SigLIP：PatchNCE 使用 BCEWithLogits（多正样本独立建模）
        alignment_sigmoid: bool = True,
        # original: 同位置正样本；proximity: 邻域正样本；
        # exclude_changed: 邻域正样本，但仅使用标签为 0 的不变/背景 token
        alignment_mode: str = "proximity",
        align_cap_ratio: float = 0.3,
        alignment_cap_mode: str = "hard",
        alignment_independent_projection: bool = False,
        alignment_projection_dim: int = 128,
        alignment_exclude_boundaries: bool = True,
        alignment_warmup_epochs: int = 0,
        alignment_decay_start_epoch: int = -1,
        alignment_decay_end_epoch: int = -1,
        alignment_position_weight: float = 0.25,
        alignment_stop_gradient: bool = True,
        alignment_semantic_stop_gradient: bool = False,
        # 只在同一图像内构造配对矩阵：显存从 (B·K)^2 降到 B·K^2，可在更高分辨率层对齐
        alignment_in_image_only: bool = False,
        # 可靠性加权（RULE式）：用 detach 的同位置余弦相似度做软目标/权重，低可靠对应少参与对齐
        alignment_reliability_weight: bool = False,
        # 训练期模态dropout：以该概率将一路输入整幅置零（两路各占一半概率）
        modality_dropout_p: float = 0.0,
        # SMAGNet-lite（2026-09-04）：训练期像素级光学遮挡 + 可靠性门控掩膜 + SAR-only 共享解码器辅助路径
        pixel_optical_dropout_p: float = 0.0,      # 每张图被施加光学像素遮挡的概率
        pixel_optical_dropout_area: float = 0.3,   # 遮挡面积比例（1–3 个随机矩形）
        pixel_dropout_warmup_epochs: int = 3,      # 前 N 个 epoch 不遮挡
        pixel_dropout_ramp_epochs: int = 0,        # warmup 后再用 N 个 epoch 把遮挡概率从 0 线性升到 p（0=一步到位）
        sar_only_head_weight: float = 0.0,         # SAR-only 路径损失权重（共享解码器）
        optical_only_head_weight: float = 0.0,     # Optical-only 路径损失权重（共享解码器；对称配方）
        pixel_sar_dropout_p: float = 0.0,          # 每张图被施加 SAR 像素遮挡的概率（对称配方；面积与光学相同）
        pixel_dropout_exclusive: bool = False,     # True：同一张图不同时遮光学和 SAR（避免无信息区域强迫预测，稳定训练）
        optical_valid_mask_source: str = "none",   # none | zeros：测试期由光学全零像素推断有效掩膜（已知云掩膜）
        reliability_use_known_mask: bool = True,   # False：门控不乘已知掩膜，须自行从特征判断光学可靠性（mask-free 变体）
        # P2 融合表征一致性（跨条件对齐）：单模态路径的融合特征向 dual 融合特征（stop-grad）对齐，
        # loss = mean_l (1 - cos(F_single_l, sg(F_dual_l)))；权重 0 关闭。levels 为负索引列表。
        fused_consistency_weight: float = 0.0,
        fused_consistency_levels: Optional[List[int]] = None,
        # 测试期退化：none | sar_noise | optical_cloud | sar_shift
        test_degradation: str = "none",
        test_degradation_strength: float = 0.5,
        # 输入级拼接（单键 image）模型：SAR 通道在 image 中的起始位置；<0 表示不处理单键输入
        test_degradation_image_sar_start: int = -1,
        # 测试期退化的随机种子：<0 表示沿用全局 RNG（历史行为）；>=0 时用独立 CPU Generator，
        # 每次 test 开始重置，使遮挡位置 / 噪声在不同模型间可复现、可共享
        test_degradation_seed: int = -1,
        # optical_cloud 遮挡区域的填充值（归一化后的单位；0 = 历史行为，正值模拟亮云，负值模拟云影）
        test_degradation_fill: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        # 保存所有超参数，确保checkpoint自包含
        self.save_hyperparameters()
        
        # 实例化组件
        self.encoder = self._instantiate_component(encoder, "encoder")
        
        # 从编码器读取融合与解码所需meta
        feature_channels = getattr(self.encoder, 'feature_channels', [64, 256, 512, 1024, 2048])
        feature_reductions = getattr(self.encoder, 'feature_reductions', [4, 8, 16, 32])
        optical_channels = getattr(self.encoder, 'optical_feature_channels', None)
        sar_channels = getattr(self.encoder, 'sar_feature_channels', None)
        
        fusion_config = self._prepare_fusion_config(fusion, feature_channels, optical_channels, sar_channels)

        # 先实例化融合模块以获取实际输出通道，再据此实例化解码器
        self.fusion = self._instantiate_component(fusion_config, "fusion")

        fused_feature_channels = getattr(self.fusion, 'output_channels', feature_channels)
        decoder_config = self._prepare_decoder_config(decoder, fused_feature_channels, feature_reductions)
        self.decoder = self._instantiate_component(decoder_config, "decoder")

        # 记录融合后的通道（复现实验所需关键信息）
        self.hparams.fused_feature_channels = list(fused_feature_channels)
        self.loss_fn = self._instantiate_component(loss_fn, "loss_fn")
        
        # 辅助损失
        self.aux_loss_fn = self._instantiate_component(aux_loss_fn, "aux_loss_fn") if aux_loss_fn else None
        self.aux_loss_weight = float(aux_loss_weight)
        if self.aux_loss_weight > 0 and self.aux_loss_fn:
            fused_last_channels = self.fusion.output_channels[-1]
            # 重命名为更准确的名称，避免混淆（两个头都接收融合特征）
            self.aux_head_1 = nn.Conv2d(fused_last_channels, num_classes, 1)
            self.aux_head_2 = nn.Conv2d(fused_last_channels, num_classes, 1)

        # 重建路径已删除

        # 对齐配置（优先复用融合策略中的投影头，不额外创建独立投影）
        self.alignment_enabled = bool(alignment_enabled)
        self.alignment_type = str(alignment_type)
        self.alignment_target_weight = float(alignment_target_weight)
        self.alignment_layers = list(alignment_layers) if alignment_layers is not None else [-1]
        self.alignment_temperature = float(alignment_temperature)
        self.alignment_num_samples = int(alignment_num_samples)
        self.alignment_patch_radius = int(alignment_patch_radius)
        self.alignment_sigmoid = bool(alignment_sigmoid)
        self.alignment_mode = str(alignment_mode).lower()
        valid_alignment_modes = {
            "original",
            "proximity",
            "semantic_local",
            "soft_correspondence",
            "prototype",
            "exclude_changed",
            "distill",
            "disabled",
        }
        if self.alignment_mode not in valid_alignment_modes:
            raise ValueError(
                f"alignment_mode 必须是 {sorted(valid_alignment_modes)} 之一，"
                f"当前为 {alignment_mode!r}"
            )
        # disabled 是 alignment_enabled=false 的配置化别名；保留旧开关以兼容已有配置。
        if self.alignment_mode == "disabled":
            self.alignment_enabled = False

        # 对齐项限幅所需：主损的 EMA 与固定比例
        self.register_buffer("align_ema_main", torch.tensor(0.0), persistent=False)
        self.mra_ema_decay: float = 0.9
        self.align_cap_ratio: float = float(align_cap_ratio)
        self.alignment_cap_mode = str(alignment_cap_mode).lower()
        if self.alignment_cap_mode not in {"hard", "soft", "none"}:
            raise ValueError(
                "alignment_cap_mode 必须是 hard、soft 或 none，"
                f"当前为 {alignment_cap_mode!r}"
            )
        self.alignment_independent_projection = bool(
            alignment_independent_projection
        )
        self.alignment_projection_dim = int(alignment_projection_dim)
        self.alignment_exclude_boundaries = bool(alignment_exclude_boundaries)
        self.alignment_warmup_epochs = max(0, int(alignment_warmup_epochs))
        self.alignment_decay_start_epoch = int(alignment_decay_start_epoch)
        self.alignment_decay_end_epoch = int(alignment_decay_end_epoch)
        if (
            self.alignment_decay_start_epoch >= 0
            and self.alignment_decay_end_epoch
            <= self.alignment_decay_start_epoch
        ):
            raise ValueError(
                "alignment_decay_end_epoch 必须大于 start_epoch"
            )
        self.alignment_position_weight = float(alignment_position_weight)
        self.alignment_stop_gradient = bool(alignment_stop_gradient)
        self.alignment_semantic_stop_gradient = bool(
            alignment_semantic_stop_gradient
        )
        self.num_classes = int(num_classes)
        self.alignment_in_image_only = bool(alignment_in_image_only)
        self.alignment_reliability_weight = bool(alignment_reliability_weight)
        self.modality_dropout_p = float(modality_dropout_p)
        self.pixel_optical_dropout_p = float(pixel_optical_dropout_p)
        self.pixel_optical_dropout_area = float(pixel_optical_dropout_area)
        self.pixel_dropout_warmup_epochs = int(pixel_dropout_warmup_epochs)
        self.pixel_dropout_ramp_epochs = int(pixel_dropout_ramp_epochs)
        self.sar_only_head_weight = float(sar_only_head_weight)
        self.optical_only_head_weight = float(optical_only_head_weight)
        self.pixel_sar_dropout_p = float(pixel_sar_dropout_p)
        self.pixel_dropout_exclusive = bool(pixel_dropout_exclusive)
        self.fused_consistency_weight = float(fused_consistency_weight)
        self.fused_consistency_levels = list(fused_consistency_levels) if fused_consistency_levels else [-1, -2, -3]
        self._sar_valid_mask: Optional[torch.Tensor] = None
        self.optical_valid_mask_source = str(optical_valid_mask_source).lower()
        self.reliability_use_known_mask = bool(reliability_use_known_mask)
        self._optical_valid_mask: Optional[torch.Tensor] = None
        if not 0.0 <= self.modality_dropout_p < 1.0:
            raise ValueError("modality_dropout_p 必须在 [0, 1) 内")
        self.test_degradation = str(test_degradation).lower()
        if self.test_degradation not in {"none", "sar_noise", "optical_cloud", "sar_shift"}:
            raise ValueError(
                "test_degradation 必须是 none、sar_noise、optical_cloud 或 sar_shift，"
                f"当前为 {test_degradation!r}"
            )
        self.test_degradation_strength = float(test_degradation_strength)
        self.test_degradation_fill = float(test_degradation_fill)
        self.test_degradation_image_sar_start = int(test_degradation_image_sar_start)
        self.test_degradation_seed = int(test_degradation_seed)
        self._deg_gen: Optional[torch.Generator] = None
        self._align_keep_mask: Optional[torch.Tensor] = None
        # distill 模式：只有 SAR（学生）侧有可学习投影；光学（教师）保持原始特征并 detach。
        self.alignment_distill_proj: Optional[nn.ModuleList] = None
        if self.alignment_mode == "distill" and self.alignment_enabled:
            if not optical_channels or not sar_channels:
                raise ValueError("alignment_mode=distill 需要双流 encoder 通道信息")
            self.alignment_distill_proj = nn.ModuleList(
                [
                    nn.Identity()
                    if int(s) == int(o)
                    else nn.Conv2d(int(s), int(o), kernel_size=1, bias=False)
                    for o, s in zip(optical_channels, sar_channels)
                ]
            )
        self.alignment_project_opt: Optional[nn.ModuleList] = None
        self.alignment_project_sar: Optional[nn.ModuleList] = None
        if self.alignment_independent_projection:
            if not optical_channels or not sar_channels:
                raise ValueError("独立 alignment projection 需要双流 encoder 通道信息")
            if self.alignment_projection_dim <= 0:
                raise ValueError("alignment_projection_dim 必须为正整数")
            self.alignment_project_opt = nn.ModuleList(
                [
                    self._make_alignment_projection(int(ch))
                    for ch in optical_channels
                ]
            )
            self.alignment_project_sar = nn.ModuleList(
                [self._make_alignment_projection(int(ch)) for ch in sar_channels]
            )

        # 重建头已删除
        
        # 初始化评估指标
        self._init_metrics(num_classes)

    def _make_alignment_projection(self, in_channels: int) -> nn.Module:
        """构造只服务于对齐损失的投影头，避免改写融合投影空间。"""
        dim = self.alignment_projection_dim
        return nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
    
    def _instantiate_component(self, component, name: str):
        """实例化组件，要求传入配置字典而非实例"""
        if component is None:
            return None
            
        if not isinstance(component, (dict, DictConfig)):
            raise ValueError(
                f"组件 '{name}' 必须是配置字典(dict/DictConfig)，当前类型: {type(component)}。"
                f"请在实验配置中使用 _target_ 指定类路径。"
            )
        
        # 特殊处理FocalLoss的alpha参数
        if ('FocalLoss' in str(component.get('_target_', '')) and 'alpha' in component):
            # 转换为原生类型
            from omegaconf import OmegaConf
            component_native = OmegaConf.to_container(component, resolve=True)
            
            # 处理alpha参数
            alpha_value = component_native['alpha']
            if isinstance(alpha_value, list):
                # 根据源码分析：MULTICLASS_MODE下alpha应为单个float值
                focal_alpha = alpha_value[1] if len(alpha_value) > 1 else alpha_value[0]
                component_native['alpha'] = focal_alpha
                logger.info(f"FocalLoss alpha参数转换: {alpha_value} -> {focal_alpha}")
            
            return hydra.utils.instantiate(component_native)
        
        # 标准的Hydra实例化
        return hydra.utils.instantiate(component)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Normalize state-dict keys saved from ``torch.compile`` wrappers.

        Older runs compiled encoder/fusion/decoder independently, which added
        ``._orig_mod.`` to checkpoint keys. Current evaluation constructs the
        equivalent uncompiled modules, so the wrapper segment must be removed
        before Lightning performs strict state-dict loading.
        """
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, dict):
            return
        remapped = {}
        changed = 0
        for key, value in state_dict.items():
            normalized = key.replace("._orig_mod.", ".")
            if normalized.startswith("_orig_mod."):
                normalized = normalized[len("_orig_mod.") :]
            changed += int(normalized != key)
            remapped[normalized] = value
        if changed:
            checkpoint["state_dict"] = remapped
            logger.info("已规范化 %d 个 torch.compile checkpoint 参数名", changed)
    
    def _prepare_fusion_config(self, config, feature_channels, optical_channels, sar_channels):
        """为融合配置添加特征通道数"""
        if not isinstance(config, (dict, DictConfig)):
            return config
            
        cfg = dict(config)
        cfg['feature_channels'] = feature_channels
        if optical_channels and sar_channels:
            cfg['optical_channels'] = optical_channels
            cfg['sar_channels'] = sar_channels
        return cfg
    
    def _prepare_decoder_config(self, config: Dict[str, Any], feature_channels: List[int], feature_reductions: List[int]) -> Dict[str, Any]:
        """为解码器准备配置，包含feature_reductions信息"""
        cfg = dict(config)
        cfg['feature_channels'] = feature_channels
        cfg['feature_reductions'] = feature_reductions
        return cfg
    
    def _init_metrics(self, num_classes: int):
        """初始化评估指标"""
        # 基础指标
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.test_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        # 按类别的IoU（用于记录 water_iou 等）
        self.train_iou_pc = JaccardIndex(task="multiclass", num_classes=num_classes, average=None)
        self.val_iou_pc = JaccardIndex(task="multiclass", num_classes=num_classes, average=None)
        self.test_iou_pc = JaccardIndex(task="multiclass", num_classes=num_classes, average=None)
        
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # 水文关键指标：精确率 (宏平均，用于整体评估)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        
        # 水文关键指标：召回率 (按类别，用于计算洪水类漏检率)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        
        # 水文关键指标：特异性 (按类别，用于计算背景类虚警率)
        self.train_specificity = Specificity(task="multiclass", num_classes=num_classes, average=None)
        self.val_specificity = Specificity(task="multiclass", num_classes=num_classes, average=None)
        self.test_specificity = Specificity(task="multiclass", num_classes=num_classes, average=None)
        
        logger.info("评估指标初始化成功: IoU, F1, Accuracy, Precision, Recall, Specificity")
        logger.info("水文指标支持: 洪水漏检率=1-洪水类召回率, 背景虚警率=1-背景类特异性")
    
    def _get_input_size(self, batch: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        """获取输入图像的空间尺寸 (H, W)"""
        if 'image' in batch:
            return batch['image'].shape[-2:]
        elif 'image_optical' in batch:
            return batch['image_optical'].shape[-2:]
        elif 'image_sar' in batch:
            return batch['image_sar'].shape[-2:]
        else:
            raise ValueError("batch中未找到有效的图像数据")

    def _resize_to_target(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """将logits双线性插值到与target相同的空间大小。"""
        if logits.shape[2:] == target.shape[1:]:
            return logits
        return F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

    def _compute_valid_preds_targets(self, preds: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """过滤无效标签(-1)，返回(valid_mask, valid_preds, valid_targets)。"""
        valid_mask = masks != -1
        if valid_mask.sum() == 0:
            return valid_mask, preds.new_empty(0), masks.new_empty(0)
        return valid_mask, preds[valid_mask], masks[valid_mask]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            batch: 输入batch
                - 单模态: {"image": tensor, "mask": tensor}
                - 双模态: {"image_optical": tensor, "image_sar": tensor, "mask": tensor}
        
        Returns:
            预测结果字典
        """
        # 1. 特征提取
        raw_features = self.encoder(batch)
        
        # 2. 特征融合（将空列表标准化为None，避免误走双模态分支导致空融合）
        optical_feats = raw_features.get("optical")
        sar_feats = raw_features.get("sar")
        if isinstance(optical_feats, (list, tuple)) and len(optical_feats) == 0:
            optical_feats = None
        if isinstance(sar_feats, (list, tuple)) and len(sar_feats) == 0:
            sar_feats = None

        # 统一入口：直接标准融合（策略内 DropBranch 已移除）
        valid_mask = self._optical_valid_mask if (optical_feats is not None and sar_feats is not None and self.reliability_use_known_mask) else None
        if valid_mask is not None:
            fused_features = self.fusion(optical_features=optical_feats, sar_features=sar_feats, optical_valid_mask=valid_mask)
        else:
            fused_features = self.fusion(optical_features=optical_feats, sar_features=sar_feats)
        
        # 3. 解码（传递原图尺寸用于精确插值）
        input_size = self._get_input_size(batch)
        main_logits = self.decoder(fused_features, input_size=input_size)
        sar_only_logits = None
        optical_only_logits = None
        optical_only_fused = None
        sar_only_fused = None
        if self.training and self.optical_only_head_weight > 0 and optical_feats is not None and hasattr(self.fusion, "optical_only"):
            bn_layers = [m for m in self.decoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            for m in bn_layers:
                m.eval()
            try:
                optical_only_fused = self.fusion.optical_only(list(optical_feats))
                optical_only_logits = self.decoder(optical_only_fused, input_size=input_size)
            finally:
                for m in bn_layers:
                    m.train()
        if self.training and self.sar_only_head_weight > 0 and sar_feats is not None and hasattr(self.fusion, "sar_only"):
            # 共享解码器的第二次前向不得污染 BN 的 running statistics（否则验证/测试时 BN 统计是
            # 融合特征与 SAR-only 特征的混合，val 会大幅退化）：SAR-only 路径下临时把解码器的 BN 置为 eval。
            bn_layers = [m for m in self.decoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
            for m in bn_layers:
                m.eval()
            try:
                sar_only_fused = self.fusion.sar_only(list(sar_feats))
                sar_only_logits = self.decoder(sar_only_fused, input_size=input_size)
            finally:
                for m in bn_layers:
                    m.train()
        
        # 4. 准备输出
        num_levels = len(getattr(self.fusion, 'output_channels', getattr(self.encoder, 'feature_channels', [])))
        return {
            "main_logits": main_logits,
            "optical_features": raw_features.get("optical", [None] * num_levels),
            "sar_features": raw_features.get("sar", [None] * num_levels),
            "fused_features": fused_features,
            "sar_only_logits": sar_only_logits,
            "optical_only_logits": optical_only_logits,
            "sar_only_fused": sar_only_fused,
            "optical_only_fused": optical_only_fused,
        }

    def _apply_pixel_optical_dropout(self, batch: Dict[str, torch.Tensor]) -> None:
        """训练期像素级光学遮挡（模拟云/缺测）：每张图以 p 的概率抹掉 1–3 个随机矩形（总面积≈area），
        并生成光学有效掩膜（1=有效，0=遮挡）供可靠性门控使用。测试期若 optical_valid_mask_source=zeros，
        由光学全零像素推断掩膜（等价于已知云掩膜）。"""
        self._optical_valid_mask = None
        self._sar_valid_mask = None
        if "image_optical" not in batch or "image_sar" not in batch:
            return
        opt = batch["image_optical"]
        b, _, h, w = opt.shape
        if self.training:
            if self.pixel_optical_dropout_p <= 0 or self.current_epoch < self.pixel_dropout_warmup_epochs:
                if getattr(self.fusion, "strategy", None) is not None and getattr(self.fusion.strategy, "reliability_gate", None) is not None:
                    self._optical_valid_mask = torch.ones((b, 1, h, w), device=opt.device, dtype=opt.dtype)
                return
            p_eff = self.pixel_optical_dropout_p
            if self.pixel_dropout_ramp_epochs > 0:
                frac = (self.current_epoch - self.pixel_dropout_warmup_epochs + 1) / float(self.pixel_dropout_ramp_epochs)
                p_eff = self.pixel_optical_dropout_p * float(min(1.0, max(0.0, frac)))
            mask = torch.ones((b, 1, h, w), device=opt.device, dtype=opt.dtype)
            apply = torch.rand(b, device=opt.device) < p_eff
            for i in torch.nonzero(apply).flatten().tolist():
                n_rect = int(torch.randint(1, 4, (1,)).item())
                area_each = self.pixel_optical_dropout_area / n_rect
                for _ in range(n_rect):
                    ar = float(torch.empty(1).uniform_(0.5, 2.0).item())
                    rh = max(1, min(h, int(round(h * math.sqrt(area_each * ar)))))
                    rw = max(1, min(w, int(round(w * math.sqrt(area_each / ar)))))
                    y0 = int(torch.randint(0, h - rh + 1, (1,)).item())
                    x0 = int(torch.randint(0, w - rw + 1, (1,)).item())
                    mask[i, :, y0:y0 + rh, x0:x0 + rw] = 0
            batch["image_optical"] = opt * mask
            self._optical_valid_mask = mask.detach()
            # 对称配方：以 pixel_sar_dropout_p 的概率对 SAR 施加同样形状的随机矩形遮挡（与光学遮挡独立采样）
            self._sar_valid_mask = None
            if self.pixel_sar_dropout_p > 0:
                p_sar = self.pixel_sar_dropout_p * (p_eff / self.pixel_optical_dropout_p if self.pixel_optical_dropout_p > 0 else 1.0)
                sar = batch["image_sar"]
                smask = torch.ones((b, 1, h, w), device=sar.device, dtype=sar.dtype)
                apply_s = torch.rand(b, device=sar.device) < p_sar
                if self.pixel_dropout_exclusive:
                    # 互斥：已遮光学的图不再遮 SAR，保证每张图至少一个模态完整
                    apply_s = apply_s & ~apply
                for i in torch.nonzero(apply_s).flatten().tolist():
                    n_rect = int(torch.randint(1, 4, (1,)).item())
                    area_each = self.pixel_optical_dropout_area / n_rect
                    for _ in range(n_rect):
                        ar = float(torch.empty(1).uniform_(0.5, 2.0).item())
                        rh = max(1, min(h, int(round(h * math.sqrt(area_each * ar)))))
                        rw = max(1, min(w, int(round(w * math.sqrt(area_each / ar)))))
                        y0 = int(torch.randint(0, h - rh + 1, (1,)).item())
                        x0 = int(torch.randint(0, w - rw + 1, (1,)).item())
                        smask[i, :, y0:y0 + rh, x0:x0 + rw] = 0
                batch["image_sar"] = sar * smask
                self._sar_valid_mask = smask.detach()
        elif self.optical_valid_mask_source == "zeros":
            self._optical_valid_mask = (opt.abs().sum(dim=1, keepdim=True) > 0).to(opt.dtype)
        elif getattr(self.fusion, "strategy", None) is not None and getattr(self.fusion.strategy, "reliability_gate", None) is not None:
            self._optical_valid_mask = torch.ones((b, 1, h, w), device=opt.device, dtype=opt.dtype)
    
    def _apply_modality_dropout(self, batch: Dict[str, torch.Tensor]) -> None:
        """训练期按样本置零一路输入，并记录未被置零的样本供对齐损失使用。"""
        self._align_keep_mask = None
        if not (self.training and self.modality_dropout_p > 0):
            return
        if "image_optical" not in batch or "image_sar" not in batch:
            return
        opt = batch["image_optical"]
        sar = batch["image_sar"]
        u = torch.rand(opt.shape[0], device=opt.device)
        drop_opt = u < (self.modality_dropout_p / 2.0)
        drop_sar = (u >= (self.modality_dropout_p / 2.0)) & (u < self.modality_dropout_p)
        keep_opt = (~drop_opt).to(opt.dtype).view(-1, 1, 1, 1)
        keep_sar = (~drop_sar).to(sar.dtype).view(-1, 1, 1, 1)
        batch["image_optical"] = opt * keep_opt
        batch["image_sar"] = sar * keep_sar
        self._align_keep_mask = ~(drop_opt | drop_sar)

    def _deg_randn_like(self, x: torch.Tensor) -> torch.Tensor:
        if self._deg_gen is None:
            return torch.randn_like(x)
        return torch.randn(x.shape, generator=self._deg_gen, dtype=torch.float32).to(device=x.device, dtype=x.dtype)

    def _deg_randint(self, high: int) -> int:
        if self._deg_gen is None:
            return int(torch.randint(0, high, (1,)).item())
        return int(torch.randint(0, high, (1,), generator=self._deg_gen).item())

    def on_test_epoch_start(self) -> None:
        if self.test_degradation_seed >= 0:
            self._deg_gen = torch.Generator().manual_seed(self.test_degradation_seed)
        else:
            self._deg_gen = None

    def _apply_test_degradation(self, batch: Dict[str, torch.Tensor]) -> None:
        """测试期退化：SAR加性噪声（按通道std缩放）或光学随机云块遮挡（置为归一化均值0）。"""
        if self.test_degradation == "none":
            return
        s = self.test_degradation_strength
        if self.test_degradation == "sar_noise" and "image_sar" in batch:
            sar = batch["image_sar"]
            std = sar.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6).to(sar.dtype)
            batch["image_sar"] = sar + self._deg_randn_like(sar) * std * s
        elif self.test_degradation == "sar_noise" and "image" in batch and self.test_degradation_image_sar_start >= 0:
            # 输入级拼接模型（单一 image 键）：仅对 SAR 通道（从 sar_start 起）加噪，与双流模型的 sar_noise 定义一致
            img = batch["image"]; c0 = self.test_degradation_image_sar_start
            sar = img[:, c0:]
            std = sar.float().std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6).to(sar.dtype)
            batch["image"] = torch.cat([img[:, :c0], sar + self._deg_randn_like(sar) * std * s], dim=1)
        elif self.test_degradation == "optical_cloud" and ("image_optical" in batch or ("image" in batch and self.test_degradation_image_sar_start >= 0)):
            # 输入级拼接模型只有单一 image 键：光学为前 sar_start 个通道，只对这些通道施加云块
            single_key = "image_optical" not in batch
            if single_key:
                img = batch["image"]; c0 = self.test_degradation_image_sar_start
                opt = img[:, :c0]
            else:
                opt = batch["image_optical"]
            b, _, h, w = opt.shape
            # 每张图一个覆盖约 s 比例面积的矩形云块，位置随机
            side_h = max(1, int(round(h * math.sqrt(s))))
            side_w = max(1, int(round(w * math.sqrt(s))))
            mask = torch.ones((b, 1, h, w), device=opt.device, dtype=opt.dtype)
            for i in range(b):
                y0 = self._deg_randint(h - side_h + 1)
                x0 = self._deg_randint(w - side_w + 1)
                mask[i, :, y0 : y0 + side_h, x0 : x0 + side_w] = 0
            fill = float(getattr(self, "test_degradation_fill", 0.0))
            occluded = opt * mask + fill * (1.0 - mask)
            if single_key:
                batch["image"] = torch.cat([occluded, img[:, c0:]], dim=1)
            else:
                batch["image_optical"] = occluded
        elif self.test_degradation == "sar_shift":
            # 模拟光学/SAR残余配准误差：SAR整体平移 s 像素（对角方向，边缘复制填充），
            # 标签与光学不动。s 取整数像素，如 1/2/4/8。
            k = int(round(s))
            if k <= 0:
                return

            def _shift(x: torch.Tensor) -> torch.Tensor:
                h, w = x.shape[-2:]
                return torch.nn.functional.pad(x, (k, k, k, k), mode="replicate")[..., 0:h, 0:w]

            if "image_sar" in batch:
                batch["image_sar"] = _shift(batch["image_sar"])
            elif "image" in batch and self.test_degradation_image_sar_start >= 0:
                # 输入级拼接模型：只平移 image 中从 sar_start 起的 SAR 通道
                img = batch["image"]
                c0 = self.test_degradation_image_sar_start
                batch["image"] = torch.cat([img[:, :c0], _shift(img[:, c0:])], dim=1)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """共享的训练/验证/测试步骤"""
        if stage == "train":
            self._apply_modality_dropout(batch)
        elif stage == "test":
            self._apply_test_degradation(batch)
        self._apply_pixel_optical_dropout(batch)
        outputs = self(batch)
        masks = batch["mask"]
        
        # 调整输出尺寸以匹配标签
        main_logits = self._resize_to_target(outputs["main_logits"], masks)
        
        # 主损失（统一调用多类损失）
        main_loss = self.loss_fn(main_logits, masks)
        total_loss = main_loss
        # 累计各项（原始与加权）用于日志
        device = masks.device
        aux_raw = torch.tensor(0.0, device=device)
        aux_scaled = torch.tensor(0.0, device=device)
        align_raw = torch.tensor(0.0, device=device)
        align_scaled = torch.tensor(0.0, device=device)
        # 对齐项是否计算的标志，需在任何分支之外先初始化，避免未定义引用
        align_computed = False
        
        # 训练/验证记录主损；测试阶段不再记录 loss_main 以精简最终表格
        if stage != "test":
            self.log(f"{stage}/loss_main", main_loss, on_step=True, on_epoch=True, prog_bar=True)
        # SAR-only 共享解码器辅助路径（SMAGNet 双路径）：保证光学缺失时不劣于纯 SAR
        if self.training and self.optical_only_head_weight > 0 and outputs.get("optical_only_logits") is not None:
            opt_logits = self._resize_to_target(outputs["optical_only_logits"], masks)
            optical_only_loss = self.loss_fn(opt_logits, masks)
            total_loss = total_loss + self.optical_only_head_weight * optical_only_loss
            self.log(f"{stage}/loss_optical_only", optical_only_loss, on_step=False, on_epoch=True)
        if self.training and self.sar_only_head_weight > 0 and outputs.get("sar_only_logits") is not None:
            sar_logits = self._resize_to_target(outputs["sar_only_logits"], masks)
            sar_only_loss = self.loss_fn(sar_logits, masks)
            total_loss = total_loss + self.sar_only_head_weight * sar_only_loss
            self.log(f"{stage}/loss_sar_only", sar_only_loss, on_step=False, on_epoch=True)
        # P2 融合表征一致性：单模态融合特征 → dual 融合特征（stop-grad），只在遮挡/缺失训练下有意义
        if self.training and self.fused_consistency_weight > 0:
            cons_terms = []
            dual_fused = outputs.get("fused_features")
            for key in ("sar_only_fused", "optical_only_fused"):
                single = outputs.get(key)
                if single is None or dual_fused is None:
                    continue
                n_lv = min(len(single), len(dual_fused))
                for idx in self.fused_consistency_levels:
                    i = idx if idx >= 0 else n_lv + idx
                    if not 0 <= i < n_lv:
                        continue
                    cos = F.cosine_similarity(single[i].float(), dual_fused[i].detach().float(), dim=1, eps=1e-6)
                    cons_terms.append((1.0 - cos).mean())
            if cons_terms:
                cons_loss = torch.stack(cons_terms).mean()
                total_loss = total_loss + self.fused_consistency_weight * cons_loss
                self.log(f"{stage}/loss_fused_consistency", cons_loss, on_step=False, on_epoch=True)
        
        # 更新主损 EMA（用于对齐项限幅，避免主导）
        with torch.no_grad():
            if self.align_ema_main.device != main_loss.device:
                self.align_ema_main = self.align_ema_main.to(main_loss.device)
            if float(self.align_ema_main.item()) == 0.0:
                self.align_ema_main.copy_(main_loss.detach())
            else:
                self.align_ema_main.copy_(
                    self.mra_ema_decay * self.align_ema_main + (1.0 - self.mra_ema_decay) * main_loss.detach()
                )
        
        # 辅助损失（仅当权重>0 且配置了函数时启用）
        if self.training and self.aux_loss_weight > 0 and self.aux_loss_fn:
            aux_loss = self._compute_aux_loss(outputs, masks, stage)
            aux_raw = aux_loss
            aux_scaled = self.aux_loss_weight * aux_loss
            total_loss = total_loss + aux_scaled
        
        # 重建损失已移除
        
        # 对齐正则（仅训练，且仅在双模态时启用，避免单模态任务出现全0曲线）
        if self.training and self.alignment_enabled:
            opt_feats: List[torch.Tensor] = outputs.get("optical_features", [])  # type: ignore[assignment]
            sar_feats: List[torch.Tensor] = outputs.get("sar_features", [])      # type: ignore[assignment]
            has_dual_modal = (('image_optical' in batch and 'image_sar' in batch) or 
                               (isinstance(opt_feats, (list, tuple)) and isinstance(sar_feats, (list, tuple)) and len(opt_feats) > 0 and len(sar_feats) > 0))
            if has_dual_modal:
                align_loss, align_stats = self._compute_alignment_loss(
                    opt_feats, sar_feats, target=masks
                )
                lambda_align = self._get_alignment_schedule()
                align_raw = align_loss
                align_scaled_raw = torch.tensor(lambda_align, device=device) * align_loss
                align_scaled, cap = self._apply_alignment_cap(align_scaled_raw)
                total_loss = total_loss + align_scaled
                align_computed = True
                self.log(f"{stage}/loss_alignment", align_loss, on_step=True, on_epoch=True)
                self.log(f"{stage}/lambda_align", torch.tensor(lambda_align, device=masks.device), on_step=False, on_epoch=True)
                if isinstance(align_stats, dict):
                    for k, v in align_stats.items():
                        if isinstance(v, torch.Tensor):
                            self.log(f"{stage}/{k}", v, on_step=False, on_epoch=True)
            else:
                # 单模态或数据缺少一路：对齐不计算也不记录（保持曲线缺省）
                align_computed = False

        # 加权项与占比：仅在 train 记录；val/test 不记录这些派生指标
        if stage == "train":
            self.log(f"{stage}/loss_main_scaled", main_loss, on_step=False, on_epoch=True)
            if self.aux_loss_weight > 0 and self.aux_loss_fn:
                self.log(f"{stage}/loss_aux_scaled", aux_scaled, on_step=False, on_epoch=True)
            if self.alignment_enabled and ('align_scaled' in locals()) and align_computed:
                self.log(f"{stage}/loss_align_scaled", align_scaled, on_step=False, on_epoch=True)
                if 'align_scaled_raw' in locals():
                    self.log(f"{stage}/loss_align_scaled_raw", align_scaled_raw, on_step=False, on_epoch=True)
                if 'cap' in locals():
                    # 当禁用限幅时，cap 输出为 +inf（便于识别）
                    self.log(f"{stage}/loss_align_cap", cap, on_step=False, on_epoch=True)

            total_scaled = main_loss + aux_scaled + (align_scaled if ('align_scaled' in locals()) and align_computed else torch.tensor(0.0, device=device))
            eps = torch.tensor(1e-12, device=device)
            denom = torch.maximum(total_scaled.detach(), eps)
            self.log(f"{stage}/loss_main_pct", (main_loss.detach() / denom), on_step=False, on_epoch=True)
            self.log(f"{stage}/loss_aux_pct", (aux_scaled.detach() / denom), on_step=False, on_epoch=True)
            self.log(f"{stage}/loss_align_pct", (align_scaled.detach() / denom), on_step=False, on_epoch=True)

        # 不记录注意力熵：仅保留 loss 及其占比
        
        # 计算指标
        preds = torch.argmax(main_logits.detach(), dim=1)
        self._update_metrics(preds, masks, stage)
        # 事件级 IoU 累计（仅验证/测试，且 batch 含 activation 才处理）
        self._accumulate_event_iou(batch, preds, masks, stage)
        if stage == "test":
            self._accumulate_test_per_image(batch, preds, masks)
        
        return total_loss
    
    def _compute_aux_loss(self, outputs, masks, stage):
        """计算辅助损失"""
        fused_features = outputs.get("fused_features")
        if not fused_features or not self.aux_loss_fn:
            return 0.0
        
        fused_last = fused_features[-1]
        aux_losses = []
        
        # 计算辅助损失（两个分割头）
        for head_name in ['aux_head_1', 'aux_head_2']:
            if hasattr(self, head_name):
                aux_head = getattr(self, head_name)
                aux_logits = aux_head(fused_last)
                aux_logits = F.interpolate(
                    aux_logits, size=masks.shape[1:], 
                    mode='bilinear', align_corners=False
                )
                aux_loss = self.aux_loss_fn(aux_logits, masks)
                aux_losses.append(aux_loss)
                self.log(f"{stage}/aux_loss_{head_name.split('_')[-1]}", aux_loss, on_step=True, on_epoch=True)
        
        if aux_losses:
            avg_aux_loss = sum(aux_losses) / len(aux_losses)
            self.log(f"{stage}/loss_aux", avg_aux_loss, on_step=True, on_epoch=True)
            return avg_aux_loss
        
        return 0.0

    # 重建路径已删除

    # ----------------- MRA：由融合策略托管（此处无实现） -----------------

    # 重建相关的调度与度量已删除

    # ----------------- 对齐正则 -----------------
    def _get_alignment_schedule(self) -> float:
        """返回对齐权重：可选线性 warm-up 与后期 cosine decay。"""
        target = float(self.alignment_target_weight)
        warmup = 1.0
        if self.alignment_warmup_epochs > 0:
            warmup = min(
                1.0,
                float(self.current_epoch + 1)
                / float(self.alignment_warmup_epochs),
            )
        decay = 1.0
        start = self.alignment_decay_start_epoch
        end = self.alignment_decay_end_epoch
        epoch = int(self.current_epoch)
        if start >= 0 and end > start:
            if epoch >= end:
                decay = 0.0
            elif epoch >= start:
                progress = float(epoch - start) / float(end - start)
                decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return target * warmup * decay

    def _apply_alignment_cap(
        self, scaled_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """限制对齐项幅度；soft 模式饱和后仍保留连续梯度。"""
        if self.align_cap_ratio <= 0 or self.alignment_cap_mode == "none":
            cap = scaled_loss.new_tensor(float("inf"))
            return scaled_loss, cap
        cap = self.align_cap_ratio * self.align_ema_main
        if self.alignment_cap_mode == "hard":
            return torch.minimum(scaled_loss, cap), cap
        safe_cap = cap.clamp_min(torch.finfo(scaled_loss.dtype).eps)
        return safe_cap * torch.tanh(scaled_loss / safe_cap), cap

    def _sample_spatial_positions(self, h: int, w: int, k: int, device: torch.device) -> torch.Tensor:
        k = int(min(k, h * w))
        idx = torch.randperm(h * w, device=device)[:k]
        return idx

    def _compute_alignment_loss(
        self,
        opt_feats: List[torch.Tensor],
        sar_feats: List[torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats: Dict[str, torch.Tensor] = {}
        if getattr(self, "alignment_mode", "proximity") == "distill":
            return self._distill_alignment_loss(opt_feats, sar_feats, target)
        # 修正版可使用独立投影头，避免对齐梯度改写融合所用投影空间。
        if (
            getattr(self, "alignment_independent_projection", False)
            and getattr(self, "alignment_project_opt", None) is not None
            and getattr(self, "alignment_project_sar", None) is not None
        ):
            proj_opt = self.alignment_project_opt
            proj_sar = self.alignment_project_sar
        else:
            strategy = getattr(self.fusion, "strategy", None)
            proj_opt = getattr(strategy, "project_opt", None)
            proj_sar = getattr(strategy, "project_sar", None)
        if proj_opt is None or proj_sar is None:
            raise RuntimeError("alignment_enabled 仅支持带统一投影头的策略（如 xattn）。当前融合策略不支持对齐，请改用 xattn 或关闭 alignment_enabled。")

        num_levels = min(len(opt_feats), len(sar_feats), len(proj_opt), len(proj_sar))
        # 使用 alignment_layers 选择层级
        sel_indices: List[int] = []
        for idx in self.alignment_layers:
            i = idx if idx >= 0 else (num_levels + idx)
            if 0 <= i < num_levels:
                sel_indices.append(i)
        sel_indices = sorted(set(sel_indices))
        if len(sel_indices) == 0:
            device = (opt_feats[-1].device if len(opt_feats) else sar_feats[-1].device)
            return torch.tensor(0.0, device=device), stats

        total = 0.0
        for i in sel_indices:
            fo = opt_feats[i]
            fs = sar_feats[i]
            # 空间对齐：为节省显存，统一缩放到两者的最小分辨率（min），而非放大到最大
            if fo.shape[2:] != fs.shape[2:]:
                target_h = min(fo.shape[2], fs.shape[2])
                target_w = min(fo.shape[3], fs.shape[3])
                if fo.shape[2:] != (target_h, target_w):
                    fo = F.interpolate(fo, size=(target_h, target_w), mode='bilinear', align_corners=False)
                if fs.shape[2:] != (target_h, target_w):
                    fs = F.interpolate(fs, size=(target_h, target_w), mode='bilinear', align_corners=False)
            # 通道投影到融合使用的统一宽度
            xo = proj_opt[i](fo)
            ys = proj_sar[i](fs)
            # original 等价于旧 InfoNCE 正对（r=0）；proximity 保留旧默认邻域行为。
            mode = getattr(self, "alignment_mode", "proximity")
            r = 0 if mode == "original" else max(
                0, int(getattr(self, "alignment_patch_radius", 0))
            )
            valid_mask = None
            semantic_labels = None
            if mode == "exclude_changed":
                if target is None:
                    raise ValueError("alignment_mode=exclude_changed 需要 batch 中的 mask 标签")
                target_mask = target
                if target_mask.ndim == 4 and target_mask.shape[1] == 1:
                    target_mask = target_mask[:, 0]
                if target_mask.ndim != 3:
                    raise ValueError(
                        "alignment target 应为 [B,H,W] 或 [B,1,H,W]，"
                        f"当前形状为 {tuple(target.shape)}"
                    )
                valid_mask = F.interpolate(
                    (target_mask == 0).to(dtype=torch.float32).unsqueeze(1),
                    size=xo.shape[2:],
                    mode="nearest",
                ).squeeze(1).bool()
            elif mode in {"semantic_local", "soft_correspondence", "prototype"}:
                if target is None:
                    raise ValueError(f"alignment_mode={mode} 需要 batch 中的 mask 标签")
                target_mask = target
                if target_mask.ndim == 4 and target_mask.shape[1] == 1:
                    target_mask = target_mask[:, 0]
                if target_mask.ndim != 3:
                    raise ValueError(
                        "alignment target 应为 [B,H,W] 或 [B,1,H,W]，"
                        f"当前形状为 {tuple(target.shape)}"
                    )
                semantic_labels = F.interpolate(
                    target_mask.to(dtype=torch.float32).unsqueeze(1),
                    size=xo.shape[2:],
                    mode="nearest",
                ).squeeze(1).long()
                valid_mask = semantic_labels >= 0
                if getattr(self, "alignment_exclude_boundaries", True):
                    labels_float = semantic_labels.clamp_min(0).float().unsqueeze(1)
                    local_max = F.max_pool2d(labels_float, 3, stride=1, padding=1)
                    local_min = -F.max_pool2d(
                        -labels_float, 3, stride=1, padding=1
                    )
                    interior = (local_max == local_min).squeeze(1)
                    valid_mask = valid_mask & interior
            # 像素级光学遮挡区域不参与对齐：遮挡处的光学特征不代表真实观测（否则会把 SAR 特征对齐到"零光学"）。
            for vm_name in ("_optical_valid_mask", "_sar_valid_mask"):
                ovm = getattr(self, vm_name, None)
                if ovm is not None and bool((ovm < 1).any()):
                    ovm_map = F.interpolate(ovm.to(dtype=torch.float32), size=xo.shape[2:], mode="area").squeeze(1) >= 0.999
                    valid_mask = ovm_map if valid_mask is None else (valid_mask & ovm_map)
            keep = getattr(self, "_align_keep_mask", None)
            if keep is not None and bool((~keep).any()):
                # 被模态dropout置零的样本不参与对齐：其一路特征不代表真实观测。
                keep_map = keep.to(xo.device).view(-1, 1, 1).expand(
                    xo.shape[0], xo.shape[2], xo.shape[3]
                )
                valid_mask = keep_map if valid_mask is None else (valid_mask & keep_map)
            if mode == "soft_correspondence":
                loss_i, st = self._soft_correspondence_loss(
                    xo,
                    ys,
                    semantic_labels=semantic_labels,
                    valid_mask=valid_mask,
                    radius=r,
                )
            elif mode == "prototype":
                loss_i, st = self._prototype_alignment_loss(
                    xo,
                    ys,
                    semantic_labels=semantic_labels,
                    valid_mask=valid_mask,
                )
            elif getattr(self, "alignment_in_image_only", False):
                # 逐图像构造 K×K 配对矩阵；跨图像 token 本就不构成有意义的负样本。
                per_image: List[torch.Tensor] = []
                st_acc: Dict[str, List[torch.Tensor]] = {}
                for b in range(xo.shape[0]):
                    l_b, st_b = self._patch_siglip_loss(
                        xo[b : b + 1],
                        ys[b : b + 1],
                        radius=r,
                        valid_mask=None if valid_mask is None else valid_mask[b : b + 1],
                        semantic_labels=None if semantic_labels is None else semantic_labels[b : b + 1],
                    )
                    per_image.append(l_b)
                    for k, v in st_b.items():
                        st_acc.setdefault(k, []).append(v)
                loss_i = torch.stack(per_image).mean()
                st = {k: torch.stack(v).mean() for k, v in st_acc.items()}
            else:
                loss_i, st = self._patch_siglip_loss(
                    xo,
                    ys,
                    radius=r,
                    valid_mask=valid_mask,
                    semantic_labels=semantic_labels,
                )
            total = total + loss_i
            # 统一命名为负索引层标记（l-1、l-2 ...）
            for k, v in st.items():
                lvl = i - num_levels
                stats[f"{k}_l{lvl}"] = v

        total = total / float(len(sel_indices))
        return total, stats

    def _distill_alignment_loss(
        self,
        opt_feats: List[torch.Tensor],
        sar_feats: List[torch.Tensor],
        target: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """方向性跨模态蒸馏：SAR（学生）token 向 detach 的光学（教师）同位置 token 靠近。

        光学流通常带预训练权重，SAR流随机初始化；只让SAR侧移动避免对称对齐把
        强表示拽向弱表示。仅使用类别内部（非边界）位置，被模态dropout置零的样本
        不参与。损失为 1 − cos，范围 [0, 2]。
        """
        stats: Dict[str, torch.Tensor] = {}
        if self.alignment_distill_proj is None:
            raise RuntimeError("alignment_mode=distill 需要 alignment_distill_proj")
        num_levels = min(len(opt_feats), len(sar_feats), len(self.alignment_distill_proj))
        sel: List[int] = []
        for idx in self.alignment_layers:
            i = idx if idx >= 0 else (num_levels + idx)
            if 0 <= i < num_levels:
                sel.append(i)
        sel = sorted(set(sel))
        if not sel:
            device = opt_feats[-1].device if len(opt_feats) else sar_feats[-1].device
            return torch.tensor(0.0, device=device), stats

        total = 0.0
        for i in sel:
            teacher = opt_feats[i].detach()
            student = sar_feats[i]
            if teacher.shape[2:] != student.shape[2:]:
                th = min(teacher.shape[2], student.shape[2])
                tw = min(teacher.shape[3], student.shape[3])
                teacher = F.interpolate(teacher, size=(th, tw), mode="bilinear", align_corners=False)
                student = F.interpolate(student, size=(th, tw), mode="bilinear", align_corners=False)
            student = self.alignment_distill_proj[i](student)
            n, c, h, w = student.shape

            valid = torch.ones((n, h, w), dtype=torch.bool, device=student.device)
            if target is not None and getattr(self, "alignment_exclude_boundaries", True):
                tm = target
                if tm.ndim == 4 and tm.shape[1] == 1:
                    tm = tm[:, 0]
                labels = F.interpolate(
                    tm.to(dtype=torch.float32).unsqueeze(1), size=(h, w), mode="nearest"
                )
                labels_c = labels.clamp_min(0)
                local_max = F.max_pool2d(labels_c, 3, stride=1, padding=1)
                local_min = -F.max_pool2d(-labels_c, 3, stride=1, padding=1)
                valid = valid & (local_max == local_min).squeeze(1) & (labels.squeeze(1) >= 0)
            keep = getattr(self, "_align_keep_mask", None)
            if keep is not None:
                valid = valid & keep.to(student.device).view(-1, 1, 1)

            cos = F.cosine_similarity(student, teacher, dim=1)  # [n, h, w]
            valid_f = valid.to(cos.dtype)
            if getattr(self, "alignment_reliability_weight", False):
                # 低可靠位置（当前相似度在有效位置中排名靠后）降权，避免把云/噪声位置的
                # SAR 特征强行拉向无意义的光学目标。
                with torch.no_grad():
                    flat = cos.detach().flatten()
                    vflat = valid.flatten()
                    rel = torch.zeros_like(flat)
                    if int(vflat.sum()) > 1:
                        vals = flat[vflat]
                        ranks = vals.argsort().argsort().to(flat.dtype)
                        rel[vflat] = ranks / float(vals.numel() - 1)
                    rel = rel.view_as(cos)
                valid_f = valid_f * rel
            denom = valid_f.sum().clamp_min(1.0)
            loss_i = ((1.0 - cos) * valid_f).sum() / denom
            total = total + loss_i
            lvl = i - num_levels
            stats[f"align_distill_cos_l{lvl}"] = (
                (cos.detach() * valid_f).sum() / denom
            )
            stats[f"align_valid_tokens_l{lvl}"] = denom.detach()
        return total / float(len(sel)), stats

    def _soft_correspondence_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        semantic_labels: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
        radius: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """在语义一致的局部窗口内软选择跨模态对应，复杂度为 O(B·K·W²)。"""
        if semantic_labels is None or valid_mask is None:
            raise ValueError("soft_correspondence 需要 semantic_labels 与 valid_mask")
        n, c, h, w = x.shape
        if semantic_labels.shape != (n, h, w) or valid_mask.shape != (n, h, w):
            raise ValueError("soft_correspondence 的标签/有效掩膜尺寸与特征不一致")
        radius = max(0, int(radius))
        kernel = 2 * radius + 1
        window_size = kernel * kernel
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        x_tokens = x_norm.flatten(2).transpose(1, 2)
        y_tokens = y_norm.flatten(2).transpose(1, 2)

        def patches(feats: torch.Tensor) -> torch.Tensor:
            unfolded = F.unfold(feats, kernel_size=kernel, padding=radius)
            return unfolded.view(n, c, window_size, h * w).permute(0, 3, 2, 1)

        x_patches = patches(x_norm)
        y_patches = patches(y_norm)
        label_patches = F.unfold(
            semantic_labels.float().unsqueeze(1),
            kernel_size=kernel,
            padding=radius,
        ).view(n, window_size, h * w).transpose(1, 2).long()
        valid_patches = F.unfold(
            valid_mask.float().unsqueeze(1),
            kernel_size=kernel,
            padding=radius,
        ).view(n, window_size, h * w).transpose(1, 2).bool()
        offsets = torch.arange(-radius, radius + 1, device=x.device)
        dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
        distance_sq = (dy.square() + dx.square()).reshape(1, window_size)

        losses: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        anchors_total = 0
        valid_flat = valid_mask.reshape(n, -1)
        labels_flat = semantic_labels.reshape(n, -1)
        max_samples = int(self.alignment_num_samples)

        for batch_idx in range(n):
            candidates = torch.nonzero(valid_flat[batch_idx], as_tuple=False).flatten()
            if candidates.numel() == 0 or max_samples <= 0:
                continue
            count = min(max_samples, int(candidates.numel()))
            idx = candidates[
                torch.randperm(candidates.numel(), device=x.device)[:count]
            ]
            anchor_labels = labels_flat[batch_idx, idx]
            candidate_labels = label_patches[batch_idx, idx]
            candidate_valid = valid_patches[batch_idx, idx]
            same_class = candidate_labels == anchor_labels.unsqueeze(1)
            pair_valid = candidate_valid & same_class
            usable = pair_valid.any(dim=1)
            if not usable.any():
                continue
            idx = idx[usable]
            pair_valid = pair_valid[usable]

            def directional(
                anchors: torch.Tensor, candidate_patches: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                anchor = anchors[batch_idx, idx]
                candidate = candidate_patches[batch_idx, idx]
                candidate_for_match = (
                    candidate.detach()
                    if self.alignment_stop_gradient
                    else candidate
                )
                scores = torch.einsum(
                    "kc,kwc->kw", anchor, candidate_for_match
                )
                scores = scores / max(1e-6, self.alignment_temperature)
                scores = scores - self.alignment_position_weight * distance_sq
                scores = scores.masked_fill(~pair_valid, -torch.inf)
                weights = torch.softmax(scores, dim=1)
                matched = torch.einsum(
                    "kw,kwc->kc", weights, candidate_for_match
                )
                cosine = F.cosine_similarity(anchor, matched, dim=1)
                entropy = -(weights.clamp_min(1e-8).log() * weights).sum(dim=1)
                return (1.0 - cosine).mean(), entropy.mean()

            forward_loss, forward_entropy = directional(x_tokens, y_patches)
            reverse_loss, reverse_entropy = directional(y_tokens, x_patches)
            losses.append(0.5 * (forward_loss + reverse_loss))
            entropies.append(0.5 * (forward_entropy + reverse_entropy))
            anchors_total += int(idx.numel())

        if not losses:
            zero = (x.sum() + y.sum()) * 0.0
            return zero, {
                "align_soft_entropy": zero.detach(),
                "align_valid_tokens": zero.detach(),
            }
        return torch.stack(losses).mean(), {
            "align_soft_entropy": torch.stack(entropies).mean().detach(),
            "align_valid_tokens": x.new_tensor(float(anchors_total)).detach(),
        }

    def _prototype_alignment_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        semantic_labels: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """用另一模态的批内类别原型监督当前模态，成本为 O(B·K·C)。"""
        if semantic_labels is None or valid_mask is None:
            raise ValueError("prototype alignment 需要 semantic_labels 与 valid_mask")
        n, c, h, w = x.shape
        x_flat = F.normalize(x, dim=1).flatten(2).transpose(1, 2)
        y_flat = F.normalize(y, dim=1).flatten(2).transpose(1, 2)
        labels_flat = semantic_labels.reshape(n, -1)
        valid_flat = valid_mask.reshape(n, -1)
        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        max_samples = int(self.alignment_num_samples)
        for batch_idx in range(n):
            candidates = torch.nonzero(valid_flat[batch_idx], as_tuple=False).flatten()
            if candidates.numel() == 0 or max_samples <= 0:
                continue
            count = min(max_samples, int(candidates.numel()))
            idx = candidates[
                torch.randperm(candidates.numel(), device=x.device)[:count]
            ]
            xs.append(x_flat[batch_idx, idx])
            ys.append(y_flat[batch_idx, idx])
            labels.append(labels_flat[batch_idx, idx])
        if not xs:
            zero = (x.sum() + y.sum()) * 0.0
            return zero, {
                "align_proto_acc": zero.detach(),
                "align_valid_tokens": zero.detach(),
            }
        X = torch.cat(xs)
        Y = torch.cat(ys)
        target = torch.cat(labels).long()
        present_classes = torch.unique(target)
        if present_classes.numel() < 2:
            zero = (x.sum() + y.sum()) * 0.0
            return zero, {
                "align_proto_acc": zero.detach(),
                "align_valid_tokens": X.new_tensor(float(X.shape[0])).detach(),
            }
        proto_x = torch.stack(
            [F.normalize(X[target == cls].mean(dim=0), dim=0) for cls in present_classes]
        )
        proto_y = torch.stack(
            [F.normalize(Y[target == cls].mean(dim=0), dim=0) for cls in present_classes]
        )
        remapped_target = torch.searchsorted(present_classes, target)
        target_x = proto_y.detach() if self.alignment_stop_gradient else proto_y
        target_y = proto_x.detach() if self.alignment_stop_gradient else proto_x
        logits_x = X @ target_x.t() / max(1e-6, self.alignment_temperature)
        logits_y = Y @ target_y.t() / max(1e-6, self.alignment_temperature)
        loss = 0.5 * (
            F.cross_entropy(logits_x, remapped_target)
            + F.cross_entropy(logits_y, remapped_target)
        )
        acc = 0.5 * (
            (logits_x.argmax(dim=1) == remapped_target).float().mean()
            + (logits_y.argmax(dim=1) == remapped_target).float().mean()
        )
        return loss, {
            "align_proto_acc": acc.detach(),
            "align_valid_tokens": X.new_tensor(float(X.shape[0])).detach(),
        }

    def _patch_siglip_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        radius: int = 1,
        valid_mask: Optional[torch.Tensor] = None,
        semantic_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """SigLIP 版 PatchNCE：对 (B·K, B·K) 相似度矩阵做逐元素 BCEWithLogits，
        正样本为同图像内 Chebyshev 距离 ≤ r 的采样对。

        ``valid_mask`` 用于变化感知对齐。提供时，每张图仅从 True 位置采样，
        因而变化位置既不会成为正样本，也不会进入有效负样本集合。
        """
        n, c, h, w = x.shape
        k_cfg = int(self.alignment_num_samples)
        x_flat = x.reshape(n, c, -1)
        y_flat = y.reshape(n, c, -1)

        if valid_mask is None:
            idx = self._sample_spatial_positions(h, w, k_cfg, x.device)
            k_eff = int(idx.numel())
            if k_eff == 0:
                zero = (x.sum() + y.sum()) * 0.0
                return zero, {
                    "align_nce_acc": zero.detach(),
                    "align_valid_tokens": zero.detach(),
                }
            x_s = x_flat[:, :, idx]
            y_s = y_flat[:, :, idx]
            X = F.normalize(x_s.permute(0, 2, 1).reshape(-1, c), dim=1)
            Y = F.normalize(y_s.permute(0, 2, 1).reshape(-1, c), dim=1)
            batch_ids = torch.arange(n, device=x.device).repeat_interleave(k_eff)
            sampled_idx = idx.repeat(n)
        else:
            if valid_mask.shape != (n, h, w):
                raise ValueError(
                    f"valid_mask 应为 {(n, h, w)}，当前为 {tuple(valid_mask.shape)}"
                )
            sampled_x: List[torch.Tensor] = []
            sampled_y: List[torch.Tensor] = []
            sampled_indices: List[torch.Tensor] = []
            sampled_batches: List[torch.Tensor] = []
            valid_flat = valid_mask.to(device=x.device, dtype=torch.bool).reshape(n, -1)
            for b in range(n):
                candidates = torch.nonzero(valid_flat[b], as_tuple=False).flatten()
                if candidates.numel() == 0 or k_cfg <= 0:
                    continue
                count = min(k_cfg, int(candidates.numel()))
                choice = candidates[
                    torch.randperm(candidates.numel(), device=x.device)[:count]
                ]
                sampled_x.append(x_flat[b, :, choice].t())
                sampled_y.append(y_flat[b, :, choice].t())
                sampled_indices.append(choice)
                sampled_batches.append(
                    torch.full((count,), b, device=x.device, dtype=torch.long)
                )
            if not sampled_x:
                zero = (x.sum() + y.sum()) * 0.0
                return zero, {
                    "align_nce_acc": zero.detach(),
                    "align_valid_tokens": zero.detach(),
                }
            X = F.normalize(torch.cat(sampled_x, dim=0), dim=1)
            Y = F.normalize(torch.cat(sampled_y, dim=0), dim=1)
            sampled_idx = torch.cat(sampled_indices)
            batch_ids = torch.cat(sampled_batches)

        logits = (X @ Y.t()) / max(1e-6, self.alignment_temperature)  # [n*k_eff, n*k_eff]

        # 正样本掩码 P：同一图像且位于局部邻域。
        yy = (sampled_idx // w).to(torch.int64)
        xx = (sampled_idx % w).to(torch.int64)
        dy = yy.unsqueeze(0) - yy.unsqueeze(1)
        dx = xx.unsqueeze(0) - xx.unsqueeze(1)
        cheb = torch.maximum(dy.abs(), dx.abs())
        same_batch = batch_ids.unsqueeze(0) == batch_ids.unsqueeze(1)
        local_pair = same_batch & (cheb <= int(radius))
        if semantic_labels is None:
            positive_mask = local_pair
            valid_pair_mask = torch.ones_like(positive_mask)
        else:
            if semantic_labels.shape != (n, h, w):
                raise ValueError(
                    f"semantic_labels 应为 {(n, h, w)}，"
                    f"当前为 {tuple(semantic_labels.shape)}"
                )
            labels_flat = semantic_labels.to(device=x.device).reshape(n, -1)
            sampled_labels = labels_flat[batch_ids, sampled_idx]
            same_class = sampled_labels.unsqueeze(0) == sampled_labels.unsqueeze(1)
            positive_mask = local_pair & same_class
            # 同类非局部 pair 不作为负样本，避免把同质水体/背景错误推远。
            valid_pair_mask = positive_mask | (~same_class)

        P = positive_mask.to(logits.dtype)
        valid_pairs = valid_pair_mask.to(logits.dtype)
        if getattr(self, "alignment_reliability_weight", False):
            # RULE式可靠性：同位置对的 detach 余弦相似度在 batch 内做秩归一化到 [0,1]，
            # 正样本目标改为软目标 min(r_i, r_j)。低可靠对应（云、speckle、局部变化）
            # 不被强行拉近，也不被当作可靠负样本。
            with torch.no_grad():
                co_sim = (X * Y).sum(dim=1)
                ranks = co_sim.argsort().argsort().to(logits.dtype)
                rel = ranks / max(1.0, float(co_sim.numel() - 1))
                rel_pair = torch.minimum(rel.unsqueeze(0), rel.unsqueeze(1))
            P = P * rel_pair
        pos = P.sum()
        neg = (valid_pairs - P).sum()
        if pos.item() == 0 or neg.item() == 0:
            zero = (x.sum() + y.sum()) * 0.0
            return zero, {
                "align_nce_acc": zero.detach(),
                "align_valid_tokens": logits.new_tensor(
                    float(logits.shape[0])
                ).detach(),
            }
        pos_weight = (neg / (pos + 1e-6)).to(logits.dtype)
        def masked_bce(
            directional_logits: torch.Tensor,
            targets: torch.Tensor,
            pair_mask: torch.Tensor,
        ) -> torch.Tensor:
            per_pair = F.binary_cross_entropy_with_logits(
                directional_logits,
                targets,
                pos_weight=pos_weight,
                reduction="none",
            )
            return (per_pair * pair_mask).sum() / pair_mask.sum().clamp_min(
                1.0
            )

        if (
            semantic_labels is not None
            and self.alignment_semantic_stop_gradient
        ):
            temperature = max(1e-6, self.alignment_temperature)
            logits_xy = (X @ Y.detach().t()) / temperature
            logits_yx = (Y @ X.detach().t()) / temperature
            loss = 0.5 * (
                masked_bce(logits_xy, P, valid_pairs)
                + masked_bce(logits_yx, P.t(), valid_pairs.t())
            )
            top1_xy = logits_xy.argmax(dim=1)
            top1_yx = logits_yx.argmax(dim=1)
            rows = torch.arange(logits_xy.size(0), device=logits_xy.device)
            acc = 0.5 * (
                P[rows, top1_xy].float().mean()
                + P.t()[rows, top1_yx].float().mean()
            )
            acc = acc.detach()
        else:
            loss = masked_bce(logits, P, valid_pairs)
            # Top-1 是否命中正样本集合
            top1 = logits.argmax(dim=1)
            rows = torch.arange(logits.size(0), device=logits.device)
            acc = P[rows, top1].float().mean().detach()
        stats = {
            "align_nce_acc": acc,
            "align_valid_tokens": logits.new_tensor(float(logits.shape[0])).detach(),
        }
        return loss, stats

    # KL 策略已移除
    
    def _update_metrics(self, preds, masks, stage):
        """更新评估指标"""
        # 过滤无效标签
        valid_mask, valid_preds, valid_masks = self._compute_valid_preds_targets(preds, masks)
        if valid_mask.sum() == 0:
            logger.warning(f"当前batch没有有效标签，跳过指标计算")
            return
        
        # 更新基础指标
        if stage == "train":
            self.train_iou(valid_preds, valid_masks)
            self.train_iou_pc(valid_preds, valid_masks)
            self.train_f1(valid_preds, valid_masks)
            self.train_acc(valid_preds, valid_masks)
            self.train_precision(valid_preds, valid_masks)
            self.train_recall(valid_preds, valid_masks)
            self.train_specificity(valid_preds, valid_masks)
        elif stage == "val":
            self.val_iou(valid_preds, valid_masks)
            self.val_iou_pc(valid_preds, valid_masks)
            self.val_f1(valid_preds, valid_masks)
            self.val_acc(valid_preds, valid_masks)
            self.val_precision(valid_preds, valid_masks)
            self.val_recall(valid_preds, valid_masks)
            self.val_specificity(valid_preds, valid_masks)
        elif stage == "test":
            self.test_iou(valid_preds, valid_masks)
            self.test_iou_pc(valid_preds, valid_masks)
            self.test_f1(valid_preds, valid_masks)
            self.test_acc(valid_preds, valid_masks)
            self.test_precision(valid_preds, valid_masks)
            self.test_recall(valid_preds, valid_masks)
            self.test_specificity(valid_preds, valid_masks)
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self._shared_step(batch, "test")
    
    def on_train_epoch_start(self):
        """训练轮次开始"""
        # 将当前 epoch 传给融合层（用于 start_epoch 延迟启用）
        if hasattr(self.fusion, "set_epoch"):
            self.fusion.set_epoch(self.current_epoch)

    def on_train_epoch_end(self):
        """训练轮次结束"""
        self._log_epoch_metrics("train")
        self._reset_metrics("train")
    
    def on_validation_epoch_end(self):
        """验证轮次结束"""
        self._log_epoch_metrics("val")
        self._reset_metrics("val")
    
    def on_test_epoch_end(self):
        """测试轮次结束"""
        self._log_epoch_metrics("test")
        self._reset_metrics("test")
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        统一的预测步骤：返回与推理Writer兼容的输出字典。
        - 始终返回 'main_logits'（已根据mask尺寸对齐，如可用）
        - 同时返回 'preds' 便于直接落盘与统计
        """
        outputs = self(batch)
        logits = outputs["main_logits"]
        masks = batch.get("mask")
        if isinstance(masks, torch.Tensor):
            logits = self._resize_to_target(logits, masks)
        preds = torch.argmax(logits, dim=1)
        return {"main_logits": logits, "preds": preds}
    
    def _log_epoch_metrics(self, stage: str):
        """记录epoch级别的指标"""
        # 基础指标
        iou_metric = getattr(self, f'{stage}_iou')
        f1_metric = getattr(self, f'{stage}_f1')
        acc_metric = getattr(self, f'{stage}_acc')
        precision_metric = getattr(self, f'{stage}_precision')
        recall_metric = getattr(self, f'{stage}_recall')
        specificity_metric = getattr(self, f'{stage}_specificity')
        
        # 记录基础指标
        self.log(f"{stage}/iou", iou_metric.compute(), prog_bar=True)
        self.log(f"{stage}/f1", f1_metric.compute(), prog_bar=True)
        self.log(f"{stage}/acc", acc_metric.compute(), prog_bar=True)
        self.log(f"{stage}/precision", precision_metric.compute(), prog_bar=False)
        
        # 计算按类别的recall和specificity
        current_recall = recall_metric.compute()      # [C]
        current_specificity = specificity_metric.compute()  # [C]
        
        # 记录宏平均的recall和specificity（用于整体评估）
        macro_recall = current_recall.mean()
        macro_specificity = current_specificity.mean()
        self.log(f"{stage}/recall", macro_recall, prog_bar=False)
        self.log(f"{stage}/specificity", macro_specificity, prog_bar=False)
        
        # 仅在二分类时记录水文特有指标
        self._log_hydrology_metrics(stage, current_recall, current_specificity)

        # 最优权重策略：
        # - train/val：仅当本轮 val 指标取得“历史最优”时，导出 train 与 val 的事件级CSV；否则仅清空缓存
        # - test：始终导出一次（请使用 best ckpt 运行 test）
        try:
            if stage == "val":
                current_val_iou = float(iou_metric.compute().detach().cpu().item())  # type: ignore[name-defined]
                best = getattr(self, "_best_val_iou", None)
                if (best is None) or (current_val_iou > float(best)):
                    setattr(self, "_best_val_iou", current_val_iou)
                    # 导出本轮的 train/val 事件指标
                    self._dump_event_iou_csv("train")
                    self._dump_event_iou_csv("val")
                else:
                    # 清空缓存，避免跨epoch累积
                    self._reset_event_iou_cache("train")
                    self._reset_event_iou_cache("val")
            elif stage == "train":
                # 训练阶段仅清空缓存，真实导出由 val 最优时触发
                self._reset_event_iou_cache("train")
            elif stage == "test":
                self._dump_event_iou_csv("test")
                self._dump_test_per_image_csv()
        except Exception:
            # 安全回退：如上逻辑失败，不影响训练流程
            pass

    # ----------------- 测试期逐图像混淆计数（用于 paired bootstrap） -----------------
    def _accumulate_test_per_image(self, batch: Dict[str, torch.Tensor], preds: torch.Tensor, masks: torch.Tensor) -> None:
        """记录每张测试图像的水体类 TP/FP/FN/TN。测试 loader 不打乱，
        因此同一数据集上不同模型的行序一致，可直接做逐图像配对 bootstrap。"""
        valid = masks != -1
        pred_fg = (preds == 1) & valid
        tgt_fg = (masks == 1) & valid
        tp = (pred_fg & tgt_fg).sum(dim=(1, 2))
        fp = (pred_fg & ~tgt_fg).sum(dim=(1, 2))
        fn = (~pred_fg & tgt_fg & valid).sum(dim=(1, 2))
        tn = (~pred_fg & ~tgt_fg & valid).sum(dim=(1, 2))
        act = batch.get("activation")
        if act is None:
            act = torch.full_like(tp, -1)
        rows = torch.stack([act.to(tp.device).long(), tp.long(), fp.long(), fn.long(), tn.long()], dim=1).cpu()
        if not hasattr(self, "_test_per_image_rows"):
            self._test_per_image_rows: List[torch.Tensor] = []
        self._test_per_image_rows.append(rows)

    def _dump_test_per_image_csv(self) -> None:
        rows = getattr(self, "_test_per_image_rows", None)
        if not rows:
            return
        import os
        from lightning_utilities.core.rank_zero import rank_zero_only

        @rank_zero_only
        def _write(all_rows: torch.Tensor):
            log_dir = getattr(self.logger, 'log_dir', None) or getattr(self.logger, 'save_dir', '.')
            out_dir = os.path.join(log_dir, 'event_metrics')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, 'test_per_image_confusion.csv')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('idx,activation,tp,fp,fn,tn\n')
                for i, r in enumerate(all_rows.tolist()):
                    f.write(f"{i},{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")
        _write(torch.cat(rows, dim=0))
        self._test_per_image_rows = []

    def _log_hydrology_metrics(self, stage: str, current_recall: torch.Tensor, current_specificity: torch.Tensor) -> None:
        """记录与水文相关的指标（仅二分类时）。"""
        num_classes = self.hparams.num_classes if hasattr(self.hparams, "num_classes") else 2
        if num_classes != 2 or current_recall.numel() != 2 or current_specificity.numel() != 2:
            logger.debug(
                f"跳过水文特有指标记录：num_classes={num_classes},"
                f" recall.shape={tuple(current_recall.shape)}"
            )
            return
        # water_iou: 类别1的IoU
        per_class_iou = getattr(self, f"{stage}_iou_pc").compute()
        if per_class_iou.numel() == 2:
            water_iou = per_class_iou[1]
            self.log(f"{stage}/water_iou", water_iou, prog_bar=True)
        # 漏检率/虚警率
        flood_recall = current_recall[1]
        flood_miss_rate = 1.0 - flood_recall
        bg_specificity_true = current_specificity[1]
        bg_false_alarm_rate = 1.0 - bg_specificity_true
        self.log(f"{stage}/flood_miss_rate", flood_miss_rate, prog_bar=False)
        self.log(f"{stage}/bg_false_alarm_rate", bg_false_alarm_rate, prog_bar=False)
        # 细粒度记录
        self.log(f"{stage}/flood_recall", flood_recall, prog_bar=False)
        self.log(f"{stage}/bg_recall", current_recall[0], prog_bar=False)
        self.log(f"{stage}/flood_specificity", current_specificity[1], prog_bar=False)
        self.log(f"{stage}/bg_specificity", bg_specificity_true, prog_bar=False)

    def _reset_metrics(self, stage: str):
        """在一个epoch结束后显式reset，避免跨epoch的累积"""
        for name in ("iou", "iou_pc", "f1", "acc", "precision", "recall", "specificity"):
            metric = getattr(self, f"{stage}_{name}")
            metric.reset()

    # ----------------- 事件级 IoU 累积与导出 -----------------
    def _accumulate_event_iou(self, batch: Dict[str, torch.Tensor], preds: torch.Tensor, masks: torch.Tensor, stage: str) -> None:
        """按激活ID累计 IoU 与 water_iou 分子/分母，epoch 结束时导出。
        需要 batch 中包含键 'activation'（来自 Dataset），shape=[B]。
        """
        # 支持 train/val/test
        act_ids = batch.get("activation")
        if act_ids is None:
            return
        # 过滤无效像素
        valid_mask = masks != -1
        if valid_mask.sum() == 0:
            return
        # 仅统计洪水类(1)的 IoU（即 water_iou）
        pred_fg = (preds == 1) & valid_mask
        tgt_fg = (masks == 1) & valid_mask
        inter = (pred_fg & tgt_fg).sum(dim=(1, 2)).to(torch.float32)
        union = (pred_fg | tgt_fg).sum(dim=(1, 2)).to(torch.float32)

        # 缓存容器
        cache_name = f"_{stage}_event_iou_cache"
        if not hasattr(self, cache_name):
            setattr(self, cache_name, {})  # type: ignore[attr-defined]
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = getattr(self, cache_name)  # type: ignore[assignment]

        for i in range(act_ids.shape[0]):
            act = int(act_ids[i])
            num = inter[i].detach()
            den = union[i].detach()
            if act in cache:
                prev_num, prev_den = cache[act]
                cache[act] = (prev_num + num, prev_den + den)
            else:
                cache[act] = (num, den)

    def _dump_event_iou_csv(self, stage: str) -> None:
        cache_name = f"_{stage}_event_iou_cache"
        if not hasattr(self, cache_name):
            return
        cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = getattr(self, cache_name)
        if not cache:
            return
        # 计算 IoU
        import os
        from lightning_utilities.core.rank_zero import rank_zero_only

        @rank_zero_only
        def _write_csv(d: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
            log_dir = getattr(self.logger, 'log_dir', None) or getattr(self.logger, 'save_dir', '.')
            out_dir = os.path.join(log_dir, 'event_metrics')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'{stage}_event_water_iou.csv')
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write('actid,water_iou,inter,union\n')
                for act in sorted(d.keys()):
                    num, den = d[act]
                    iou = float((num / den).cpu().item()) if float(den.cpu().item()) > 0 else 0.0
                    f.write(f"{act},{iou:.6f},{float(num.cpu().item()):.0f},{float(den.cpu().item()):.0f}\n")
        _write_csv(cache)
        # 清空缓存
        setattr(self, cache_name, {})

    def _reset_event_iou_cache(self, stage: str) -> None:
        cache_name = f"_{stage}_event_iou_cache"
        if hasattr(self, cache_name):
            setattr(self, cache_name, {})
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if self.hparams.scheduler_config:
            scheduler = hydra.utils.instantiate(
                self.hparams.scheduler_config,
                optimizer=optimizer
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/iou",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        
        return optimizer
