"""LightningModule that orchestrates multi-modal segmentation training/evaluation."""

from typing import Any, Dict, Optional, Union, Tuple, List, cast
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule  # type: ignore[import]
import hydra  # type: ignore[import]
from omegaconf import DictConfig  # type: ignore[import]
from torchmetrics import Accuracy, JaccardIndex, F1Score, Precision, Recall, Specificity  # type: ignore[import]
import logging

logger = logging.getLogger(__name__)


class MultiModalSegmentationModule(LightningModule):
    """
    Config-driven segmentation module.

    All components are instantiated from config dicts (with `_target_`) so checkpoints remain self-contained.
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
        # Alignment regularization (shared projection space + contrastive-style alignment)
        alignment_enabled: bool = False,
        alignment_type: str = "nce",  # only nce retained
        alignment_target_weight: float = 0.02,
        alignment_layers: Optional[List[int]] = None,
        alignment_temperature: float = 0.1,
        alignment_num_samples: int = 256,
        # PatchNCE: positives are local neighbors (radius r; r=0 matches standard InfoNCE)
        alignment_patch_radius: int = 1,
        # SigLIP-style: PatchNCE uses BCEWithLogits (multi-positive modeling)
        alignment_sigmoid: bool = True,
        align_cap_ratio: float = 0.3,
        **kwargs
    ):
        super().__init__()
        
        # Save hparams to keep the checkpoint self-contained.
        self.save_hyperparameters()
        
        # Instantiate components
        self.encoder = cast(Any, self._instantiate_component(encoder, "encoder"))
        
        # Read encoder meta needed by fusion/decoder.
        feature_channels = getattr(self.encoder, 'feature_channels', [64, 256, 512, 1024, 2048])
        feature_reductions = getattr(self.encoder, 'feature_reductions', [4, 8, 16, 32])
        optical_channels = getattr(self.encoder, 'optical_feature_channels', None)
        sar_channels = getattr(self.encoder, 'sar_feature_channels', None)
        
        fusion_config = self._prepare_fusion_config(fusion, feature_channels, optical_channels, sar_channels)

        # Instantiate fusion first to obtain output_channels for the decoder.
        self.fusion = cast(Any, self._instantiate_component(fusion_config, "fusion"))

        fused_feature_channels = getattr(self.fusion, 'output_channels', feature_channels)
        decoder_config = self._prepare_decoder_config(decoder, fused_feature_channels, feature_reductions)
        self.decoder = cast(Any, self._instantiate_component(decoder_config, "decoder"))

        # Record fused channels for reproducibility.
        self.hparams.fused_feature_channels = list(fused_feature_channels)
        self.loss_fn = cast(Any, self._instantiate_component(loss_fn, "loss_fn"))
        
        # Auxiliary loss (optional)
        self.aux_loss_fn = self._instantiate_component(aux_loss_fn, "aux_loss_fn") if aux_loss_fn else None
        self.aux_loss_weight = float(aux_loss_weight)
        if self.aux_loss_weight > 0 and self.aux_loss_fn:
            fused_last_channels = int(fused_feature_channels[-1])
            # Two auxiliary heads (both take fused features).
            self.aux_head_1 = nn.Conv2d(fused_last_channels, num_classes, 1)
            self.aux_head_2 = nn.Conv2d(fused_last_channels, num_classes, 1)

        # Reconstruction path removed.

        # Alignment config (reuse fusion projection heads when available).
        self.alignment_enabled = bool(alignment_enabled)
        self.alignment_type = str(alignment_type)
        self.alignment_target_weight = float(alignment_target_weight)
        self.alignment_layers = list(alignment_layers) if alignment_layers is not None else [-1]
        self.alignment_temperature = float(alignment_temperature)
        self.alignment_num_samples = int(alignment_num_samples)
        self.alignment_patch_radius = int(alignment_patch_radius)
        self.alignment_sigmoid = bool(alignment_sigmoid)

        # For alignment loss capping: EMA of main loss and a fixed ratio.
        self.register_buffer("align_ema_main", torch.tensor(0.0), persistent=False)
        self.mra_ema_decay: float = 0.9
        self.align_cap_ratio: float = float(align_cap_ratio)

        # Reconstruction head removed.
        
        # Initialize metrics
        self._init_metrics(num_classes)
    
    def _instantiate_component(self, component, name: str):
        """Instantiate a component from a config dict (expects `_target_`)."""
        if component is None:
            return None
            
        if not isinstance(component, (dict, DictConfig)):
            raise ValueError(
                f"Component '{name}' must be a config dict (dict/DictConfig), got: {type(component)}. "
                f"Please use `_target_` in the config to specify the class path."
            )
        
        # Special-case: FocalLoss alpha can be provided as a list; convert to a float.
        if ('FocalLoss' in str(component.get('_target_', '')) and 'alpha' in component):
            # Convert to native container
            from omegaconf import OmegaConf  # type: ignore[import]
            component_native = OmegaConf.to_container(component, resolve=True)
            
            # Handle alpha
            alpha_value = component_native['alpha']
            if isinstance(alpha_value, list):
                # In multiclass mode, alpha is expected to be a single float.
                focal_alpha = alpha_value[1] if len(alpha_value) > 1 else alpha_value[0]
                component_native['alpha'] = focal_alpha
                logger.info(f"FocalLoss alpha converted: {alpha_value} -> {focal_alpha}")
            
            return hydra.utils.instantiate(component_native)
        
        # Standard Hydra instantiation
        return hydra.utils.instantiate(component)
    
    def _prepare_fusion_config(self, config, feature_channels, optical_channels, sar_channels):
        """Attach feature_channels (and per-stream channels if available) to fusion config."""
        if not isinstance(config, (dict, DictConfig)):
            return config
            
        cfg = dict(config)
        cfg['feature_channels'] = feature_channels
        if optical_channels and sar_channels:
            cfg['optical_channels'] = optical_channels
            cfg['sar_channels'] = sar_channels
        return cfg
    
    def _prepare_decoder_config(self, config: Dict[str, Any], feature_channels: List[int], feature_reductions: List[int]) -> Dict[str, Any]:
        """Attach feature_channels/reductions to decoder config."""
        cfg = dict(config)
        cfg['feature_channels'] = feature_channels
        cfg['feature_reductions'] = feature_reductions
        return cfg
    
    def _init_metrics(self, num_classes: int):
        """Initialize metrics."""
        # Core metrics
        self.train_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        self.test_iou = JaccardIndex(task="multiclass", num_classes=num_classes)
        # Per-class IoU (for logging water_iou etc.)
        self.train_iou_pc = JaccardIndex(task="multiclass", num_classes=num_classes, average=None)
        self.val_iou_pc = JaccardIndex(task="multiclass", num_classes=num_classes, average=None)
        self.test_iou_pc = JaccardIndex(task="multiclass", num_classes=num_classes, average=None)
        
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)
        
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        
        # Precision (macro)
        self.train_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        
        # Recall (per-class)
        self.train_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average=None)
        
        # Specificity (per-class)
        self.train_specificity = Specificity(task="multiclass", num_classes=num_classes, average=None)
        self.val_specificity = Specificity(task="multiclass", num_classes=num_classes, average=None)
        self.test_specificity = Specificity(task="multiclass", num_classes=num_classes, average=None)
        
        logger.info("Metrics initialized: IoU, F1, Accuracy, Precision, Recall, Specificity")
    
    def _get_input_size(self, batch: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        """Get input spatial size (H, W)."""
        if 'image' in batch:
            h, w = batch['image'].shape[-2:]
            return int(h), int(w)
        elif 'image_optical' in batch:
            h, w = batch['image_optical'].shape[-2:]
            return int(h), int(w)
        elif 'image_sar' in batch:
            h, w = batch['image_sar'].shape[-2:]
            return int(h), int(w)
        else:
            raise ValueError("No valid image tensor found in batch.")

    def _resize_to_target(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Resize logits to match target spatial size (bilinear)."""
        if logits.shape[2:] == target.shape[1:]:
            return logits
        return F.interpolate(logits, size=target.shape[1:], mode='bilinear', align_corners=False)

    def _compute_valid_preds_targets(self, preds: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter ignore labels (-1). Returns (valid_mask, valid_preds, valid_targets)."""
        valid_mask = masks != -1
        if valid_mask.sum() == 0:
            return valid_mask, preds.new_empty(0), masks.new_empty(0)
        return valid_mask, preds[valid_mask], masks[valid_mask]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward.

        Expected batch:
        - single-modal: {"image": tensor, "mask": tensor}
        - dual-modal: {"image_optical": tensor, "image_sar": tensor, "mask": tensor}
        """
        # 1) Feature extraction
        raw_features = self.encoder(batch)
        
        # 2) Feature fusion (normalize empty lists to None)
        optical_feats = raw_features.get("optical")
        sar_feats = raw_features.get("sar")
        if isinstance(optical_feats, (list, tuple)) and len(optical_feats) == 0:
            optical_feats = None
        if isinstance(sar_feats, (list, tuple)) and len(sar_feats) == 0:
            sar_feats = None

        # Unified entry: standard fusion (DropBranch removed).
        fused_features = self.fusion(optical_features=optical_feats, sar_features=sar_feats)
        
        # 3) Decode (pass input size for exact upsampling)
        input_size = self._get_input_size(batch)
        main_logits = self.decoder(fused_features, input_size=input_size)
        
        # 4) Outputs
        num_levels = len(getattr(self.fusion, 'output_channels', getattr(self.encoder, 'feature_channels', [])))
        return {
            "main_logits": main_logits,
            "optical_features": raw_features.get("optical", [None] * num_levels),
            "sar_features": raw_features.get("sar", [None] * num_levels),
            "fused_features": fused_features
        }
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared train/val/test step."""
        outputs = self(batch)
        masks = batch["mask"]
        
        # Resize logits to match masks
        main_logits = self._resize_to_target(outputs["main_logits"], masks)
        
        # Main loss (multi-class)
        main_loss = self.loss_fn(main_logits, masks)
        total_loss = main_loss
        # Track scaled components for logging
        device = masks.device
        aux_raw = torch.tensor(0.0, device=device)
        aux_scaled = torch.tensor(0.0, device=device)
        align_raw = torch.tensor(0.0, device=device)
        align_scaled = torch.tensor(0.0, device=device)
        # Track whether alignment is computed to avoid logging uninitialized tensors.
        align_computed = False
        
        # Only log main loss for train/val (skip test to keep tables concise).
        if stage != "test":
            self.log(f"{stage}/loss_main", main_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Update EMA of main loss (used for alignment capping).
        with torch.no_grad():
            if self.align_ema_main.device != main_loss.device:
                self.align_ema_main = self.align_ema_main.to(main_loss.device)
            if float(self.align_ema_main.item()) == 0.0:
                self.align_ema_main.copy_(main_loss.detach())
            else:
                self.align_ema_main.copy_(
                    self.mra_ema_decay * self.align_ema_main + (1.0 - self.mra_ema_decay) * main_loss.detach()
                )
        
        # Auxiliary loss (optional)
        if self.training and self.aux_loss_weight > 0 and self.aux_loss_fn:
            aux_loss = cast(torch.Tensor, self._compute_aux_loss(outputs, masks, stage))
            aux_raw = aux_loss
            aux_scaled = self.aux_loss_weight * aux_loss
            total_loss = total_loss + aux_scaled
        
        # Reconstruction loss removed.
        
        # Alignment regularization (train-only; dual-modal only).
        if self.training and self.alignment_enabled:
            opt_feats_any = outputs.get("optical_features", [])
            sar_feats_any = outputs.get("sar_features", [])
            has_dual_modal = (
                ("image_optical" in batch and "image_sar" in batch)
                or (
                    isinstance(opt_feats_any, (list, tuple))
                    and isinstance(sar_feats_any, (list, tuple))
                    and len(opt_feats_any) > 0
                    and len(sar_feats_any) > 0
                )
            )
            if has_dual_modal:
                opt_feats = cast(List[torch.Tensor], opt_feats_any)
                sar_feats = cast(List[torch.Tensor], sar_feats_any)
                align_loss, align_stats = self._compute_alignment_loss(opt_feats, sar_feats)
                lambda_align = self._get_alignment_schedule()
                align_raw = align_loss
                align_scaled_raw = torch.tensor(lambda_align, device=device) * align_loss
                # Hard cap (optional): if align_cap_ratio <= 0, do not cap.
                if self.align_cap_ratio <= 0:
                    cap = torch.tensor(float('inf'), device=device)
                    align_scaled = align_scaled_raw
                else:
                    cap = self.align_cap_ratio * self.align_ema_main
                    align_scaled = torch.minimum(align_scaled_raw, cap)
                total_loss = total_loss + align_scaled
                align_computed = True
                self.log(f"{stage}/loss_alignment", align_loss, on_step=True, on_epoch=True)
                self.log(f"{stage}/lambda_align", torch.tensor(lambda_align, device=masks.device), on_step=False, on_epoch=True)
                if isinstance(align_stats, dict):
                    for k, v in align_stats.items():
                        if isinstance(v, torch.Tensor):
                            self.log(f"{stage}/{k}", v, on_step=False, on_epoch=True)
            else:
                # Single-modal or missing modality: do not compute/log alignment.
                align_computed = False

        # Log scaled components and ratios (train-only).
        if stage == "train":
            self.log(f"{stage}/loss_main_scaled", main_loss, on_step=False, on_epoch=True)
            if self.aux_loss_weight > 0 and self.aux_loss_fn:
                self.log(f"{stage}/loss_aux_scaled", aux_scaled, on_step=False, on_epoch=True)
            if self.alignment_enabled and ('align_scaled' in locals()) and align_computed:
                self.log(f"{stage}/loss_align_scaled", align_scaled, on_step=False, on_epoch=True)
                if 'align_scaled_raw' in locals():
                    self.log(f"{stage}/loss_align_scaled_raw", align_scaled_raw, on_step=False, on_epoch=True)
                if 'cap' in locals():
                    # When capping is disabled, cap is +inf.
                    self.log(f"{stage}/loss_align_cap", cap, on_step=False, on_epoch=True)

            total_scaled = main_loss + aux_scaled + (align_scaled if ('align_scaled' in locals()) and align_computed else torch.tensor(0.0, device=device))
            eps = torch.tensor(1e-12, device=device)
            denom = torch.maximum(total_scaled.detach(), eps)
            self.log(f"{stage}/loss_main_pct", (main_loss.detach() / denom), on_step=False, on_epoch=True)
            self.log(f"{stage}/loss_aux_pct", (aux_scaled.detach() / denom), on_step=False, on_epoch=True)
            self.log(f"{stage}/loss_align_pct", (align_scaled.detach() / denom), on_step=False, on_epoch=True)

        # Keep loss-only logging (no entropy logging).
        
        # Metrics
        preds = torch.argmax(main_logits.detach(), dim=1)
        self._update_metrics(preds, masks, stage)
        # Event-level IoU accumulation (val/test only).
        self._accumulate_event_iou(batch, preds, masks, stage)
        
        return total_loss
    
    def _compute_aux_loss(self, outputs, masks, stage) -> torch.Tensor:
        """Compute auxiliary loss (optional)."""
        fused_features = outputs.get("fused_features")
        if not fused_features or not self.aux_loss_fn:
            return torch.tensor(0.0, device=masks.device)
        
        fused_last = fused_features[-1]
        aux_losses = []
        
        # Two auxiliary heads
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
            avg_aux_loss = torch.stack(aux_losses, dim=0).mean()
            self.log(f"{stage}/loss_aux", avg_aux_loss, on_step=True, on_epoch=True)
            return avg_aux_loss
        
        return torch.tensor(0.0, device=masks.device)

    # Reconstruction path removed.

    # ----------------- MRA: managed by fusion strategy (no implementation here) -----------------

    # Reconstruction schedules/metrics removed.

    # ----------------- Alignment -----------------
    def _get_alignment_schedule(self) -> float:
        """Return a constant alignment weight (no ramp schedule)."""
        return float(self.alignment_target_weight)

    def _sample_spatial_positions(self, h: int, w: int, k: int, device: torch.device) -> torch.Tensor:
        k = int(min(k, h * w))
        idx = torch.randperm(h * w, device=device)[:k]
        return idx

    def _compute_alignment_loss(
        self,
        opt_feats: List[torch.Tensor],
        sar_feats: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        stats: Dict[str, torch.Tensor] = {}
        # Prefer reusing fusion strategy projection layers (same space as xattn).
        strategy = getattr(self.fusion, "strategy", None)
        proj_opt = getattr(strategy, "project_opt", None)
        proj_sar = getattr(strategy, "project_sar", None)
        if proj_opt is None or proj_sar is None:
            raise RuntimeError(
                "alignment_enabled requires a fusion strategy with shared projection heads (e.g., xattn). "
                "Please switch to xattn or disable alignment_enabled."
            )

        num_levels = min(len(opt_feats), len(sar_feats), len(proj_opt), len(proj_sar))
        # Select levels via alignment_layers
        sel_indices: List[int] = []
        for idx in self.alignment_layers:
            i = idx if idx >= 0 else (num_levels + idx)
            if 0 <= i < num_levels:
                sel_indices.append(i)
        sel_indices = sorted(set(sel_indices))
        if len(sel_indices) == 0:
            device = (opt_feats[-1].device if len(opt_feats) else sar_feats[-1].device)
            return torch.tensor(0.0, device=device), stats

        total = torch.tensor(0.0, device=opt_feats[sel_indices[0]].device)
        for i in sel_indices:
            fo = opt_feats[i]
            fs = sar_feats[i]
            # Spatial alignment: downscale to min resolution (avoid upscaling to max).
            if fo.shape[2:] != fs.shape[2:]:
                target_h = min(fo.shape[2], fs.shape[2])
                target_w = min(fo.shape[3], fs.shape[3])
                if fo.shape[2:] != (target_h, target_w):
                    fo = F.interpolate(fo, size=(target_h, target_w), mode='bilinear', align_corners=False)
                if fs.shape[2:] != (target_h, target_w):
                    fs = F.interpolate(fs, size=(target_h, target_w), mode='bilinear', align_corners=False)
            # Project to the shared fusion width.
            xo = proj_opt[i](fo)
            ys = proj_sar[i](fs)
            # SigLIP-style PatchNCE; r<=0 matches diagonal-only positives.
            r = int(self.alignment_patch_radius) if hasattr(self, 'alignment_patch_radius') else 0
            r = max(0, r)
            loss_i, st = self._patch_siglip_loss(xo, ys, radius=r)
            total = total + loss_i
            # Use negative indices for level naming (l-1, l-2, ...).
            for k, v in st.items():
                lvl = i - num_levels
                stats[f"{k}_l{lvl}"] = v

        total = total / float(len(sel_indices))
        return total, stats

    def _patch_siglip_loss(self, x: torch.Tensor, y: torch.Tensor, radius: int = 1) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """SigLIP-style PatchNCE with BCEWithLogits over the similarity matrix."""
        n, c, h, w = x.shape
        k_cfg = int(self.alignment_num_samples)
        idx = self._sample_spatial_positions(h, w, k_cfg, x.device)
        # Effective sample count (can be < K when HW < K).
        k_eff = int(idx.numel())

        x_flat = x.reshape(n, c, -1)
        y_flat = y.reshape(n, c, -1)
        x_s = x_flat[:, :, idx]
        y_s = y_flat[:, :, idx]

        X = F.normalize(x_s.permute(0, 2, 1).reshape(-1, c), dim=1)
        Y = F.normalize(y_s.permute(0, 2, 1).reshape(-1, c), dim=1)
        logits = (X @ Y.t()) / max(1e-6, self.alignment_temperature)  # [n*k_eff, n*k_eff]

        # Positive mask P (block-diagonal + local neighborhood)
        yy = (idx // w).to(torch.int64)
        xx = (idx % w).to(torch.int64)
        dy = yy.unsqueeze(0) - yy.unsqueeze(1)
        dx = xx.unsqueeze(0) - xx.unsqueeze(1)
        cheb = torch.maximum(dy.abs(), dx.abs())
        local_mask = (cheb <= int(radius)).to(logits.dtype)
        local_mask = torch.maximum(local_mask, torch.eye(k_eff, device=logits.device, dtype=logits.dtype))
        P = torch.zeros((n * k_eff, n * k_eff), device=logits.device, dtype=logits.dtype)
        for b in range(n):
            P[b * k_eff:(b + 1) * k_eff, b * k_eff:(b + 1) * k_eff] = local_mask

        pos = P.sum()
        total = P.numel()
        neg = total - pos
        pos_weight = (neg / (pos + 1e-6)).to(logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, P, pos_weight=pos_weight)

        # Whether top-1 matches the positive mask
        top1 = logits.argmax(dim=1)
        rows = torch.arange(logits.size(0), device=logits.device)
        acc = P[rows, top1].float().mean().detach()
        stats = {"align_nce_acc": acc}
        return loss, stats

    # KL variant removed
    
    def _update_metrics(self, preds, masks, stage):
        """Update metrics."""
        # Filter ignore labels
        valid_mask, valid_preds, valid_masks = self._compute_valid_preds_targets(preds, masks)
        if valid_mask.sum() == 0:
            logger.warning("No valid labels in this batch; skip metric update.")
            return
        
        # Update metrics
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
        """Training step."""
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._shared_step(batch, "test")
    
    def on_train_epoch_start(self):
        """Train epoch start."""
        setter = getattr(self.fusion, "set_epoch", None)
        if callable(setter):
            setter(self.current_epoch)

    def on_train_epoch_end(self):
        """Train epoch end."""
        self._log_epoch_metrics("train")
        self._reset_metrics("train")
    
    def on_validation_epoch_end(self):
        """Validation epoch end."""
        self._log_epoch_metrics("val")
        self._reset_metrics("val")
    
    def on_test_epoch_end(self):
        """Test epoch end."""
        self._log_epoch_metrics("test")
        self._reset_metrics("test")
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Prediction step (compatible with inference writers).

        Returns:
        - 'main_logits' (optionally resized to match mask)
        - 'preds' (argmax)
        """
        outputs = self(batch)
        logits = outputs["main_logits"]
        masks = batch.get("mask")
        if isinstance(masks, torch.Tensor):
            logits = self._resize_to_target(logits, masks)
        preds = torch.argmax(logits, dim=1)
        return {"main_logits": logits, "preds": preds}
    
    def _log_epoch_metrics(self, stage: str):
        """Log epoch-level metrics."""
        # Core metrics
        iou_metric = getattr(self, f'{stage}_iou')
        f1_metric = getattr(self, f'{stage}_f1')
        acc_metric = getattr(self, f'{stage}_acc')
        precision_metric = getattr(self, f'{stage}_precision')
        recall_metric = getattr(self, f'{stage}_recall')
        specificity_metric = getattr(self, f'{stage}_specificity')
        
        # Log core metrics
        self.log(f"{stage}/iou", iou_metric.compute(), prog_bar=True)
        self.log(f"{stage}/f1", f1_metric.compute(), prog_bar=True)
        self.log(f"{stage}/acc", acc_metric.compute(), prog_bar=True)
        self.log(f"{stage}/precision", precision_metric.compute(), prog_bar=False)
        
        # Per-class recall/specificity
        current_recall = recall_metric.compute()      # [C]
        current_specificity = specificity_metric.compute()  # [C]
        
        # Log macro recall/specificity
        macro_recall = current_recall.mean()
        macro_specificity = current_specificity.mean()
        self.log(f"{stage}/recall", macro_recall, prog_bar=False)
        self.log(f"{stage}/specificity", macro_specificity, prog_bar=False)
        
        # Binary-only hydrology metrics
        self._log_hydrology_metrics(stage, current_recall, current_specificity)

        # Export policy:
        # - train/val: export event CSV only when val improves the best-so-far; otherwise reset caches
        # - test: always export once (recommended to run with best ckpt)
        try:
            if stage == "val":
                current_val_iou = float(iou_metric.compute().detach().cpu().item())  # type: ignore[name-defined]
                best = getattr(self, "_best_val_iou", None)
                if (best is None) or (current_val_iou > float(best)):
                    setattr(self, "_best_val_iou", current_val_iou)
                    # Export train/val event metrics for this epoch.
                    self._dump_event_iou_csv("train")
                    self._dump_event_iou_csv("val")
                else:
                    # Reset caches to avoid cross-epoch accumulation.
                    self._reset_event_iou_cache("train")
                    self._reset_event_iou_cache("val")
            elif stage == "train":
                # During training, only reset cache; actual export is triggered by val improvement.
                self._reset_event_iou_cache("train")
            elif stage == "test":
                self._dump_event_iou_csv("test")
        except Exception:
            # Safety fallback: export/reset errors should not break training.
            pass

    def _log_hydrology_metrics(self, stage: str, current_recall: torch.Tensor, current_specificity: torch.Tensor) -> None:
        """Log hydrology-related metrics (binary-only)."""
        num_classes = self.hparams.num_classes if hasattr(self.hparams, "num_classes") else 2
        if num_classes != 2 or current_recall.numel() != 2 or current_specificity.numel() != 2:
            logger.debug(
                f"Skip hydrology metrics: num_classes={num_classes},"
                f" recall.shape={tuple(current_recall.shape)}"
            )
            return
        # water_iou: IoU of class 1
        per_class_iou = getattr(self, f"{stage}_iou_pc").compute()
        if per_class_iou.numel() == 2:
            water_iou = per_class_iou[1]
            self.log(f"{stage}/water_iou", water_iou, prog_bar=True)
        # Miss rate / false alarm rate
        flood_recall = current_recall[1]
        flood_miss_rate = 1.0 - flood_recall
        bg_specificity_true = current_specificity[1]
        bg_false_alarm_rate = 1.0 - bg_specificity_true
        self.log(f"{stage}/flood_miss_rate", flood_miss_rate, prog_bar=False)
        self.log(f"{stage}/bg_false_alarm_rate", bg_false_alarm_rate, prog_bar=False)
        # Fine-grained logs
        self.log(f"{stage}/flood_recall", flood_recall, prog_bar=False)
        self.log(f"{stage}/bg_recall", current_recall[0], prog_bar=False)
        self.log(f"{stage}/flood_specificity", current_specificity[1], prog_bar=False)
        self.log(f"{stage}/bg_specificity", bg_specificity_true, prog_bar=False)

    def _reset_metrics(self, stage: str):
        """Reset metrics at epoch end (avoid cross-epoch accumulation)."""
        for name in ("iou", "iou_pc", "f1", "acc", "precision", "recall", "specificity"):
            metric = getattr(self, f"{stage}_{name}")
            metric.reset()

    # ----------------- Event-level IoU accumulation/export -----------------
    def _accumulate_event_iou(self, batch: Dict[str, torch.Tensor], preds: torch.Tensor, masks: torch.Tensor, stage: str) -> None:
        """Accumulate event-level IoU (water_iou numerator/denominator) and export at epoch end.

        Requires `batch["activation"]` (shape [B]) from the dataset.
        """
        act_ids = batch.get("activation")
        if act_ids is None:
            return
        # Ignore invalid pixels
        valid_mask = masks != -1
        if valid_mask.sum() == 0:
            return
        # Only compute IoU for class 1 (water/flood)
        pred_fg = (preds == 1) & valid_mask
        tgt_fg = (masks == 1) & valid_mask
        inter = (pred_fg & tgt_fg).sum(dim=(1, 2)).to(torch.float32)
        union = (pred_fg | tgt_fg).sum(dim=(1, 2)).to(torch.float32)

        # Cache container
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
        # Compute IoU
        import os
        try:
            from lightning_utilities.core.rank_zero import rank_zero_only  # type: ignore[import]
        except Exception:
            def rank_zero_only(fn):  # type: ignore[no-redef]
                return fn

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
        # Clear cache
        setattr(self, cache_name, {})

    def _reset_event_iou_cache(self, stage: str) -> None:
        cache_name = f"_{stage}_event_iou_cache"
        if hasattr(self, cache_name):
            setattr(self, cache_name, {})
    
    def configure_optimizers(self):
        """Configure optimizers."""
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
