"""Encoder utilities and a dual-stream encoder for optical/SAR backbones (timm)."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import logging

from .feature_meta import probe_timm_feature_meta, match_out_indices_by_reduction
from .shallow import ShallowDetailEncoder

logger = logging.getLogger(__name__)

# Internal constants
_DEFAULT_FEATURE_SIZE = (256, 256)


def create_encoder(
    model_name: str,
    in_channels: int,
    pretrained: bool = True,
    name: str = "encoder",
    out_indices: Optional[Tuple[int, ...]] = None,
    output_stride: Optional[int] = None,
    drop_path_rate: Optional[float] = None,  # stochastic depth
    pretrained_cfg_overlay: Optional[dict] = None,
) -> nn.Module:
    """Create a timm backbone with features_only=True and attach feature meta."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required. Please install it (pip install timm).")

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
    # If a local checkpoint is provided, load from it and disable remote pretrained fetch.
    overlay_file: Optional[str] = None
    if isinstance(pretrained_cfg_overlay, dict):
        overlay_file = pretrained_cfg_overlay.get("file")  # type: ignore[assignment]
        if overlay_file:
            # Load from local checkpoint_path and disable pretrained remote fetch.
            kwargs["checkpoint_path"] = overlay_file
            kwargs["pretrained"] = False
        else:
            # Keep other overlay keys if needed.
            kwargs["pretrained_cfg_overlay"] = dict(pretrained_cfg_overlay)

    try:
        model = timm.create_model(model_name, **kwargs)
        level_desc = f"{len(out_indices)} levels" if out_indices is not None else "default levels"
        logger.info(
            f"{name} created ({level_desc}): {model_name}, in_chans={in_channels}, "
            f"drop_path_rate={kwargs.get('drop_path_rate')}"
        )
    except Exception as e:
        # Fallback only when out_indices/output_stride are unsupported; otherwise re-raise.
        msg = str(e).lower()
        is_param_issue = isinstance(e, (TypeError, ValueError)) and (
            ("out_indices" in msg) or ("output_stride" in msg)
        )
        if not is_param_issue:
            raise
        for k in ("output_stride", "out_indices"):
            if k in kwargs:
                kwargs.pop(k)
        logger.info(f"{name}: out_indices/output_stride unsupported; retry without them: {e}")
        model = timm.create_model(model_name, **kwargs)
        logger.info(
            f"{name} created (after fallback): {model_name}, in_chans={in_channels}, "
            f"drop_path_rate={kwargs.get('drop_path_rate')}"
        )

    # Attach channel and reduction metadata.
    model.feature_channels = _get_feature_channels(model, model_name, in_channels)
    model.feature_reductions = _get_feature_reductions(model, model_name, in_channels)

    logger.info(
        f"{name} encoder: {model_name}, in_chans={in_channels}, "
        f"channels={model.feature_channels}, reductions={model.feature_reductions}"
    )
    return model


def _get_feature_channels(encoder: nn.Module, model_name: str, in_channels: int) -> List[int]:
    """Get per-level channel widths (prefer feature_info, else run a small forward probe)."""
    # 1) Prefer feature_info (most reliable).
    if hasattr(encoder, "feature_info"):
        try:
            ch = list(encoder.feature_info.channels())
            if all(isinstance(c, int) and c > 0 for c in ch):
                return ch
        except Exception:
            pass

    # 2) Fallback: forward probe.
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

    raise RuntimeError(f"Failed to infer feature channels for {model_name} (features_only must be usable).")


def _get_feature_reductions(encoder: nn.Module, model_name: str, in_channels: int) -> List[int]:
    """Get per-level spatial reductions (prefer feature_info, else estimate from a forward probe)."""
    # 1) Prefer feature_info.
    if hasattr(encoder, "feature_info"):
        try:
            rd = list(encoder.feature_info.reduction())
            if all(isinstance(r, int) and r > 0 for r in rd):
                return rd[:len(getattr(encoder, "feature_channels", rd))]
        except Exception:
            pass

    # 2) Fallback: estimate from output spatial sizes.
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
                    logger.warning(f"Unexpected feature shape: {f.shape}; fallback reduction=16")
                    rds.append(16)
            return rds
    finally:
        if was_training:
            encoder.train()
    
    raise RuntimeError(f"Failed to infer feature reductions for {model_name} (features_only must be usable).")


class DualStreamEncoder(nn.Module):
    """
    Dual-stream encoder that aligns feature pyramid levels by spatial reduction.
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
        output_stride: Optional[int] = None,  # optional (CNN-only)
        drop_path_rate: Optional[float] = None,  # optional (forwarded to timm)
        pretrained_cfg_overlay: Optional[dict] = None,
        # For single-scale backbones (e.g., ViT/DINOv3), optionally add shallow 4x/8x details.
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

        # Probe feature meta (cached; no pretrained weights).
        probe_opt_ch, probe_opt_rd = self._probe_if_needed(optical_model, optical_channels)
        probe_sar_ch, probe_sar_rd = self._probe_if_needed(sar_model, sar_channels)

        # Choose reference stream: fewer levels wins (tie -> optical).
        len_opt = len(probe_opt_ch) if probe_opt_ch else 1e9
        len_sar = len(probe_sar_ch) if probe_sar_ch else 1e9
        # Reference stream used for alignment.
        ref_stream = "sar" if len_sar < len_opt else "optical"
        logger.info(
            f"reference_stream={ref_stream} (optical_levels={len_opt if len_opt<1e9 else 0}, "
            f"sar_levels={len_sar if len_sar<1e9 else 0})"
        )

        # Build encoders
        self.encoder_optical = None
        self.encoder_sar = None
        self.shallow_opt = None
        self.shallow_sar = None
        
        if ref_stream == "optical":
            # Optical is the reference stream
            if optical_channels > 0:
                self.encoder_optical = create_encoder(
                    model_name=optical_model,
                    in_channels=optical_channels,
                    pretrained=optical_pretrained,
                    name="optical",
                    out_indices=None,  # keep default levels for reference stream
                    output_stride=output_stride,
                    drop_path_rate=drop_path_rate,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                )
            
            # Align the other stream by reduction.
            sar_out_indices = None
            if sar_channels > 0 and probe_opt_rd and probe_sar_rd:
                ref_rd = self.encoder_optical.feature_reductions
                sar_out_indices = match_out_indices_by_reduction(ref_rd, probe_sar_rd)
                logger.info(f"SAR aligned out_indices={sar_out_indices} (ref={ref_rd}, cand={probe_sar_rd})")
            
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
            # SAR is the reference stream
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
                optical_out_indices = match_out_indices_by_reduction(ref_rd, probe_opt_rd)
                logger.info(f"Optical aligned out_indices={optical_out_indices} (ref={ref_rd}, cand={probe_opt_rd})")
            
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

        # Save per-stream channel/reduction meta
        self.optical_feature_channels = getattr(self.encoder_optical, "feature_channels", [])
        self.sar_feature_channels = getattr(self.encoder_sar, "feature_channels", [])
        self.optical_feature_reductions = getattr(self.encoder_optical, "feature_reductions", [])
        self.sar_feature_reductions = getattr(self.encoder_sar, "feature_reductions", [])

        # If enabled and backbone is single-scale, prepend shallow 2/4/8 reductions.
        def _maybe_build_shallow(channels: int, name: str):
            nonlocal shallow_c2, shallow_c4, shallow_c8
            return ShallowDetailEncoder(channels, shallow_c2, shallow_c4, shallow_c8, norm=shallow_norm)

        if shallow_enabled:
            def _is_single_scale(rd: List[int]) -> bool:
                try:
                    return len(set(rd)) == 1
                except Exception:
                    return False

            # optical: treat as single-scale if all reductions are identical
            if self.encoder_optical is not None and _is_single_scale(self.optical_feature_reductions):
                self.shallow_opt = _maybe_build_shallow(optical_channels, "opt")
                # Compose [2,4,8] from shallow and keep the deepest from the backbone.
                self.optical_feature_channels = [shallow_c2, shallow_c4, shallow_c8] + (self.optical_feature_channels[-1:] or [])
                self.optical_feature_reductions = [2, 4, 8] + (self.optical_feature_reductions[-1:] or [16])
                logger.info(
                    f"shallow_enabled(optical): c2={shallow_c2}, c4={shallow_c4}, c8={shallow_c8}; "
                    f"levels=[2,4,8,16]"
                )
            # sar
            if self.encoder_sar is not None and (_is_single_scale(self.sar_feature_reductions) or len(self.sar_feature_reductions) <= 2):
                self.shallow_sar = _maybe_build_shallow(sar_channels, "sar")
                self.sar_feature_channels = [shallow_c2, shallow_c4, shallow_c8] + (self.sar_feature_channels[-1:] or [])
                self.sar_feature_reductions = [2, 4, 8] + (self.sar_feature_reductions[-1:] or [16])
                logger.info(
                    f"shallow_enabled(sar): c2={shallow_c2}, c4={shallow_c4}, c8={shallow_c8}; "
                    f"levels=[2,4,8,16]"
                )

        # Align to the shortest length and keep reductions from the reference stream.
        if self.encoder_optical and self.encoder_sar:
            n = min(len(self.optical_feature_channels), len(self.sar_feature_channels))
            self.optical_feature_channels = self.optical_feature_channels[:n]
            self.sar_feature_channels = self.sar_feature_channels[:n]
            self.feature_channels = [o + s for o, s in zip(self.optical_feature_channels, self.sar_feature_channels)]

            # reductions: take from reference stream
            ref_rd = self.optical_feature_reductions if ref_stream == "optical" else self.sar_feature_reductions
            self.feature_reductions = ref_rd[:n]
        elif self.encoder_optical:
            self.feature_channels = self.optical_feature_channels
            self.feature_reductions = self.optical_feature_reductions
        elif self.encoder_sar:
            self.feature_channels = self.sar_feature_channels
            self.feature_reductions = self.sar_feature_reductions
        else:
            raise RuntimeError("At least one encoder stream must be enabled.")

        logger.info(f"encoder_meta: channels={self.feature_channels}, reductions={self.feature_reductions}")
        logger.info(f"dual_stream_ready: optical={optical_channels}ch({optical_model}), sar={sar_channels}ch({sar_model})")

    def _probe_if_needed(self, model_name: str, channels: int) -> Tuple[List[int], List[int]]:
        """Probe feature meta (cached)."""
        return probe_timm_feature_meta(model_name, channels, out_indices=None) if channels > 0 else ([], [])

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """Forward (supports single-modal and dual-modal inputs)."""
        result = {"optical": [], "sar": []}
        
        if 'image' in batch:
            # Single-modal input
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

        # Dual-modal input
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