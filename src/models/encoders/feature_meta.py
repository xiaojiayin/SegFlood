"""Lightweight feature metadata probing utilities for timm models."""

from typing import List, Optional, Tuple
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=128)
def probe_timm_feature_meta(
    model_name: str,
    in_chans: int,
    out_indices: Optional[Tuple[int, ...]] = None,
) -> Tuple[List[int], List[int]]:
    """Probe timm model feature_info (channels, reductions) without loading pretrained weights."""
    try:
        import timm
    except ImportError:
        logger.warning("timm is not installed; cannot probe feature_meta")
        return [], []

    kwargs = dict(pretrained=False, features_only=True, in_chans=in_chans)
    if out_indices is not None:
        kwargs["out_indices"] = out_indices

    try:
        m = timm.create_model(model_name, **kwargs)
    except Exception as e:
        # Fallback: drop out_indices if unsupported.
        kwargs.pop("out_indices", None)
        try:
            m = timm.create_model(model_name, **kwargs)
            logger.info(f"[probe] {model_name} out_indices unsupported; fallback to default: {e}")
        except Exception as e2:
            logger.warning(f"[probe] Failed to create model {model_name}: {e2}")
            return [], []

    if not hasattr(m, "feature_info"):
        return [], []

    try:
        fi = m.feature_info
        ch = list(fi.channels())
        rd = list(fi.reduction())
        if ch and rd:
            return ch, rd
    except Exception:
        pass

    return [], []


def match_out_indices_by_reduction(
    ref_reductions: List[int],
    cand_reductions: List[int],
) -> Tuple[int, ...]:
    """Match candidate feature levels to reference levels by spatial reduction."""
    if not ref_reductions or not cand_reductions:
        return tuple()

    matched: List[int] = []
    for r in ref_reductions:
        # Choose the smallest cand reduction >= r
        ge = [(i, c) for i, c in enumerate(cand_reductions) if c >= r]
        idx = min(ge, key=lambda kv: kv[1])[0] if ge else (len(cand_reductions) - 1)
        matched.append(idx)
    
    # Sort + deduplicate (keep monotonic hierarchy)
    matched = sorted(set(matched))
    return tuple(matched)