"""
特征元数据工具（feature_meta）：跨架构特征层对齐的轻量探测

职责
- 以轻量方式探测 timm 模型的 feature_info（channels, reductions）
- 依据降采样倍率（reduction）匹配不同架构的层级
- 使用 LRU 缓存避免重复创建模型

说明
- 不加载预训练权重，仅用于元数据探测与对齐决策
- 若自定义 out_indices 不被支持，自动回退到默认层级
"""

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
    """
    轻量探测 timm 模型的 feature_info (channels, reductions)，不加载预训练权重。
    结果做了 LRU 缓存：key=(model_name, in_chans, out_indices)。
    
    Args:
        model_name: timm模型名称
        in_chans: 输入通道数
        out_indices: 指定输出层级（可选）
        
    Returns:
        (channels_list, reductions_list): 特征通道数和空间降采样倍率
        
    注意：
        - 不加载预训练权重，避免网络开销
        - 仅用于"决定如何裁剪 out_indices"，不用于正式推理/训练
        - 若 out_indices 不被支持，会自动回退到默认层级
        - 结果会被缓存，相同参数的后续调用直接返回
    """
    try:
        import timm
    except ImportError:
        logger.warning("timm 未安装，无法探测 feature_meta")
        return [], []

    kwargs = dict(
        pretrained=False,          # 不加载权重，避免网络与开销
        features_only=True,
        in_chans=in_chans,
    )
    if out_indices is not None:
        kwargs["out_indices"] = out_indices

    try:
        m = timm.create_model(model_name, **kwargs)
    except Exception as e:
        # 回退：删除不支持的 out_indices
        kwargs.pop("out_indices", None)
        try:
            m = timm.create_model(model_name, **kwargs)
            logger.info(f"[probe] {model_name} 自定义 out_indices 不支持，回退默认层级: {e}")
        except Exception as e2:
            logger.warning(f"[probe] 无法创建模型 {model_name}: {e2}")
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


@lru_cache(maxsize=128)
def probe_timm_default_out_indices(model_name: str, in_chans: int) -> Tuple[int, ...]:
    """探测 timm features_only 默认输出层对应的真实 block/stage 索引。

    对 CNN 通常为 (0,1,2,3,4)；对 ViT 类单尺度骨干通常为最后几个 block，
    例如 vit_small_patch16 为 (9,10,11)。若无法获取则返回空元组。
    """
    try:
        import timm
        m = timm.create_model(model_name, pretrained=False, features_only=True, in_chans=in_chans)
        fi = getattr(m, "feature_info", None)
        idx = getattr(fi, "out_indices", None)
        if idx:
            return tuple(int(i) for i in idx)
    except Exception as e:
        logger.info(f"[probe] 无法获取 {model_name} 默认 out_indices: {e}")
    return tuple()


def match_out_indices_by_reduction(
    ref_reductions: List[int],
    cand_reductions: List[int],
    cand_out_indices: Optional[Tuple[int, ...]] = None,
) -> Optional[Tuple[int, ...]]:
    """
    依据 reduction 将候选模型层与参考模型层匹配，返回应传给 timm 的 out_indices。

    Args:
        ref_reductions: 参考模型的降采样倍率列表
        cand_reductions: 候选模型（默认层级）的降采样倍率列表
        cand_out_indices: 候选模型默认层级对应的真实 block/stage 索引；
            匹配得到的"列表位置"必须经此映射，否则对 ViT 会被 timm 当成
            block 0..k 并裁掉其后全部 block（历史 DINOv3 双流 SAR 分支即因此
            只保留了 1 个 block）。

    Returns:
        None 表示"使用候选模型默认层级"；否则为升序、去重的真实索引元组。

    策略：
        - 两侧 reduction 列表完全相同（如两个 ViT 均为 [16,16,16]）→ 直接用默认层级；
        - 否则对每个 r ∈ ref，在 cand 中取 >= r 的最小 reduction；若存在多个相同
          reduction，取其中最深的一层；若不存在，取 cand 的最后一层。
    """
    if not ref_reductions or not cand_reductions:
        return None
    if list(ref_reductions) == list(cand_reductions):
        return None

    positions: List[int] = []
    for r in ref_reductions:
        ge = [(i, c) for i, c in enumerate(cand_reductions) if c >= r]
        if ge:
            best = min(c for _, c in ge)
            pos = max(i for i, c in ge if c == best)
        else:
            pos = len(cand_reductions) - 1
        positions.append(pos)
    positions = sorted(set(positions))

    if cand_out_indices and len(cand_out_indices) == len(cand_reductions):
        return tuple(int(cand_out_indices[p]) for p in positions)
    # 无法映射到真实索引时退回位置索引（CNN 默认 out_indices=(0..k)，二者一致）
    return tuple(positions)