"""WorldFloodsv2 数据集加载器。

目录期望（与官方公开数据一致）：

    root/
      ├── train/
      │    ├── S2/*.tif                # 多波段 Sentinel-2 影像（通常15个波段，int16 DN）
      │    ├── gt/*.tif                # 掩码（多为单/双波段，int16，值域常见 {0,1,2}）
      │    ├── PERMANENTWATERJRC/*.tif # 可选，永久水体年际层
      │    ├── floodmaps/*.geojson     # 事件矢量
      │    └── meta/*.json             # 元数据
      ├── val/ ...
      └── test/ ...

Dataset 只负责：
- 文件匹配、影像读取（可选通道选择）、掩码读取与可选二值化
- 不做归一化、几何增强，全部交由 DataModule 处理

掩码语义：不同版本/子集存在差异，默认假设值 2 表示可训练的“水体/洪水”像素，0 为背景/无效，1 为其它/陆地。
如有差异，请通过 `water_values`、`target_type` 参数配置。
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio as rio  # type: ignore[import]
import torch  # type: ignore[import]
from rasterio.windows import Window  # type: ignore[import]
import math

try:
    from torchgeo.datasets import NonGeoDataset  # type: ignore[import]
    BaseDataset = NonGeoDataset
except Exception:
    from torch.utils.data import Dataset as BaseDataset  # type: ignore[import]


# 参考 third_party/ml4floods 配置：S2 常用13波段的名称顺序（0-based 索引）
BANDS_S2 = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
]

# 通道组合（0-based 基于上面的 BANDS_S2）
CHANNELS_CONFIGURATIONS = {
    "all": list(range(len(BANDS_S2))),
    # 特殊：使用影像内的全部波段（不基于上面固定的 S2 名称顺序），按文件原始顺序读取
    # 通过在构造函数中传入 channels="all_raw" 触发
    "rgb": [3, 2, 1],
    "bgr": [1, 2, 3],
    "bgri": [1, 2, 3, 7],
    "riswir": [3, 7, 11],
    "bgriswir": [1, 2, 3, 7, 11],
    "bgriswirs": [1, 2, 3, 7, 11, 12],
    "l89s2": [0, 1, 2, 3, 7, 10, 11, 12],
    "sub_20": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],
}


class WorldFloodsv2(BaseDataset):
    """WorldFloodsv2 栅格瓦片数据集。

    Args:
        root: 数据根目录，包含 train/val/test 及子文件夹。
        split: "train" | "val" | "test"。
        channels: 指定通道配置（字符串见 CHANNELS_CONFIGURATIONS）或 0-based 通道索引列表
            注意：实际 S2 影像可能有 15 个波段；此处的通道索引按常见的前 13 个 S2 波段映射。
            若你的数据波段顺序不同，请显式提供自定义索引列表。
        target_type: "binary" | "multiclass"。二值时将 `water_values` 合并为 1，其余为 0。
        water_values: 在二值任务中判定为“水体”的掩码值集合，默认 [2]。
        ignore_index: 掩码中视为无效像素的值，将在 DataModule/损失中作为 ignore_index 使用（此处不改写）。
        transforms: 可选后处理（一般交由 DataModule）。
    """

    classes_binary = ["background", "water"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        channels: str | Sequence[int] = "all",
        target_type: str = "binary",
        water_values: Sequence[int] = (2,),
        ignore_index: Optional[int] = -1,
        ignore_values: Optional[Sequence[int]] = (0, 3),
        transforms: Optional[Any] = None,
        add_mndwi_input: bool = False,
    ) -> None:
        super().__init__()
        assert split in {"train", "val", "test"}
        assert target_type in {"binary", "multiclass"}

        self.root = root
        self.split = split
        self.transforms = transforms
        self.target_type = target_type
        self.water_values = set(int(v) for v in water_values)
        self.ignore_index = int(ignore_index) if ignore_index is not None else None
        self.ignore_values = set(int(v) for v in (ignore_values or []))
        self.add_mndwi_input = bool(add_mndwi_input)

        self.dir_S2 = os.path.join(root, split, "S2")
        self.dir_gt = os.path.join(root, split, "gt")
        if not os.path.isdir(self.dir_S2) or not os.path.isdir(self.dir_gt):
            raise FileNotFoundError(f"找不到 S2 或 gt 目录: {self.dir_S2} 或 {self.dir_gt}")

        self.samples = self._scan_samples()

        # 解析通道配置
        if isinstance(channels, str):
            if channels == "all_raw":
                # 使用文件内全部波段；在读取时不执行索引选择
                self.channel_indices = None
            else:
                if channels not in CHANNELS_CONFIGURATIONS:
                    raise ValueError(f"未知通道配置 '{channels}'，可选: {list(CHANNELS_CONFIGURATIONS)} + ['all_raw']")
                self.channel_indices = CHANNELS_CONFIGURATIONS[channels]
        else:
            self.channel_indices = list(int(i) for i in channels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]
        img = self._read_image(item["img"])  # [C, H, W] float32
        msk = self._read_mask(item["msk"])   # [H, W] int64

        if self.target_type == "binary":
            # 二值化：水=1，其它=0
            m_bin = np.isin(msk, list(self.water_values)).astype(np.int64)
            # 忽略：将原始标签中属于 ignore_values 的位置置为 ignore_index
            if self.ignore_index is not None and len(self.ignore_values) > 0:
                ign_mask = np.zeros_like(msk, dtype=bool)
                for v in self.ignore_values:
                    ign_mask |= (msk == v)
                m_bin = m_bin.astype(np.int64)
                m_bin[ign_mask] = int(self.ignore_index)
            msk = m_bin
        else:
            # 多分类：保持原值（通常 {0,1,2}）
            msk = msk.astype(np.int64)

        # 注意：torch.from_numpy 与 rasterio 返回的数组可能共享只读/不可调整的底层存储，
        # 在 DataLoader 默认 collate 下会触发 "storage not resizable" 错误。
        # 通过 clone() 生成独立、可调整大小的张量存储以避免该问题。
        sample: Dict[str, Any] = {
            "image": torch.from_numpy(img.astype(np.float32)).clone().contiguous(),
            "mask": torch.from_numpy(msk).clone().contiguous(),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    # -------------------- internals --------------------
    def _scan_samples(self) -> List[Dict[str, str]]:
        s2_files = sorted(glob.glob(os.path.join(self.dir_S2, "*.tif")))
        gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(self.dir_gt, "*.tif"))}
        pairs: List[Dict[str, str]] = []
        for p in s2_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            gt_path = gt_map.get(stem)
            if gt_path is not None:
                pairs.append({"img": p, "msk": gt_path})
        if len(pairs) == 0:
            raise RuntimeError(f"在 {self.dir_S2} 下没有找到与 gt 配对的样本")
        return pairs

    def _read_image(self, path: str) -> np.ndarray:
        with rio.open(path) as src:
            arr_full = src.read()  # [B, H, W], int16 DN
        if arr_full.ndim == 2:
            arr_full = arr_full[None, ...]
        arr_full = arr_full.astype(np.float32)

        # 先按配置选择通道
        if self.channel_indices is not None and len(self.channel_indices) > 0:
            max_idx = int(np.max(self.channel_indices)) if len(self.channel_indices) > 0 else -1
            if arr_full.shape[0] <= max_idx:
                valid = [i for i in self.channel_indices if i < arr_full.shape[0]]
            else:
                valid = self.channel_indices
            arr = arr_full[valid]
        else:
            arr = arr_full

        # 可选：追加 MNDWI 通道（基于 B3/B11 原始波段）
        if self.add_mndwi_input and (arr_full.shape[0] > 11):
            g = arr_full[2]
            swir = arr_full[11]
            denom = g + swir
            denom = np.where(denom == 0, 1.0, denom)
            mndwi = (g - swir) / denom
            arr = np.concatenate([arr, mndwi[None, ...].astype(np.float32)], axis=0)

        return arr

    def _read_mask(self, path: str) -> np.ndarray:
        # gt 有时为 1 或 2 个波段。严格优先：若有第2通道，则使用第2通道作为“水”语义；否则退回启发式。
        with rio.open(path) as src:
            if src.count == 1:
                m = src.read(1)
            else:
                bands = src.read()  # [B, H, W]
                if bands.shape[0] >= 2:
                    m = bands[1]
                else:
                    # 退回启发式选择
                    chosen: Optional[np.ndarray] = None
                    for b in bands:
                        uniq = np.unique(b)
                        if (b.sum() != 0) and (uniq.size > 1):
                            chosen = b
                            break
                    if chosen is None:
                        chosen = bands[0]
                    m = chosen
        # 转为整型标签
        m = np.nan_to_num(m).astype(np.int64)
        return m


# -------------------- Online tiling (官方风格) --------------------
def get_list_of_window_slices(
    pairs: List[Dict[str, str]], window_size: Tuple[int, int]
) -> List[Dict[str, Any]]:
    """为每个 (img, msk) 文件对生成无重叠窗口的切片列表。

    返回的每个元素包含：{"img": path_img, "msk": path_msk, "row": r, "col": c, "window": Window}
    """
    list_of_windows: List[Dict[str, Any]] = []
    wh, ww = int(window_size[0]), int(window_size[1])
    for p in pairs:
        img_path, msk_path = p["img"], p["msk"]
        with rio.open(img_path) as src:
            H, W = src.height, src.width
        # 使用 ceil 覆盖到边界，配合 boundless=True 读取实现零填充
        rows = math.ceil(H / wh)
        cols = math.ceil(W / ww)
        for r in range(rows):
            for c in range(cols):
                y0, x0 = r * wh, c * ww
                win = Window(col_off=x0, row_off=y0, width=ww, height=wh)
                list_of_windows.append({
                    "img": img_path,
                    "msk": msk_path,
                    "row": r,
                    "col": c,
                    "window": win,
                })
    return list_of_windows


class WorldFloodsv2Tiled(BaseDataset):
    """在线切片的 Tiled 数据集（对齐官方 WorldFloodsDatasetTiled）。

    Args:
        list_of_windows: 由 get_list_of_window_slices 生成的窗口列表
        channels: 通道配置（字符串或索引列表）
        target_type: "binary" | "multiclass"
        water_values: 二值任务中判定为水体的标签集合
        transforms: 可选增强（一般交给 DataModule）
    """

    def __init__(
        self,
        list_of_windows: List[Dict[str, Any]],
        channels: str | Sequence[int] = "all",
        target_type: str = "binary",
        water_values: Sequence[int] = (2,),
        transforms: Optional[Any] = None,
        add_mndwi_input: bool = False,
        ignore_index: Optional[int] = -1,
        ignore_values: Optional[Sequence[int]] = (0, 3),
    ) -> None:
        super().__init__()
        self.list_of_windows = list_of_windows
        self.transforms = transforms
        self.target_type = target_type
        self.water_values = set(int(v) for v in water_values)
        self.add_mndwi_input = bool(add_mndwi_input)
        self.ignore_index = int(ignore_index) if ignore_index is not None else None
        self.ignore_values = set(int(v) for v in (ignore_values or []))

        if isinstance(channels, str):
            if channels == "all_raw":
                self.channel_indices = None
            else:
                if channels not in CHANNELS_CONFIGURATIONS:
                    raise ValueError(f"未知通道配置 '{channels}'，可选: {list(CHANNELS_CONFIGURATIONS)} + ['all_raw']")
                self.channel_indices = CHANNELS_CONFIGURATIONS[channels]
        else:
            self.channel_indices = list(int(i) for i in channels)

    def __len__(self) -> int:
        return len(self.list_of_windows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        info = self.list_of_windows[index]
        img_path, msk_path, win = info["img"], info["msk"], info["window"]
        with rio.open(img_path) as src:
            # 与官方 tiled 读取保持一致：越界时填 0
            arr_full = src.read(window=win, boundless=True, fill_value=0)
        with rio.open(msk_path) as src_m:
            if src_m.count == 1:
                m = src_m.read(1, window=win, boundless=True, fill_value=0)
            else:
                bands = src_m.read(window=win, boundless=True, fill_value=0)
                if bands.shape[0] >= 2:
                    m = bands[1]
                else:
                    # 退回启发式选择
                    chosen: Optional[np.ndarray] = None
                    for b in bands:
                        uniq = np.unique(b)
                        if (b.sum() != 0) and (uniq.size > 1):
                            chosen = b
                            break
                    if chosen is None:
                        chosen = bands[0]
                    m = chosen

        if arr_full.ndim == 2:
            arr_full = arr_full[None, ...]
        arr_full = arr_full.astype(np.float32)
        if self.channel_indices is not None and len(self.channel_indices) > 0:
            max_idx = int(np.max(self.channel_indices)) if len(self.channel_indices) > 0 else -1
            if arr_full.shape[0] <= max_idx:
                valid = [i for i in self.channel_indices if i < arr_full.shape[0]]
            else:
                valid = self.channel_indices
            arr = arr_full[valid]
        else:
            arr = arr_full

        if self.add_mndwi_input and (arr_full.shape[0] > 11):
            g = arr_full[2]
            swir = arr_full[11]
            denom = g + swir
            denom = np.where(denom == 0, 1.0, denom)
            mndwi = (g - swir) / denom
            arr = np.concatenate([arr, mndwi[None, ...].astype(np.float32)], axis=0)

        if m is None:
            m = np.zeros((arr.shape[-2], arr.shape[-1]), dtype=np.int64)
        if self.target_type == "binary":
            m_bin = np.isin(m, list(self.water_values)).astype(np.int64)
            if self.ignore_index is not None and len(self.ignore_values) > 0:
                ign_mask = np.zeros_like(m, dtype=bool)
                for v in self.ignore_values:
                    ign_mask |= (m == v)
                m_bin = m_bin.astype(np.int64)
                m_bin[ign_mask] = int(self.ignore_index)
            m = m_bin
        else:
            m = m.astype(np.int64)

        # 同上，避免 DataLoader 堆叠时对只读/不可调整存储的张量 resize 报错
        sample: Dict[str, Any] = {
            "image": torch.from_numpy(arr.astype(np.float32)).clone().contiguous(),
            "mask": torch.from_numpy(m).clone().contiguous(),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
