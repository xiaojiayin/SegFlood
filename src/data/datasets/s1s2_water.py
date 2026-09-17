"""S1S2-Water 数据集加载器。

支持经 `scripts/data_pre/prepare_s1s2_water_tiles.py` 生成的统一多通道 tiles 目录结构::

        root/
            ├── train/{images,masks}/ *.tif
            ├── val/{images,masks}/   *.tif
            └── test/{images,masks}/  *.tif

每个 `images/*.tif` 为多通道堆叠，**通道顺序不再写死**，而是通过
`s1s2_water_metadata.json` 或 `channel_stats.json` 中的 `band_order` 自适应解析，
典型顺序为::

    [S2_B0..S2_B5, S1_VV, S1_VH, DEM_ELEVATION, DEM_SLOPE]

模态控制：
- optical: 返回 RGB+NIR + 可选 DEM/SLOPE（追加在后）
- sar: 仅返回 VV,VH  
- dual: 返回两路分支

Dataset 仅做：文件匹配、读取、通道拆分、mask 二值化；不做归一化/增强。
"""

import glob
import os
import re
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import rasterio as rio
import torch

try:
    from torchgeo.datasets import NonGeoDataset  # type: ignore[import]
    BaseDataset = NonGeoDataset
except Exception:
    from torch.utils.data import Dataset as BaseDataset


class S1S2Water(BaseDataset):
    """S1S2-Water 预处理瓦片数据集。

    Args:
        root: 统一多通道根目录（包含 train/val/test 子目录）。
        split: 数据划分，"train" | "val" | "test"。
        modal_type: "sar" | "optical" | "dual"。
        add_dem: 是否添加 DEM_ELEVATION 到光学分支，默认 False。
        add_slope: 是否添加 DEM_SLOPE 到光学分支，默认 False。
        transforms: 可选，后处理/转换（一般留空，由DataModule统一处理）。
    """

    classes = ["background", "water"]

    def __init__(
        self,
        root: Optional[str] = None,
        split: str = "train",
        modal_type: str = "dual",
        add_dem: bool = False,
        add_slope: bool = False,
        transforms: Optional[Any] = None,
    ) -> None:
        super().__init__()

        assert split in {"train", "val", "test"}
        assert modal_type in {"sar", "optical", "dual"}

        self.transforms = transforms
        self.split = split
        self.modal_type = modal_type
        self.add_dem = add_dem
        self.add_slope = add_slope

        if not root:
            raise ValueError("需要提供 root (统一多通道 tiles 根目录)")
        
        images_dir = os.path.join(root, split, "images")
        masks_dir = os.path.join(root, split, "masks")
        if not os.path.isdir(images_dir) or not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"统一目录缺失: {images_dir} 或 {masks_dir}")
        
        self.root = root
        self.samples = self._build_samples(root, split)
        if len(self.samples) == 0:
            raise RuntimeError(f"S1S2-Water: 在 {images_dir} 下未找到任何样本")

        # 读取 band_order 元数据并建立名称->索引映射（0-based）
        self._load_band_metadata()
        self._init_band_indices()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.samples[index]

        img = _read_tiff(item["img"])  # [C, H, W]
        mask = _read_mask(item["msk"])  # [H, W]

        if self.modal_type == "dual":
            optical = img[self._optical_indices]
            sar = img[self._sar_indices]
            sample = {
                "image_optical": torch.from_numpy(optical.astype(np.float32)),
                "image_sar": torch.from_numpy(sar.astype(np.float32)),
                "mask": torch.from_numpy(mask.astype(np.int64)),
            }
        elif self.modal_type == "optical":
            optical = img[self._optical_indices]
            sample = {
                "image_optical": torch.from_numpy(optical.astype(np.float32)),
                "mask": torch.from_numpy(mask.astype(np.int64)),
            }
        else:  # sar
            sar = img[self._sar_indices]
            sample = {
                "image_sar": torch.from_numpy(sar.astype(np.float32)),
                "mask": torch.from_numpy(mask.astype(np.int64)),
            }

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _build_samples(self, root: str, split: str) -> List[Dict[str, str]]:
        img_dir = os.path.join(root, split, "images")
        msk_dir = os.path.join(root, split, "masks")
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.tif")))
        msk_files = {os.path.basename(p): p for p in glob.glob(os.path.join(msk_dir, "*.tif"))}
        samples: List[Dict[str, str]] = []
        for p in img_files:
            name = os.path.basename(p)
            if name in msk_files:
                samples.append({"img": p, "msk": msk_files[name]})
        return samples

    # ------------------------------------------------------------------
    # band_order / 索引解析
    # ------------------------------------------------------------------

    def _load_band_metadata(self) -> None:
        """从 root 目录读取 band_order 元数据，并与实际通道数对齐。

        优先使用:
        - s1s2_water_metadata.json
        - channel_stats.json

        若均不存在，则退化为默认顺序:
        [S2_B0, S2_B1, S2_B2, S2_B3, S1_VV, S1_VH, DEM_ELEVATION, DEM_SLOPE]
        """
        root = self.root
        candidates = [
            os.path.join(root, "s1s2_water_metadata.json"),
            os.path.join(root, "S1S2_water_metadata.json"),
            os.path.join(root, "S1S2-Water_metadata.json"),
            os.path.join(root, "channel_stats.json"),
        ]
        band_order: Optional[List[str]] = None
        meta_path: Optional[str] = None
        for p in candidates:
            if not os.path.isfile(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                bo = meta.get("band_order", None)
                if isinstance(bo, list) and all(isinstance(x, str) for x in bo):
                    band_order = bo
                    meta_path = p
                    break
            except Exception:
                continue

        # 若元数据缺失，则使用保守默认顺序（兼容早期 8 通道 tiles）
        if band_order is None:
            band_order = [
                "S2_B0",
                "S2_B1",
                "S2_B2",
                "S2_B3",
                "S1_VV",
                "S1_VH",
                "DEM_ELEVATION",
                "DEM_SLOPE",
            ]
            meta_path = "<default>"

        # 与实际影像通道数对齐（仅用于截断/填补 UNK，占位不参与索引）
        first_img = self.samples[0]["img"]
        arr = _read_tiff(first_img)
        num_bands = int(arr.shape[0])
        if len(band_order) != num_bands:
            print(
                f"[S1S2Water] WARNING: band_order 长度={len(band_order)} 与样本通道数={num_bands} 不一致，"
                f"meta={meta_path}, sample={os.path.basename(first_img)}"
            )
            if len(band_order) > num_bands:
                band_order = band_order[:num_bands]
            else:
                # 为多余通道生成占位名，后续不会被显式索引
                for i in range(len(band_order), num_bands):
                    band_order.append(f"UNK_{i}")

        self.band_order: List[str] = band_order
        # 0-based 索引
        self.band_to_index: Dict[str, int] = {name: i for i, name in enumerate(self.band_order)}
        print(f"[S1S2Water] band_order (meta={meta_path}): {self.band_order}")

    def _init_band_indices(self) -> None:
        """根据 band_order 初始化各模态通道索引."""

        if not hasattr(self, "band_order") or not hasattr(self, "band_to_index"):
            raise RuntimeError("S1S2-Water: band_order 尚未初始化")

        def _idx(name: str) -> Optional[int]:
            return self.band_to_index.get(name, None)

        # 光学基础通道：始终按 RGB+NIR 顺序追加
        optical_names = ["S2_B0", "S2_B1", "S2_B2", "S2_B3"]
        sar_names = ["S1_VV", "S1_VH"]

        self._optical_indices: List[int] = []
        for n in optical_names:
            i = _idx(n)
            if i is not None:
                self._optical_indices.append(i)

        self._sar_indices: List[int] = []
        for n in sar_names:
            i = _idx(n)
            if i is not None:
                self._sar_indices.append(i)

        base_opt_len = len(self._optical_indices)

        if self.modal_type in {"dual", "optical"} and base_opt_len == 0:
            raise RuntimeError(
                f"S1S2-Water: 未在 band_order 中找到任何光学通道 {optical_names}，"
                f"当前 band_order={self.band_order}"
            )
        if self.modal_type in {"dual", "sar"} and len(self._sar_indices) == 0:
            raise RuntimeError(
                f"S1S2-Water: 未在 band_order 中找到任何 SAR 通道 {sar_names}，"
                f"当前 band_order={self.band_order}"
            )

        # DEM / SLOPE 可选通道：仅当存在且 add_dem/add_slope=True 时追加到光学分支
        dem_idx = _idx("DEM_ELEVATION")
        slope_idx = _idx("DEM_SLOPE")

        if self.add_dem:
            if dem_idx is None:
                raise RuntimeError(
                    "S1S2-Water: add_dem=True 但 band_order 中没有 'DEM_ELEVATION'，"
                    f"当前 band_order={self.band_order}"
                )
            self._optical_indices.append(dem_idx)
        if self.add_slope:
            if slope_idx is None:
                raise RuntimeError(
                    "S1S2-Water: add_slope=True 但 band_order 中没有 'DEM_SLOPE'，"
                    f"当前 band_order={self.band_order}"
                )
            self._optical_indices.append(slope_idx)

        # 记录实际通道数，便于 DataModule / 实验配置交叉校验
        self.optical_channels: int = len(self._optical_indices)
        self.sar_channels: int = len(self._sar_indices)

        print(
            "[S1S2Water] 解析通道索引:"
            f" optical_indices={self._optical_indices} (count={self.optical_channels}),"
            f" sar_indices={self._sar_indices} (count={self.sar_channels}),"
            f" add_dem={self.add_dem}, add_slope={self.add_slope}"
        )


def _read_tiff(path: str) -> np.ndarray:
    """读取多通道 GeoTIFF → [C, H, W] float32。"""
    with rio.open(path) as src:
        arr = src.read().astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def _read_mask(path: str) -> np.ndarray:
    """读取掩码 → [H, W]，通用口径：>0 视为水体；255 视为无效 → {-1,0,1}。"""
    with rio.open(path) as src:
        if src.count >= 1:
            arr = src.read(1)
        else:
            arr = src.read()[0]
    m = (arr > 0).astype(np.int64)
    m[arr == 255] = -1
    return m