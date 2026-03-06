"""WorldFloods v2 dataset: Sentinel-2 L1C optical flood and surface-water segmentation."""

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Band channel configurations (0-based Sentinel-2 band indices)
# ---------------------------------------------------------------------------

#: Named channel subsets for Sentinel-2 L1C (1-based GeoTIFF bands -> 0-based indices).
#: ``bgri``  = B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
#: ``rgb``   = B4, B3, B2 (display order)
#: ``bgr``   = B2, B3, B4
#: ``bgriswirs`` = BGRI + B11 (SWIR1) + B12 (SWIR2)
#: ``all``   = All 13 bands
CHANNELS_CONFIGURATIONS: Dict[str, List[int]] = {
    "bgri":      [1, 2, 3, 7],          # B2 B3 B4 B8  (4 bands)
    "rgb":       [3, 2, 1],             # B4 B3 B2      (3 bands)
    "bgr":       [1, 2, 3],             # B2 B3 B4      (3 bands)
    "bgriswirs": [1, 2, 3, 7, 10, 11],  # BGRI+SWIR     (6 bands)
    "all":       list(range(13)),        # All S2 bands  (13 bands)
}

# Official Sentinel-2 z-score normalization constants (from the WorldFloods project).
_WF2_OFFICIAL_MEAN: Tuple[float, ...] = (
    1353.0, 1116.8, 1041.5, 945.1,
    1198.7, 2004.4, 2376.0, 2303.2,
    732.5,  12.3,   1819.5, 1115.3,
    2602.1,
)
_WF2_OFFICIAL_STD: Tuple[float, ...] = (
    65.22,  154.08, 187.31, 278.42,
    228.29, 356.75, 456.95, 533.59,
    140.78, 4.44,   358.26, 291.95,
    540.49,
)


class WorldFloodsv2Dataset(Dataset):
    """
    Patch (or full-scene) dataset for WorldFloods v2.

    Directory layout under ``root``::

        root/
          train/
            S2L1C/    (images, GeoTIFF, 13 bands)
            gt/       (masks, GeoTIFF, 1 band)
          val/
            S2L1C/
            gt/
          test/
            S2L1C/
            gt/

    Mask semantics (raw):
    * 0 – no data / invalid (mapped to ``ignore_index``)
    * 1 – land
    * 2 – flood / water (default ``water_values``)
    * 3 – cloud (mapped to ``ignore_index``)

    When ``target_type="binary"`` the mask is remapped to ``{0, 1, ignore_index}``.

    The ``samples`` attribute (list of ``{"img": path, "msk": path}`` dicts) is
    used by :class:`~src.infer.worldfloodsv2.WorldFloodsPredictWriter`.
    """

    def __init__(
        self,
        root: str,
        split: str = "test",
        channels: str = "bgri",
        target_type: str = "binary",
        water_values: Optional[List[int]] = None,
        ignore_index: int = -1,
        ignore_values: Optional[List[int]] = None,
        official_normalization: bool = True,
        normalization_mean: Optional[List[float]] = None,
        normalization_std: Optional[List[float]] = None,
        add_mndwi_input: bool = False,
        test_use_tiles: bool = True,
        window_size: Optional[List[int]] = None,
        sliding_window: Optional[Dict] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.root = root
        self.split = split
        self.target_type = target_type
        self.water_values: List[int] = water_values if water_values is not None else [2]
        self.ignore_index = ignore_index
        self.ignore_values: List[int] = ignore_values if ignore_values is not None else [0, 3]
        self.add_mndwi_input = add_mndwi_input
        self.test_use_tiles = test_use_tiles
        self.window_size = window_size or [1024, 1024]
        self.sliding_window = sliding_window or {}
        self.transform = transform

        # Resolve channel indices
        if isinstance(channels, str):
            self.channel_indices = CHANNELS_CONFIGURATIONS.get(channels, CHANNELS_CONFIGURATIONS["bgri"])
            self.channels = channels
        else:
            self.channel_indices = [int(i) for i in channels]
            self.channels = "custom"

        # Normalization
        if normalization_mean is not None and normalization_std is not None:
            self._mean = list(normalization_mean)
            self._std = list(normalization_std)
        elif official_normalization:
            self._mean = [_WF2_OFFICIAL_MEAN[i] for i in self.channel_indices]
            self._std = [_WF2_OFFICIAL_STD[i] for i in self.channel_indices]
        else:
            n = len(self.channel_indices)
            self._mean = [0.0] * n
            self._std = [1.0] * n

        # Add MNDWI extra channel (appended after primary channels)
        self._n_channels = len(self.channel_indices) + (1 if add_mndwi_input else 0)

        # Discover samples
        img_dir = os.path.join(root, split, "S2L1C")
        msk_dir = os.path.join(root, split, "gt")
        self.samples: List[Dict[str, str]] = []
        if os.path.isdir(img_dir):
            for fname in sorted(os.listdir(img_dir)):
                if not fname.endswith(".tif"):
                    continue
                img_path = os.path.join(img_dir, fname)
                msk_path = os.path.join(msk_dir, fname)
                self.samples.append({"img": img_path, "msk": msk_path})

    @property
    def optical_channels(self) -> int:
        return self._n_channels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            import rasterio  # type: ignore
        except ImportError:
            raise ImportError("rasterio is required. Install it with: pip install rasterio")

        entry = self.samples[idx]
        img_path = entry["img"]
        msk_path = entry["msk"]

        # Load image bands
        with rasterio.open(img_path) as src:
            bands = [src.read(i + 1).astype(np.float32) for i in self.channel_indices if (i + 1) <= src.count]
        arr = np.stack(bands, axis=0) if bands else np.zeros((len(self.channel_indices), 256, 256), dtype=np.float32)

        # Normalize
        for i in range(arr.shape[0]):
            m = self._mean[i] if i < len(self._mean) else 0.0
            s = self._std[i] if i < len(self._std) else 1.0
            arr[i] = (arr[i] - m) / max(s, 1e-6)

        # Optional MNDWI: (Green - SWIR1) / (Green + SWIR1)
        if self.add_mndwi_input:
            try:
                g_idx = self.channel_indices.index(2)   # B3 (Green), 0-based
                sw_idx = self.channel_indices.index(10)  # B11 (SWIR1), 0-based
                green = arr[g_idx]
                swir = arr[sw_idx]
                denom = np.abs(green) + np.abs(swir) + 1e-6
                mndwi = (green - swir) / denom
                arr = np.concatenate([arr, mndwi[None]], axis=0)
            except ValueError:
                arr = np.concatenate([arr, np.zeros((1,) + arr.shape[1:], dtype=np.float32)], axis=0)

        # Load mask
        if os.path.isfile(msk_path):
            with rasterio.open(msk_path) as src:
                raw_mask = src.read(1).astype(np.int64)
        else:
            raw_mask = np.zeros(arr.shape[1:], dtype=np.int64)

        mask = self._remap_mask(raw_mask)

        sample: Dict = {
            "image": torch.from_numpy(arr),
            "image_optical": torch.from_numpy(arr),
            "mask": torch.from_numpy(mask),
            "meta": {"sample_name": os.path.splitext(os.path.basename(img_path))[0]},
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _remap_mask(self, raw: np.ndarray) -> np.ndarray:
        """Remap raw WorldFloods mask to {0, 1, ignore_index}."""
        out = np.zeros_like(raw, dtype=np.int64)
        # Ignore pixels
        for v in self.ignore_values:
            out[raw == v] = self.ignore_index
        # Water/flood pixels
        for v in self.water_values:
            out[raw == v] = 1
        return out
