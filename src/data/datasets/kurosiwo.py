"""Kuro Siwo dataset: Sentinel-1 GRD SAR (VV/VH) + optional DEM for water mapping."""

import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Global z-score statistics (Sentinel-1 GRD, dB scale) derived from the Kuro Siwo dataset.
_VV_MEAN: float = -11.8709
_VV_STD: float = 6.8735
_VH_MEAN: float = -19.0754
_VH_STD: float = 7.2449
_DEM_MEAN: float = 93.4313
_DEM_STD: float = 1410.8382


class KuroSiwoDataset(Dataset):
    """
    Patch-based dataset for Kuro Siwo (SAR-only water mapping).

    The dataset is stored as a compressed-pickle index (``KuroV2_grid_dict.gz``)
    produced by :mod:`scripts.data_pre.export_kurosiwo_pickle`, alongside
    per-sample GeoTIFF tiles under ``root/``.

    Each record in the pickle maps a sample ID to::

        {
            "path": "relative/path/to/tile.tif",   # 2-band (VV, VH) or 3-band (+DEM)
            "clz":  int,                            # KuroSiwo split assignment
            "info": {"actid": str, ...},
        }

    A companion label file (``<stem>_label.tif`` or ``<stem>_mask.tif``) is
    expected in the same directory.

    Returned dict keys:

    * ``image_sar`` – ``float32 (C_sar, H, W)``  (dB-scaled, z-score normalised)
    * ``mask``      – ``int64 (H, W)``  (0=non-water / 1=water / -1=ignore)
    * ``meta``      – ``{"sample_name": str, "event_id": str}``
    """

    # clz values for each split (Kuro Siwo V2 convention).
    SPLIT_CLZ: Dict[str, Tuple[int, ...]] = {
        "train": (0,),
        "val":   (1,),
        "test":  (2,),
    }

    def __init__(
        self,
        root: str,
        split: str = "train",
        sar_channels: int = 2,
        use_dem: bool = False,
        zscore: bool = True,
        pickle_name: str = "KuroV2_grid_dict.gz",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.root = root
        self.split = split
        self.sar_channels = sar_channels
        self.use_dem = use_dem
        self.zscore = zscore
        self.transform = transform

        # Load the pickle index (compressed pickle).
        pickle_path = os.path.join(root, "pickle", pickle_name)
        if not os.path.isfile(pickle_path):
            # Fallback: look for the pickle directly in root.
            pickle_path = os.path.join(root, pickle_name)

        self._index: List[Dict[str, Any]] = []
        self.sample_ids: List[str] = []

        if os.path.isfile(pickle_path):
            grids = _load_pickle(pickle_path)
            target_clzs = self.SPLIT_CLZ.get(split, (0,))
            for sample_id, rec in grids.items():
                clz = rec.get("clz")
                if clz in target_clzs:
                    self._index.append({"id": str(sample_id), "rec": rec})
                    self.sample_ids.append(str(sample_id))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            import rasterio  # type: ignore
        except ImportError:
            raise ImportError("rasterio is required. Install it with: pip install rasterio")

        entry = self._index[idx]
        sample_id = entry["id"]
        rec = entry["rec"]

        rel_path: Optional[str] = rec.get("path")
        if rel_path is None:
            raise RuntimeError(f"No 'path' key in Kuro Siwo record: {sample_id}")
        img_path = os.path.join(self.root, rel_path) if not os.path.isabs(rel_path) else rel_path

        # Read SAR bands.
        with rasterio.open(img_path) as src:
            n_read = min(src.count, 2 if not self.use_dem else 3)
            arr = src.read(list(range(1, n_read + 1))).astype(np.float32)

        sar = arr[:2]  # VV, VH
        sar = _to_db(sar)
        if self.zscore:
            sar[0] = (sar[0] - _VV_MEAN) / max(_VV_STD, 1e-6)
            sar[1] = (sar[1] - _VH_MEAN) / max(_VH_STD, 1e-6)

        if self.use_dem and arr.shape[0] >= 3:
            dem = arr[2:3]
            dem = (dem - _DEM_MEAN) / max(_DEM_STD, 1e-6)
            sar = np.concatenate([sar, dem], axis=0)

        # Read label.
        label_path = self._resolve_label_path(img_path)
        if label_path and os.path.isfile(label_path):
            with rasterio.open(label_path) as src:
                mask = src.read(1).astype(np.int64)
        else:
            h, w = sar.shape[1], sar.shape[2]
            mask = np.full((h, w), -1, dtype=np.int64)

        actid = str((rec.get("info") or {}).get("actid", "unknown"))
        sample: Dict[str, Any] = {
            "image_sar": torch.from_numpy(sar),
            "mask": torch.from_numpy(mask),
            "meta": {"sample_name": sample_id, "event_id": actid},
        }
        # Single-modal convenience alias
        sample["image"] = sample["image_sar"]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_label_path(img_path: str) -> Optional[str]:
        stem = os.path.splitext(img_path)[0]
        for suffix in ("_label.tif", "_mask.tif", "_label.tiff", "_mask.tiff"):
            cand = stem + suffix
            if os.path.isfile(cand):
                return cand
        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: str) -> Dict[str, Any]:
    """Load a (possibly compress_pickle compressed) pickle file."""
    try:
        from compress_pickle import load  # type: ignore
        with open(path, "rb") as f:
            return load(f)
    except ImportError:
        pass
    # Fallback: standard pickle (for .pkl files without compression).
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_db(arr: np.ndarray) -> np.ndarray:
    eps = 1e-7
    x = np.clip(arr, eps, None)
    return (10.0 * np.log10(x)).astype(np.float32)
