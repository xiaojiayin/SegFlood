"""S1S2-Water dataset: co-registered Sentinel-1 SAR + Sentinel-2 optical (+optional DEM/slope)."""

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class S1S2WaterDataset(Dataset):
    """
    Patch-based dataset for S1S2-Water global permanent water mapping.

    Each sample is a co-registered pair of S2 optical (4 bands: BGRI) and S1 SAR
    (2 bands: VV/VH) patches with an optional DEM and slope channel appended
    to the optical tensor.

    Directory layout under ``root``::

        root/
          train/
            s2/   (Sentinel-2, GeoTIFF, >=4 bands)
            s1/   (Sentinel-1, GeoTIFF, 2 bands VV/VH)
            dem/  (DEM, optional, GeoTIFF, 1 band)
            mask/ (water mask, GeoTIFF, 1 band)
          val/
            ...
          test/
            ...

    Mask values: ``0=non-water``, ``1=water``, ``-1=ignore``.

    The ``samples`` attribute (list of ``{"img": path, ...}`` dicts) is used
    by :class:`~src.infer.s1s2_water.S1S2WaterPredictWriter` for file naming.
    """

    # S2 BGRI: approximate per-channel z-score (top-of-atmosphere reflectance × 10000)
    S2_MEAN: Tuple[float, ...] = (1353.0, 1116.8, 1041.5, 2303.2)
    S2_STD: Tuple[float, ...] = (65.22, 154.08, 187.31, 533.59)
    # S1 dB-scaled
    S1_MEAN: Tuple[float, ...] = (-11.0, -18.0)
    S1_STD: Tuple[float, ...] = (5.0, 5.0)
    # DEM z-score
    DEM_MEAN: float = 93.4313
    DEM_STD: float = 1410.8382
    # Slope
    SLOPE_MEAN: float = 2.5
    SLOPE_STD: float = 5.0

    def __init__(
        self,
        root: str,
        split: str = "train",
        modal_type: str = "dual",
        optical_channels: int = 4,
        sar_channels: int = 2,
        add_dem: bool = False,
        add_slope: bool = False,
        output_image_key: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert modal_type in ("dual", "optical", "sar"), f"Unknown modal_type: {modal_type}"

        self.root = root
        self.split = split
        self.modal_type = modal_type
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.add_dem = add_dem
        self.add_slope = add_slope
        self.output_image_key = output_image_key
        self.transform = transform

        # Effective number of optical input channels (base + optional DEM/slope).
        self._effective_optical_channels = optical_channels + (1 if add_dem else 0) + (1 if add_slope else 0)

        # Discover samples
        s2_dir = os.path.join(root, split, "s2")
        s1_dir = os.path.join(root, split, "s1")
        dem_dir = os.path.join(root, split, "dem")
        msk_dir = os.path.join(root, split, "mask")

        self.samples: List[Dict[str, str]] = []

        if os.path.isdir(s2_dir):
            for fname in sorted(os.listdir(s2_dir)):
                if not fname.endswith(".tif"):
                    continue
                s2_path = os.path.join(s2_dir, fname)
                s1_path = os.path.join(s1_dir, fname)
                dem_path = os.path.join(dem_dir, fname)
                msk_path = os.path.join(msk_dir, fname)
                entry: Dict[str, str] = {"img": s2_path, "msk": msk_path}
                if os.path.isfile(s1_path):
                    entry["sar"] = s1_path
                if self.add_dem and os.path.isfile(dem_path):
                    entry["dem"] = dem_path
                self.samples.append(entry)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            import rasterio  # type: ignore
        except ImportError:
            raise ImportError("rasterio is required. Install it with: pip install rasterio")

        entry = self.samples[idx]
        sample: Dict = {}

        # --- Optical (S2) ---
        if self.modal_type in ("dual", "optical"):
            with rasterio.open(entry["img"]) as src:
                arr = src.read(
                    list(range(1, min(src.count, self.optical_channels) + 1))
                ).astype(np.float32)
            arr = _normalise(arr, self.S2_MEAN, self.S2_STD)
            extra: List[np.ndarray] = []
            if self.add_dem and "dem" in entry and os.path.isfile(entry["dem"]):
                with rasterio.open(entry["dem"]) as src:
                    dem = src.read(1).astype(np.float32)
                dem = (dem - self.DEM_MEAN) / max(self.DEM_STD, 1e-6)
                extra.append(dem[None])
            if self.add_slope and "dem" in entry and os.path.isfile(entry["dem"]):
                # Approximate slope from DEM via gradient magnitude.
                with rasterio.open(entry["dem"]) as src:
                    dem_raw = src.read(1).astype(np.float32)
                gy = np.gradient(dem_raw, axis=0)
                gx = np.gradient(dem_raw, axis=1)
                slope = np.sqrt(gy**2 + gx**2)
                slope = (slope - self.SLOPE_MEAN) / max(self.SLOPE_STD, 1e-6)
                extra.append(slope[None])
            if extra:
                arr = np.concatenate([arr] + extra, axis=0)
            sample["image_optical"] = torch.from_numpy(arr)

        # --- SAR (S1) ---
        if self.modal_type in ("dual", "sar"):
            sar_path = entry.get("sar", "")
            if sar_path and os.path.isfile(sar_path):
                with rasterio.open(sar_path) as src:
                    sar_arr = src.read(
                        list(range(1, min(src.count, self.sar_channels) + 1))
                    ).astype(np.float32)
            else:
                h = sample["image_optical"].shape[1] if "image_optical" in sample else 256
                w = sample["image_optical"].shape[2] if "image_optical" in sample else 256
                sar_arr = np.zeros((self.sar_channels, h, w), dtype=np.float32)
            sar_arr = _normalise(sar_arr, self.S1_MEAN, self.S1_STD)
            sample["image_sar"] = torch.from_numpy(sar_arr)

        # Single-modal aliases / early-fusion image key
        if self.modal_type == "optical":
            sample["image"] = sample.pop("image_optical")
        elif self.modal_type == "sar":
            sample["image"] = sample.pop("image_sar")
        elif self.output_image_key and "image_optical" in sample and "image_sar" in sample:
            # Early-fusion: concatenate optical + SAR into a single "image" tensor.
            sample["image"] = torch.cat([sample["image_optical"], sample["image_sar"]], dim=0)

        # --- Mask ---
        msk_path = entry["msk"]
        if os.path.isfile(msk_path):
            with rasterio.open(msk_path) as src:
                mask = src.read(1).astype(np.int64)
        else:
            h = list(sample.values())[0].shape[-2] if sample else 256
            w = list(sample.values())[0].shape[-1] if sample else 256
            mask = np.full((h, w), -1, dtype=np.int64)
        sample["mask"] = torch.from_numpy(mask)

        stem = os.path.splitext(os.path.basename(entry["img"]))[0]
        sample["meta"] = {"sample_name": stem}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _normalise(arr: np.ndarray, mean: Tuple, std: Tuple) -> np.ndarray:
    c = arr.shape[0]
    for i in range(c):
        m = mean[i] if i < len(mean) else mean[-1]
        s = std[i] if i < len(std) else std[-1]
        arr[i] = (arr[i] - m) / max(s, 1e-6)
    return arr
