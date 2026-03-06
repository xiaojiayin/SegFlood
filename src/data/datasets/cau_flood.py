"""CAU-Flood dataset: Sentinel-2 optical (pre-event) + Sentinel-1 SAR (post-event)."""

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CAUFloodDataset(Dataset):
    """
    Patch-based dataset for CAU-Flood flood change detection.

    Task: binary change detection — label ``1`` means *newly flooded* pixels
    (permanent water is treated as background, label ``0``).

    Directory layout expected under ``root``::

        root/
          images/
            <split>/
              <id>_opt.tif   # Sentinel-2 optical (pre-event, 4 bands BGRI)
              <id>_sar.tif   # Sentinel-1 SAR (post-event, 2 bands VV/VH)
          annotations/
            <split>/
              <id>.tif       # change mask: 0=no-change, 1=flooded, -1=ignore

    Returned dict keys match the expected Lightning batch format:

    * ``image_optical`` – ``float32 (C_opt, H, W)``
    * ``image_sar``     – ``float32 (C_sar, H, W)``
    * ``mask``          – ``int64 (H, W)``
    * ``meta``          – ``{"sample_name": str}``

    The ``sample_ids`` list is used by
    :class:`~src.infer.cauflood.CAUFloodPredictWriter` for output naming.
    """

    OPTICAL_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406, 0.45)
    OPTICAL_STD: Tuple[float, ...] = (0.229, 0.224, 0.225, 0.2)
    SAR_MEAN: Tuple[float, ...] = (-11.0, -18.0)
    SAR_STD: Tuple[float, ...] = (5.0, 5.0)

    def __init__(
        self,
        root: str,
        split: str = "train",
        modal_type: str = "dual",
        optical_channels: int = 4,
        sar_channels: int = 2,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert modal_type in ("dual", "optical", "sar"), f"Unknown modal_type: {modal_type}"

        self.root = root
        self.split = split
        self.modal_type = modal_type
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels
        self.transform = transform
        self.target_transform = target_transform

        images_dir = os.path.join(root, "images", split)
        ann_dir = os.path.join(root, "annotations", split)

        self._opt_files: List[str] = []
        self._sar_files: List[str] = []
        self._ann_files: List[str] = []
        self.sample_ids: List[str] = []

        if os.path.isdir(images_dir):
            opt_files = sorted(
                p for p in os.listdir(images_dir) if p.endswith("_opt.tif")
            )
            for opt_name in opt_files:
                stem = opt_name[: -len("_opt.tif")]
                sar_name = stem + "_sar.tif"
                ann_name = stem + ".tif"
                opt_path = os.path.join(images_dir, opt_name)
                sar_path = os.path.join(images_dir, sar_name)
                ann_path = os.path.join(ann_dir, ann_name)
                if os.path.isfile(opt_path) and os.path.isfile(ann_path):
                    self._opt_files.append(opt_path)
                    self._sar_files.append(sar_path if os.path.isfile(sar_path) else "")
                    self._ann_files.append(ann_path)
                    self.sample_ids.append(stem)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            import rasterio  # type: ignore
        except ImportError:
            raise ImportError("rasterio is required. Install it with: pip install rasterio")

        sample: Dict = {}

        if self.modal_type in ("dual", "optical"):
            with rasterio.open(self._opt_files[idx]) as src:
                arr = src.read(
                    list(range(1, min(src.count, self.optical_channels) + 1))
                ).astype(np.float32)
            arr = _normalise(arr, self.OPTICAL_MEAN, self.OPTICAL_STD)
            sample["image_optical"] = torch.from_numpy(arr)

        if self.modal_type in ("dual", "sar"):
            sar_path = self._sar_files[idx]
            if sar_path and os.path.isfile(sar_path):
                with rasterio.open(sar_path) as src:
                    arr = src.read(
                        list(range(1, min(src.count, self.sar_channels) + 1))
                    ).astype(np.float32)
            else:
                h, w = _get_spatial_size(self._opt_files[idx])
                arr = np.zeros((self.sar_channels, h, w), dtype=np.float32)
            arr = _normalise(arr, self.SAR_MEAN, self.SAR_STD)
            sample["image_sar"] = torch.from_numpy(arr)

        if self.modal_type == "optical":
            sample["image"] = sample.pop("image_optical")
        elif self.modal_type == "sar":
            sample["image"] = sample.pop("image_sar")

        with rasterio.open(self._ann_files[idx]) as src:
            mask = src.read(1).astype(np.int64)
        sample["mask"] = torch.from_numpy(mask)
        sample["meta"] = {"sample_name": self.sample_ids[idx]}

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


def _get_spatial_size(path: str) -> Tuple[int, int]:
    try:
        import rasterio  # type: ignore
        with rasterio.open(path) as src:
            return src.height, src.width
    except Exception:
        return 256, 256
