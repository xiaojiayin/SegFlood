"""GF-FloodNet dataset: GF-3 SAR (VV) + GF-2 optical patch-based flood segmentation."""

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class GFFloodNetDataset(Dataset):
    """
    Patch-based dataset for GF-FloodNet.

    Directory layout expected under ``root``::

        root/
          images/
            <split>/          # train | val | test
              <name>_opt.tif  # GF-2 optical (3-band RGB)
              <name>_sar.tif  # GF-3 SAR (1-band VV)
          annotations/
            <split>/
              <name>.tif      # binary mask: 0=land, 1=water, -1=ignore

    Each sample returns a dict:

    * ``image_optical`` – ``float32`` tensor ``(C_opt, H, W)`` (z-score normalised)
    * ``image_sar``     – ``float32`` tensor ``(C_sar, H, W)`` (z-score normalised)
    * ``mask``          – ``int64`` tensor ``(H, W)``
    * ``meta``          – dict with ``sample_name`` and ``event_id``

    The ``files`` attribute (list of optical .tif paths) is used by
    :class:`~src.infer.gffloodnet.GFFloodNetPredictWriter` for GeoTIFF geo-referencing.
    """

    # Default per-channel statistics derived from the GF-FloodNet training split.
    OPTICAL_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)   # ImageNet-style placeholder
    OPTICAL_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)
    SAR_MEAN: Tuple[float, ...] = (-11.0,)                    # approximate VV dB mean
    SAR_STD: Tuple[float, ...] = (5.0,)

    def __init__(
        self,
        root: str,
        split: str = "train",
        modal_type: str = "dual",
        optical_channels: int = 3,
        sar_channels: int = 1,
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

        # Collect paired sample paths: optical + SAR + annotation.
        self.files: List[str] = []          # optical tile paths (for geo-referencing)
        self._sar_files: List[str] = []
        self._ann_files: List[str] = []
        self.sample_ids: List[str] = []     # sample name without extension

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
                    self.files.append(opt_path)
                    self._sar_files.append(sar_path if os.path.isfile(sar_path) else "")
                    self._ann_files.append(ann_path)
                    self.sample_ids.append(stem)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            import rasterio  # type: ignore
        except ImportError:
            raise ImportError("rasterio is required. Install it with: pip install rasterio")

        sample: Dict = {}

        # --- Optical ---
        if self.modal_type in ("dual", "optical"):
            with rasterio.open(self.files[idx]) as src:
                arr = src.read(
                    list(range(1, min(src.count, self.optical_channels) + 1))
                ).astype(np.float32)
            arr = self._normalise(arr, self.OPTICAL_MEAN, self.OPTICAL_STD)
            sample["image_optical"] = torch.from_numpy(arr)

        # --- SAR ---
        if self.modal_type in ("dual", "sar"):
            sar_path = self._sar_files[idx]
            if sar_path and os.path.isfile(sar_path):
                with rasterio.open(sar_path) as src:
                    arr = src.read(
                        list(range(1, min(src.count, self.sar_channels) + 1))
                    ).astype(np.float32)
            else:
                # Fallback: zero-filled tensor.
                h, w = self._get_spatial_size(idx)
                arr = np.zeros((self.sar_channels, h, w), dtype=np.float32)
            arr = self._normalise(arr, self.SAR_MEAN, self.SAR_STD)
            sample["image_sar"] = torch.from_numpy(arr)

        # Single-modal convenience alias
        if self.modal_type == "optical" and "image_optical" in sample:
            sample["image"] = sample.pop("image_optical")
        elif self.modal_type == "sar" and "image_sar" in sample:
            sample["image"] = sample.pop("image_sar")

        # --- Mask ---
        with rasterio.open(self._ann_files[idx]) as src:
            mask = src.read(1).astype(np.int64)
        sample["mask"] = torch.from_numpy(mask)

        # --- Meta ---
        stem = self.sample_ids[idx]
        parts = stem.split("_")
        event_id = "_".join(parts[:2]) if len(parts) >= 2 else stem
        sample["meta"] = {"sample_name": stem, "event_id": event_id}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_spatial_size(self, idx: int) -> Tuple[int, int]:
        try:
            import rasterio  # type: ignore
            with rasterio.open(self.files[idx]) as src:
                return src.height, src.width
        except Exception:
            return 256, 256

    @staticmethod
    def _normalise(
        arr: np.ndarray,
        mean: Tuple[float, ...],
        std: Tuple[float, ...],
    ) -> np.ndarray:
        c = arr.shape[0]
        for i in range(c):
            m = mean[i] if i < len(mean) else mean[-1]
            s = std[i] if i < len(std) else std[-1]
            arr[i] = (arr[i] - m) / max(s, 1e-6)
        return arr
