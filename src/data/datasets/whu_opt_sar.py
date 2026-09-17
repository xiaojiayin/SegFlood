"""WHU-OPT-SAR dataset following TorchGeo multi-modal best practices.

Based on detailed dataset analysis:
- 100 image pairs, resolution 5556×3704 pixels
- Optical: 4-channel RGBA, uint8, range [1,254] 
- SAR: Single channel grayscale, uint8, range [1,248]
- Labels: Single channel, uint8, range [0,70] with 8 classes
- Label values: {0, 10, 20, 30, 40, 50, 60, 70} → mapped to indices {0, 1, 2, 3, 4, 5, 6, 7}
- File format: .tif files with aligned naming

Expected directory structure:
root/
├── optical/   # 4-channel RGBA optical images (5556×3704)
├── sar/       # Single-channel SAR images (5556×3704) 
├── lbl/       # Single-channel label images (5556×3704)
├── predict/   # Prediction examples (optional)
└── intro/     # Dataset documentation (optional)
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor

try:
    from torchgeo.datasets import NonGeoDataset
    BaseDataset = NonGeoDataset
except Exception:
    from torch.utils.data import Dataset
    BaseDataset = Dataset


class WHUOptSAR(BaseDataset):
    """WHU-OPT-SAR dataset for multi-modal remote sensing segmentation.
    
    Based on detailed analysis of the actual dataset:
    - 100 samples, each 5556×3704 pixels
    - Optical: 4-channel RGBA (R,G,B,A), uint8 range [1,254]
    - SAR: Single-channel grayscale, uint8 range [1,248]  
    - Labels: 8 classes with values {0,10,20,30,40,50,60,70}
    
    Class distribution from analysis:
    - Class 0 (background): 0.0% 
    - Class 10 (farmland): 40.4%
    - Class 20 (city): 3.8%
    - Class 30 (village): 5.7% 
    - Class 40 (water): 6.4%
    - Class 50 (forest): 40.7%
    - Class 60 (road): 1.0%
    - Class 70 (others): 1.9%

    Returns:
      - image: Tensor[5, H, W] with channels [R, G, B, A, SAR] in float32 [0,255]
      - mask: Tensor[H, W] with contiguous class indices {0,1,2,3,4,5,6,7}
    """

    # Class names based on typical remote sensing land cover categories
    classes: List[str] = [
        "background",    # 0 -> 0
        "farmland",      # 10 -> 1  
        "city",          # 20 -> 2
        "village",       # 30 -> 3
        "water",         # 40 -> 4
        "forest",        # 50 -> 5
        "road",          # 60 -> 6
        "others",        # 70 -> 7
    ]

    # Mapping from dataset label codes to contiguous indices
    code_to_index: Dict[int, int] = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 7}
    index_to_code: Dict[int, int] = {v: k for k, v in code_to_index.items()}

    def __init__(
        self,
        root: str,
        split: str | None = None,
        transforms: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
        indices: Sequence[int] | None = None,
        optical_dirname: str = "optical",
        sar_dirname: str = "sar",
        label_dirname: str = "lbl",
        exts: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
        # --- cropping ---
        crop_size: Tuple[int, int] | None = None,  # (h, w)
        crop_type: str | None = None,  # "random" | "center"
        # --- training sampling ---
        samples_per_image: int = 1,
        # --- sliding window tiling (eval/predict) ---
        tile_size: Tuple[int, int] | None = None,
        tile_stride: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()  # type: ignore[misc]

        self.root = root
        self.transforms = transforms
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.samples_per_image = max(1, int(samples_per_image))
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.opt_dir = os.path.join(root, optical_dirname)
        self.sar_dir = os.path.join(root, sar_dirname)
        self.lbl_dir = os.path.join(root, label_dirname)
        self.exts = exts

        # Validate directories
        for p in [self.opt_dir, self.sar_dir, self.lbl_dir]:
            if not os.path.isdir(p):
                raise FileNotFoundError(f"Directory not found: {p}")

        # Enumerate base filenames from label directory (authoritative)
        base_names: List[str] = []
        for fname in sorted(os.listdir(self.lbl_dir)):
            lower = fname.lower()
            if any(lower.endswith(e) for e in self.exts):
                base_names.append(os.path.splitext(fname)[0])

        if not base_names:
            raise FileNotFoundError(f"No label files found in {self.lbl_dir}")

        # Filter to only those that have corresponding optical and sar
        valid_names: List[str] = []
        for base in base_names:
            opt_path = self._first_existing(self.opt_dir, base)
            sar_path = self._first_existing(self.sar_dir, base)
            lbl_path = self._first_existing(self.lbl_dir, base)
            if opt_path and sar_path and lbl_path:
                valid_names.append(base)

        if not valid_names:
            raise FileNotFoundError("No aligned optical/sar/label triplets found.")

        # Apply subset indices if any
        if indices is not None:
            self.sample_ids = [valid_names[i] for i in indices if 0 <= i < len(valid_names)]
        else:
            self.sample_ids = valid_names

        # Precompute tile coordinates if tiling is enabled
        self._tiles: List[Tuple[int, int, int, int, int]] = []  # (img_idx, top, left, h, w)
        if self.tile_size is not None and self.tile_stride is not None:
            th, tw = int(self.tile_size[0]), int(self.tile_size[1])
            sh, sw = int(self.tile_stride[0]), int(self.tile_stride[1])
            if th <= 0 or tw <= 0 or sh <= 0 or sw <= 0:
                raise ValueError("tile_size and tile_stride must be positive")
            for img_idx, base in enumerate(self.sample_ids):
                # read label to get image shape
                lbl_path = self._first_existing(self.lbl_dir, base)
                assert lbl_path is not None
                with Image.open(lbl_path) as _img:
                    H, W = _img.size[1], _img.size[0]
                # generate positions with coverage of right/bottom edges
                tops: List[int] = list(range(0, max(H - th, 0) + 1, sh))
                lefts: List[int] = list(range(0, max(W - tw, 0) + 1, sw))
                if len(tops) == 0 or tops[-1] != H - th:
                    tops.append(max(H - th, 0))
                if len(lefts) == 0 or lefts[-1] != W - tw:
                    lefts.append(max(W - tw, 0))
                for top in tops:
                    for left in lefts:
                        self._tiles.append((img_idx, int(top), int(left), th, tw))

    # --- helpers ---
    def _first_existing(self, folder: str, base: str) -> str | None:
        for e in self.exts:
            p = os.path.join(folder, base + e)
            if os.path.isfile(p):
                return p
        return None

    def __len__(self) -> int:
        if self.tile_size is not None and self.tile_stride is not None:
            return len(self._tiles)
        if self.samples_per_image > 1:
            return len(self.sample_ids) * self.samples_per_image
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # determine which image and region to use
        if self.tile_size is not None and self.tile_stride is not None:
            img_idx, top, left, ch, cw = self._tiles[index]
            base = self.sample_ids[img_idx]
            crop_region = (int(top), int(left), int(ch), int(cw))
        else:
            base_idx = index if self.samples_per_image == 1 else index // self.samples_per_image
            base = self.sample_ids[base_idx]
            crop_region = None

        opt_path = self._first_existing(self.opt_dir, base)
        sar_path = self._first_existing(self.sar_dir, base)
        lbl_path = self._first_existing(self.lbl_dir, base)
        assert opt_path and sar_path and lbl_path  # guarded in __init__

        # Load optical image (4-channel RGBA based on analysis)
        opt_img = Image.open(opt_path)
        opt_arr = np.array(opt_img)
        
        # Verify optical image format matches analysis: (H, W, 4) RGBA
        if opt_arr.ndim == 3 and opt_arr.shape[-1] == 4:
            # Expected format: RGBA channels
            pass
        elif opt_arr.ndim == 3 and opt_arr.shape[-1] == 3:
            # RGB only, add alpha channel (fallback)
            alpha = np.full(opt_arr.shape[:2] + (1,), 255, dtype=opt_arr.dtype)
            opt_arr = np.concatenate([opt_arr, alpha], axis=-1)
        elif opt_arr.ndim == 2:
            # Grayscale, replicate to RGBA
            opt_arr = np.stack([opt_arr] * 4, axis=-1)
        else:
            raise ValueError(f"Unexpected optical image shape: {opt_arr.shape} for {opt_path}")

        # Load SAR image (single channel based on analysis)
        sar_img = Image.open(sar_path)
        sar_arr = np.array(sar_img)
        
        # Ensure SAR is single channel
        if sar_arr.ndim == 3:
            sar_arr = sar_arr[:, :, 0]  # Take first channel if multi-channel
        elif sar_arr.ndim != 2:
            raise ValueError(f"Unexpected SAR image shape: {sar_arr.shape} for {sar_path}")

        # Load label image (single channel with values {0,10,20,30,40,50,60,70})
        lbl_img = Image.open(lbl_path)
        lbl_arr = np.array(lbl_img)
        
        # Ensure label is single channel
        if lbl_arr.ndim == 3:
            lbl_arr = lbl_arr[:, :, 0]  # Take first channel if multi-channel
        elif lbl_arr.ndim != 2:
            raise ValueError(f"Unexpected label image shape: {lbl_arr.shape} for {lbl_path}")

        # Convert to float32 for image data (keep original [0,255] scale)
        opt_arr = opt_arr.astype(np.float32)
        sar_arr = sar_arr.astype(np.float32)

        # Combine optical (RGBA) + SAR → [H, W, 5]
        combined = np.concatenate([opt_arr, sar_arr[:, :, np.newaxis]], axis=-1)

        # --- cropping/tiling on CPU ---
        if crop_region is not None:
            top, left, ch, cw = crop_region
            bottom, right = top + ch, left + cw
            combined = combined[top:bottom, left:right, :]
            lbl_arr = lbl_arr[top:bottom, left:right]
        elif self.crop_size is not None:
            ch, cw = int(self.crop_size[0]), int(self.crop_size[1])
            H, W = combined.shape[0], combined.shape[1]
            if ch > H or cw > W:
                raise ValueError(
                    f"Requested crop_size {(ch, cw)} exceeds image size {(H, W)} for {base}"
                )
            if (self.crop_type or "random").lower() == "random":
                top = np.random.randint(0, H - ch + 1)
                left = np.random.randint(0, W - cw + 1)
            else:
                top = (H - ch) // 2
                left = (W - cw) // 2
            bottom = top + ch
            right = left + cw
            combined = combined[top:bottom, left:right, :]
            lbl_arr = lbl_arr[top:bottom, left:right]

        # to tensor [5, H, W]
        image = torch.from_numpy(combined.transpose(2, 0, 1)).float()

        # Map label codes {0,10,20,30,40,50,60,70} to contiguous indices {0,1,2,3,4,5,6,7}
        mask_np = np.zeros_like(lbl_arr, dtype=np.int64)
        for code, idx in self.code_to_index.items():
            mask_np[lbl_arr == code] = idx
        mask = torch.from_numpy(mask_np).long()

        sample: Dict[str, Any] = {"image": image, "mask": mask, "id": base}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @property
    def num_classes(self) -> int:
        return len(self.classes)
