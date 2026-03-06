"""WorldFloodsv2DataModule: Lightning DataModule wrapping WorldFloodsv2Dataset."""

from typing import Dict, List, Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets.worldfloodsv2 import WorldFloodsv2Dataset, CHANNELS_CONFIGURATIONS


class WorldFloodsv2DataModule(LightningDataModule):
    """
    Lightning DataModule for WorldFloods v2.

    Args:
        root:                    Dataset root directory.
        batch_size:              Training/eval batch size.
        num_workers:             DataLoader worker count.
        pin_memory:              Whether to pin memory in DataLoaders.
        channels:                Named channel config (e.g. ``"bgri"``) or a
                                 list of 0-based Sentinel-2 band indices.
        target_type:             ``"binary"`` for flood/non-flood segmentation.
        water_values:            Raw mask values that map to class ``1`` (water).
        ignore_index:            Ignore label (default ``-1``).
        ignore_values:           Raw mask values that map to ``ignore_index``.
        test_use_tiles:          When ``False``, the test split loads full scenes.
        official_normalization:  Use the official WorldFloods Sentinel-2 z-score.
        normalization_mean:      Custom per-channel mean (overrides official).
        normalization_std:       Custom per-channel std (overrides official).
        add_mndwi_input:         Append MNDWI band computed from B3/B11.
        window_size:             Spatial tile size ``[H, W]`` for sliding-window.
        sliding_window:          Kwargs forwarded to the sliding-window config.
    """

    sar_channels: int = 0  # optical-only dataset

    def __init__(
        self,
        root: str = "data/WorldFloodsV2",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        channels: str = "bgri",
        target_type: str = "binary",
        water_values: Optional[List[int]] = None,
        ignore_index: int = -1,
        ignore_values: Optional[List[int]] = None,
        test_use_tiles: bool = True,
        official_normalization: bool = True,
        normalization_mean: Optional[List[float]] = None,
        normalization_std: Optional[List[float]] = None,
        add_mndwi_input: bool = False,
        window_size: Optional[List[int]] = None,
        sliding_window: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.channels = channels
        self.target_type = target_type
        self.water_values = water_values if water_values is not None else [2]
        self.ignore_index = ignore_index
        self.ignore_values = ignore_values if ignore_values is not None else [0, 3]
        self.test_use_tiles = test_use_tiles
        self.official_normalization = official_normalization
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.add_mndwi_input = add_mndwi_input
        self.window_size = window_size or [1024, 1024]
        self.sliding_window = sliding_window or {}

        # Store kwargs so that the PredictWriter can read the channel config.
        self.kwargs = {"channels": channels}

        # Derive optical_channels from the channel config.
        if isinstance(channels, str) and channels in CHANNELS_CONFIGURATIONS:
            self.optical_channels = len(CHANNELS_CONFIGURATIONS[channels]) + (1 if add_mndwi_input else 0)
        elif isinstance(channels, (list, tuple)):
            self.optical_channels = len(channels) + (1 if add_mndwi_input else 0)
        else:
            self.optical_channels = 4

        self.train_dataset: Optional[WorldFloodsv2Dataset] = None
        self.val_dataset: Optional[WorldFloodsv2Dataset] = None
        self.test_dataset: Optional[WorldFloodsv2Dataset] = None

    def _make_dataset(self, split: str) -> WorldFloodsv2Dataset:
        return WorldFloodsv2Dataset(
            root=self.root,
            split=split,
            channels=self.channels,
            target_type=self.target_type,
            water_values=self.water_values,
            ignore_index=self.ignore_index,
            ignore_values=self.ignore_values,
            official_normalization=self.official_normalization,
            normalization_mean=self.normalization_mean,
            normalization_std=self.normalization_std,
            add_mndwi_input=self.add_mndwi_input,
            test_use_tiles=(self.test_use_tiles if split == "test" else True),
            window_size=self.window_size,
            sliding_window=self.sliding_window,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = self._make_dataset("train")
            self.val_dataset = self._make_dataset("val")
        if stage in (None, "test", "predict"):
            self.test_dataset = self._make_dataset("test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> DataLoader:
        # WorldFloods full-scene inference is driven by the Writer (not the DataLoader),
        # so we return a length-1 DataLoader that simply signals the predict loop to start.
        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
