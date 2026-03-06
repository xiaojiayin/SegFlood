"""S1S2WaterDataModule: Lightning DataModule wrapping S1S2WaterDataset."""

from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets.s1s2_water import S1S2WaterDataset


class S1S2WaterDataModule(LightningDataModule):
    """
    Lightning DataModule for S1S2-Water.

    Args:
        root:             Dataset root directory.
        modal_type:       ``"dual"`` | ``"optical"`` | ``"sar"``.
        batch_size:       Training/eval batch size.
        num_workers:      DataLoader worker count.
        pin_memory:       Whether to pin memory in DataLoaders.
        optical_channels: Base number of S2 optical channels (default 4 = BGRI).
        sar_channels:     Number of S1 SAR channels (default 2 = VV/VH).
        add_dem:          Append DEM as an extra optical channel.
        add_slope:        Append slope (derived from DEM) as an extra optical channel.
        output_image_key: When ``True``, emit a single ``"image"`` key with
                          optical+SAR concatenated (early-fusion mode).
    """

    def __init__(
        self,
        root: str = "data/S1S2Water",
        modal_type: str = "dual",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        optical_channels: int = 4,
        sar_channels: int = 2,
        add_dem: bool = False,
        add_slope: bool = False,
        output_image_key: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.root = root
        self.modal_type = modal_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.add_dem = add_dem
        self.add_slope = add_slope
        self.output_image_key = output_image_key

        # Effective channel counts exposed to Hydra interpolation.
        self.optical_channels = optical_channels + (1 if add_dem else 0) + (1 if add_slope else 0)
        self.sar_channels = sar_channels

        self.train_dataset: Optional[S1S2WaterDataset] = None
        self.val_dataset: Optional[S1S2WaterDataset] = None
        self.test_dataset: Optional[S1S2WaterDataset] = None

    def _make_dataset(self, split: str) -> S1S2WaterDataset:
        return S1S2WaterDataset(
            root=self.root,
            split=split,
            modal_type=self.modal_type,
            optical_channels=4,  # base S2 channels (DEM/slope appended internally)
            sar_channels=self.sar_channels,
            add_dem=self.add_dem,
            add_slope=self.add_slope,
            output_image_key=self.output_image_key,
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
        return DataLoader(
            self.test_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
