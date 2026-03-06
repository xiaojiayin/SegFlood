"""CAUFloodDataModule: Lightning DataModule wrapping CAUFloodDataset."""

from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets.cau_flood import CAUFloodDataset


class CAUFloodDataModule(LightningDataModule):
    """
    Lightning DataModule for CAU-Flood.

    Args:
        root:             Dataset root directory.
        modal_type:       ``"dual"`` | ``"optical"`` | ``"sar"``.
        batch_size:       Training/eval batch size.
        num_workers:      DataLoader worker count.
        pin_memory:       Whether to pin memory in DataLoaders.
        optical_channels: Number of optical input channels (S2 BGRI = 4).
        sar_channels:     Number of SAR input channels (S1 VV/VH = 2).
    """

    optical_channels: int = 4
    sar_channels: int = 2

    def __init__(
        self,
        root: str = "data/CAUFlood",
        modal_type: str = "dual",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        optical_channels: int = 4,
        sar_channels: int = 2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.root = root
        self.modal_type = modal_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels

        self.train_dataset: Optional[CAUFloodDataset] = None
        self.val_dataset: Optional[CAUFloodDataset] = None
        self.test_dataset: Optional[CAUFloodDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        shared_kwargs = dict(
            root=self.root,
            modal_type=self.modal_type,
            optical_channels=self.optical_channels,
            sar_channels=self.sar_channels,
        )
        if stage in (None, "fit"):
            self.train_dataset = CAUFloodDataset(split="train", **shared_kwargs)
            self.val_dataset = CAUFloodDataset(split="val", **shared_kwargs)
        if stage in (None, "test", "predict"):
            self.test_dataset = CAUFloodDataset(split="test", **shared_kwargs)

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
