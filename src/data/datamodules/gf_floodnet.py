"""GFFloodNetDataModule: Lightning DataModule wrapping GFFloodNetDataset."""

from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets.gf_floodnet import GFFloodNetDataset


class GFFloodNetDataModule(LightningDataModule):
    """
    Lightning DataModule for GF-FloodNet.

    Args:
        root:             Dataset root directory.
        modal_type:       ``"dual"`` | ``"optical"`` | ``"sar"``.
        batch_size:       Training/eval batch size.
        num_workers:      DataLoader worker count.
        pin_memory:       Whether to pin memory in DataLoaders.
        predict_use_full: When ``True``, the predict split uses the *full*
                          (train+val+test) dataset (for generating tile mosaics).
        optical_channels: Number of optical input channels (GF-2 RGB = 3).
        sar_channels:     Number of SAR input channels (GF-3 VV = 1).
    """

    # Exposed to Hydra/experiment configs via ``${data.optical_channels}`` etc.
    optical_channels: int = 3
    sar_channels: int = 1

    def __init__(
        self,
        root: str = "data/GFFloodNet",
        modal_type: str = "dual",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        predict_use_full: bool = False,
        optical_channels: int = 3,
        sar_channels: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.root = root
        self.modal_type = modal_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.predict_use_full = predict_use_full
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels

        self.train_dataset: Optional[GFFloodNetDataset] = None
        self.val_dataset: Optional[GFFloodNetDataset] = None
        self.test_dataset: Optional[GFFloodNetDataset] = None
        self.predict_dataset: Optional[GFFloodNetDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        shared_kwargs = dict(
            root=self.root,
            modal_type=self.modal_type,
            optical_channels=self.optical_channels,
            sar_channels=self.sar_channels,
        )
        if stage in (None, "fit"):
            self.train_dataset = GFFloodNetDataset(split="train", **shared_kwargs)
            self.val_dataset = GFFloodNetDataset(split="val", **shared_kwargs)
        if stage in (None, "test"):
            self.test_dataset = GFFloodNetDataset(split="test", **shared_kwargs)
        if stage in (None, "predict"):
            if self.predict_use_full:
                # Concatenate all splits for full-dataset mosaic prediction.
                from torch.utils.data import ConcatDataset
                splits = [
                    GFFloodNetDataset(split=s, **shared_kwargs)
                    for s in ("train", "val", "test")
                ]
                self.predict_dataset = ConcatDataset(splits)  # type: ignore[assignment]
                # Expose a flattened .files attribute for the PredictWriter.
                all_files = []
                for ds in splits:
                    all_files.extend(ds.files)
                self.predict_dataset.files = all_files  # type: ignore[attr-defined]
            else:
                if self.test_dataset is None:
                    self.test_dataset = GFFloodNetDataset(split="test", **shared_kwargs)
                self.predict_dataset = self.test_dataset

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
            self.predict_dataset,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
