"""KuroSiwoDataModule: Lightning DataModule wrapping KuroSiwoDataset."""

from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datasets.kurosiwo import KuroSiwoDataset


class KuroSiwoDataModule(LightningDataModule):
    """
    Lightning DataModule for Kuro Siwo (SAR-only water mapping).

    Args:
        root:           Dataset root directory (contains ``pickle/`` sub-dir and tiles).
        batch_size:     Training/eval batch size.
        num_workers:    DataLoader worker count.
        pin_memory:     Whether to pin memory in DataLoaders.
        sar_channels:   SAR input channels (VV/VH = 2; +DEM = 3 when ``use_dem=True``).
        use_dem:        Append DEM as an extra input channel.
        zscore:         Apply global z-score normalisation (recommended).
        pickle_name:    Filename of the compressed-pickle index.
    """

    # SAR-only dataset: optical_channels = 0
    optical_channels: int = 0
    sar_channels: int = 2

    def __init__(
        self,
        root: str = "data/KuroSiwoGRD",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        sar_channels: int = 2,
        use_dem: bool = False,
        zscore: bool = True,
        pickle_name: str = "KuroV2_grid_dict.gz",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_dem = use_dem
        self.zscore = zscore
        self.pickle_name = pickle_name
        # Effective SAR channel count (includes DEM channel when enabled).
        self.sar_channels = sar_channels + (1 if use_dem else 0)

        self.train_dataset: Optional[KuroSiwoDataset] = None
        self.val_dataset: Optional[KuroSiwoDataset] = None
        self.test_dataset: Optional[KuroSiwoDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        shared_kwargs = dict(
            root=self.root,
            sar_channels=2,  # base VV/VH (DEM is handled by the dataset itself)
            use_dem=self.use_dem,
            zscore=self.zscore,
            pickle_name=self.pickle_name,
        )
        if stage in (None, "fit"):
            self.train_dataset = KuroSiwoDataset(split="train", **shared_kwargs)
            self.val_dataset = KuroSiwoDataset(split="val", **shared_kwargs)
        if stage in (None, "test", "predict"):
            self.test_dataset = KuroSiwoDataset(split="test", **shared_kwargs)

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
