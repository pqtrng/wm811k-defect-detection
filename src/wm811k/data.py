"""Data loading and preprocessing for wm811k pipeline."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from wm811k.config import Config


class WaferDataset(Dataset):
    """Load one Parquet split into memory. Returns (float tensor [1,64,64], int label)."""

    def __init__(self, parquet_path: str | Path, labels: list[str]) -> None:
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()

        label2idx = {name: i for i, name in enumerate(labels)}
        wafers = np.stack(df["wafer"].values)
        if wafers.ndim != 2 or wafers.shape[1] != 64 * 64:
            raise ValueError(
                f"Expected 2D array of shape (N, 64, 64), got {wafers.shape}"
            )
        self.X = wafers.reshape(-1, 1, 64, 64).astype(np.float32)

        mapped = df["label"].map(label2idx)
        if mapped.isna().any():
            unknown = sorted(set(df["label"].unique()) - set(labels))
            raise ValueError(f"Unknown label(s) found in {parquet_path}: {unknown}")
        self.y = mapped.values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


class AugmentedWaferDataset(WaferDataset):
    """WaferDataset + on the fly geometric augmentation, for training only.
    Inherits all data loading from WaferDataset; only __getitem__ is overridden. Domain-safe by construction:
        - Only 90/180/270 rotation + flips. These PERMUTE pixels (never interpolate), so discrete {0,1,2} die values are preserved exactly.
        - Transforms are about the center, so edge-vs-center is preserved (a Center defect never drifts toward the edge -- unlike crop/translate).
    Never wrap val/test splits in this class -- evaluation must be raw data.
    """

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x, y = super().__getitem__(idx)

        k = torch.randint(low=0, high=4, size=(1,)).item()

        if k > 0:
            x = torch.rot90(input=x, k=k, dims=[1, 2])

        if torch.rand(1).item() < 0.5:
            x = torch.flip(input=x, dims=(-1,))  # horizontal flip

        if torch.rand(1).item() < 0.5:
            x = torch.flip(input=x, dims=(-2,))  # vertical flip
        # rot90/flip return non-contiguous views; make contiguous for the Dataloader's default collate and any downstream ops.
        return x.contiguous(), y


def build_loaders(
    config: Config, augment: bool = False, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders.
    Train uses a WeightedRandomSampler (inverse class frequency) so `shuffle` is never set on it -- the sampler controls sample order. Val/test see the natural class distribution, unshuffled.
    augment=True swaps the train dataset for AugmentedWaferDataset, val/test are always the plain WaferDataset regardless of this flag.
    """
    processed_dir = config.paths.processed_dir
    labels = config.labels
    batch_size = config.training.batch_size
    train_cls = AugmentedWaferDataset if augment else WaferDataset
    train_ds = train_cls(processed_dir / "train.parquet", labels)
    val_ds = WaferDataset(processed_dir / "val.parquet", labels)
    test_ds = WaferDataset(processed_dir / "test.parquet", labels)

    class_counts = np.bincount(train_ds.y, minlength=config.num_classes)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_ds.y]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader
