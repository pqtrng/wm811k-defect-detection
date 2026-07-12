"""Test for wm811k.data -- dataset and loader contracts
The single most important contract here: WaferDataset return RAW {0,1,2} die values. Normalization (x/2.0) lives inside model.forward(). If someone "helpfully" normalizes in the dataset, inputs get divided twice and metrics silently degrade -- these tests make that impossible to miss.
"""
from __future__ import annotations

import pytest
import torch
from wm811k.data import WaferDataset, AugmentedWaferDataset, build_loaders
from wm811k.seed import set_seed


def test_dataset_return_raw_unnormalized(tiny_config):
    ds = WaferDataset(
        tiny_config.paths.processed_dir / "train.parquet", tiny_config.labels
    )
    x, y = ds[0]
    assert x.shape == (1, 64, 64)
    assert x.dtype == torch.float32
    # RAW contract: values still in {0,1,2}, NOT {0, 0.5, 1.0}
    assert set(torch.unique(x).tolist()) == {0.0, 1.0, 2.0}
    assert isinstance(y, int)
    assert 0 <= y < tiny_config.num_classes


def test_dataset_rejects_unknown_label(make_split_df, write_parquet, labels):
    df = make_split_df()
    df.loc[0, "label"] = "Banana"
    path = write_parquet(df)
    with pytest.raises(ValueError, match="Unknown label"):
        WaferDataset(path, labels)


def test_augmentation_is_pure_permutation(tiny_config):
    """rot90/flip PERMUTE pixels, never interpolate: per-value pixel counts
    (bincount over {0,1,2}) must be identical before and after. This is the
    mathematical form of the 'domain-safe augmentation' claim in the README."""
    path = tiny_config.paths.processed_dir / "train.parquet"
    plain = WaferDataset(path, tiny_config.labels)
    augmented = AugmentedWaferDataset(path, tiny_config.labels)

    set_seed(123)  # augmentation draws from torch global RNG
    for idx in range(len(plain)):
        x_plain, y_plain = plain[idx]
        x_aug, y_aug = augmented[idx]
        assert y_aug == y_plain
        assert x_aug.is_contiguous()  # documented contract for collate
        assert torch.equal(
            torch.bincount(x_aug.flatten().long(), minlength=3),
            torch.bincount(x_plain.flatten().long(), minlength=3),
        )


def test_val_test_never_augmented(tiny_config):
    """augment=True must swap ONLY the train dataset. Exact-type check on
    purpose: AugmentedWaferDataset subclasses WaferDataset, so isinstance
    would pass even if val/test were wrongly wrapped."""
    train_loader, val_loader, test_loader = build_loaders(tiny_config, augment=True)
    assert type(train_loader.dataset) is AugmentedWaferDataset
    assert type(val_loader.dataset) is WaferDataset
    assert type(test_loader.dataset) is WaferDataset


def test_loader_order_deterministic_under_seed(tiny_config):
    """set_seed -> identical WeightedRandomSampler draw order across two
    loader constructions. Honest scope: this is LOADER-ORDER determinism;
    split determinism belongs to the split function (T8)."""

    def first_epoch_labels() -> torch.Tensor:
        set_seed(tiny_config.seed)
        train_loader, _, _ = build_loaders(tiny_config, augment=False)
        return torch.cat([yb for _, yb in train_loader])

    assert torch.equal(first_epoch_labels(), first_epoch_labels())
