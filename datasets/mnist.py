"""MNIST dataset helpers aligned with the original EADMM framework."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_IMAGE_SIZE = 32
DEFAULT_VAL_SIZE = 5_000


@dataclass(frozen=True)
class MNISTSplits:
    """Container for train, validation, and test datasets."""

    train: Dataset[tuple[torch.Tensor, int]]
    val: Dataset[tuple[torch.Tensor, int]]
    test: Dataset[tuple[torch.Tensor, int]]


def build_mnist_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    normalize: bool = False,
) -> transforms.Compose:
    """Create the preprocessing pipeline for MNIST."""
    if image_size <= 0:
        raise ValueError(f"image_size must be positive, got {image_size}.")

    transform_steps: list[transforms.Compose | transforms.Resize | transforms.ToTensor] = []
    if image_size != 28:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps.append(transforms.ToTensor())
    if normalize:
        transform_steps.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
    return transforms.Compose(transform_steps)


def denormalize_mnist(images: torch.Tensor) -> torch.Tensor:
    """Map normalized MNIST tensors from ``[-1, 1]`` back to ``[0, 1]`` when needed."""
    if images.numel() == 0:
        return images
    if images.min().item() < 0.0:
        return ((images + 1.0) / 2.0).clamp(0.0, 1.0)
    return images.clamp(0.0, 1.0)


def to_display_mnist(images: torch.Tensor) -> torch.Tensor:
    """Prepare MNIST-like tensors for visualization in ``[0, 1]``."""
    if images.numel() == 0:
        return images
    if images.min().item() >= 0.0 and images.max().item() <= 1.0:
        return images.clamp(0.0, 1.0)
    if images.min().item() >= -1.0 and images.max().item() <= 1.0:
        return denormalize_mnist(images)
    return images.clamp(0.0, 1.0)


def _split_training_dataset(
    dataset: Dataset[tuple[torch.Tensor, int]],
    val_size: int,
    seed: int,
) -> tuple[Subset[tuple[torch.Tensor, int]], Subset[tuple[torch.Tensor, int]]]:
    if val_size <= 0:
        raise ValueError(f"val_size must be positive, got {val_size}.")

    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError(
            f"val_size={val_size} is too large for dataset of length {len(dataset)}."
        )

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, val_size],
        generator=generator,
    )
    return train_subset, val_subset


def get_mnist_splits(
    root: str | Path = DEFAULT_DATA_ROOT,
    image_size: int = DEFAULT_IMAGE_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = 42,
    normalize: bool = False,
    download: bool = True,
) -> MNISTSplits:
    """Load MNIST with a reproducible train/val split and a held-out test split."""
    root = Path(root)
    transform = build_mnist_transform(image_size=image_size, normalize=normalize)

    full_train = datasets.MNIST(
        root=str(root),
        train=True,
        transform=transform,
        download=download,
    )
    test = datasets.MNIST(
        root=str(root),
        train=False,
        transform=transform,
        download=download,
    )
    train, val = _split_training_dataset(full_train, val_size=val_size, seed=seed)
    return MNISTSplits(train=train, val=val, test=test)


def get_mnist_dataloaders(
    root: str | Path = DEFAULT_DATA_ROOT,
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = 64,
    val_size: int = DEFAULT_VAL_SIZE,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
    normalize: bool = False,
    download: bool = True,
) -> Dict[str, DataLoader[tuple[torch.Tensor, int]]]:
    """Build train/val/test dataloaders for MNIST."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}.")

    splits = get_mnist_splits(
        root=root,
        image_size=image_size,
        val_size=val_size,
        seed=seed,
        normalize=normalize,
        download=download,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    return {
        "train": DataLoader(splits.train, shuffle=True, **loader_kwargs),
        "val": DataLoader(splits.val, shuffle=False, **loader_kwargs),
        "test": DataLoader(splits.test, shuffle=False, **loader_kwargs),
    }
