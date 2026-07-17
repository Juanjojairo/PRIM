"""Visualization helpers for labeled qualitative grids."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import torch
from skimage.metrics import structural_similarity

from datasets.mnist import to_display_mnist

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")


def save_labeled_mnist_grid(
    rows: list[torch.Tensor],
    row_labels: list[str],
    save_path: str | Path,
) -> None:
    """Save a labeled image grid where each row corresponds to a semantic stage."""
    if len(rows) != len(row_labels):
        raise ValueError(f"Expected one label per row, got {len(rows)} rows and {len(row_labels)} labels.")
    if not rows:
        raise ValueError("Expected at least one row to visualize.")

    num_images = rows[0].shape[0]
    if num_images <= 0:
        raise ValueError("Expected at least one image per row.")
    for row in rows:
        if row.shape[0] != num_images:
            raise ValueError("All rows must have the same number of images.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=num_images,
        figsize=(max(1.6 * num_images, 4.0), max(1.6 * len(rows), 4.0)),
    )

    if len(rows) == 1:
        axes = [axes]
    for row_index, (row, label) in enumerate(zip(rows, row_labels)):
        row_axes = axes[row_index]
        if num_images == 1:
            row_axes = [row_axes]
        for col_index, axis in enumerate(row_axes):
            image = row[col_index].detach().cpu().squeeze().numpy()
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if col_index == 0:
                axis.set_ylabel(label, rotation=90, fontsize=11, labelpad=18)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_sample_psnr_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> list[tuple[float, float]]:
    """Compute PSNR and SSIM for each sample in a batch."""
    prediction = prediction.detach().cpu().clamp(0.0, 1.0)
    target = target.detach().cpu().clamp(0.0, 1.0)
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}.")

    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    psnr_values = 10.0 * torch.log10(torch.tensor(data_range**2) / mse)
    rows: list[tuple[float, float]] = []
    for index in range(prediction.shape[0]):
        pred_2d = np.squeeze(prediction[index].numpy())
        target_2d = np.squeeze(target[index].numpy())
        ssim = structural_similarity(target_2d, pred_2d, data_range=data_range)
        rows.append((float(psnr_values[index].item()), float(ssim)))
    return rows


def save_reconstruction_triplets_svg(
    save_path: str | Path,
    backprojection: torch.Tensor,
    estimate: torch.Tensor,
    target: torch.Tensor,
    *,
    num_images: int = 8,
    title: str | None = None,
) -> None:
    """Save editable SVG triplets: H^T y, reconstruction estimate, and ground truth."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_images = min(num_images, backprojection.shape[0], estimate.shape[0], target.shape[0])
    if num_images <= 0:
        raise ValueError("Expected at least one sample for SVG visualization.")

    backprojection = to_display_mnist(backprojection[:num_images].detach().cpu())
    estimate = to_display_mnist(estimate[:num_images].detach().cpu())
    target = to_display_mnist(target[:num_images].detach().cpu())
    sample_metrics = compute_sample_psnr_ssim(estimate, target)

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=3,
        figsize=(5.1, max(1.45 * num_images, 2.6)),
        squeeze=False,
    )
    column_titles = [r"$H^T y$", "Estimate", "GT"]
    columns = [backprojection, estimate, target]

    for row_index in range(num_images):
        psnr, ssim = sample_metrics[row_index]
        for col_index, (column_title, images) in enumerate(zip(column_titles, columns)):
            axis = axes[row_index][col_index]
            axis.imshow(images[row_index].squeeze().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(column_title, fontsize=9)
            if col_index == 0:
                axis.set_ylabel(
                    f"#{row_index + 1}\nPSNR {psnr:.2f}\nSSIM {ssim:.3f}",
                    rotation=0,
                    fontsize=7,
                    ha="right",
                    va="center",
                    labelpad=32,
                )

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
