"""Reconstruction metrics used across training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from skimage.metrics import structural_similarity


@dataclass(frozen=True)
class ReconstructionMetrics:
    mse: float
    mse_std: float
    mae: float
    mae_std: float
    psnr: float
    psnr_std: float
    ssim: float
    ssim_std: float


def infer_data_range(tensor: torch.Tensor) -> float:
    """Infer the natural data range from a normalized image tensor."""
    return 2.0 if tensor.min().item() < 0.0 else 1.0


def clamp_to_range(tensor: torch.Tensor, data_range: float) -> torch.Tensor:
    """Clamp a tensor to either ``[0, 1]`` or ``[-1, 1]`` depending on range."""
    if data_range > 1.0:
        return tensor.clamp(-1.0, 1.0)
    return tensor.clamp(0.0, 1.0)


def mean_squared_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:

    mse = torch.mean((prediction - target) ** 2, dim=(1,2,3))

    return (
        mse.mean().item(),
        mse.std(unbiased=True).item(),
    )


def mean_absolute_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:

    mae = torch.mean(torch.abs(prediction - target), dim=(1,2,3))

    return (
        mae.mean().item(),
        mae.std(unbiased=True).item(),
    )


def peak_signal_to_noise_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
) -> tuple[float, float]:
    """Compute average PSNR in dB for normalized images."""
    if data_range is None:
        data_range = infer_data_range(target)
    mse_per_sample = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    mse_per_sample = torch.clamp(mse_per_sample, min=1e-12)
    psnr = 10.0 * torch.log10(
        torch.tensor(float(data_range**2), device=prediction.device) / mse_per_sample
    )
    psnr_mean = psnr.mean().item()
    psnr_std = psnr.std(unbiased=True).item()

    return psnr_mean, psnr_std


def structural_similarity_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
) -> tuple[float, float]:

    if data_range is None:
        data_range = infer_data_range(target)

    prediction_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    scores = []

    for pred_img, target_img in zip(prediction_np, target_np):
        scores.append(
            structural_similarity(
                np.squeeze(target_img),
                np.squeeze(pred_img),
                data_range=data_range,
            )
        )

    scores = np.asarray(scores)

    return (
        float(scores.mean()),
        float(scores.std(ddof=1)),
    )


def compute_reconstruction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
) -> ReconstructionMetrics:
    """Compute the main reconstruction metrics used in the project."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got {prediction.shape} and {target.shape}."
        )
    if data_range is None:
        data_range = infer_data_range(target)

    prediction = clamp_to_range(prediction, data_range=data_range)
    target = clamp_to_range(target, data_range=data_range)

    mse_mean, mse_std = mean_squared_error(prediction, target)
    mae_mean, mae_std = mean_absolute_error(prediction, target)
    psnr_mean, psnr_std = peak_signal_to_noise_ratio(prediction, target, data_range=data_range)
    ssim_mean, ssim_std = structural_similarity_index(prediction, target, data_range=data_range)

    return ReconstructionMetrics(
        mse=mse_mean,
        mse_std=mse_std,
        mae=mae_mean,
        mae_std=mae_std,
        psnr=psnr_mean,
        psnr_std=psnr_std,
        ssim=ssim_mean,
        ssim_std=ssim_std,
    )


def compute_metrics_curve(
    history: torch.Tensor,
    target: torch.Tensor,
    data_range: float | None = None,
) -> list[ReconstructionMetrics]:
    """Compute reconstruction metrics for every iteration in a trajectory."""
    if history.ndim != 5:
        raise ValueError(f"history must have shape [B, T, C, H, W], got {history.shape}.")
    if history.shape[0] != target.shape[0] or history.shape[2:] != target.shape[1:]:
        raise ValueError(
            "history and target shapes are incompatible: "
            f"{history.shape} vs {target.shape}."
        )

    return [
        compute_reconstruction_metrics(
            prediction=history[:, step],
            target=target,
            data_range=data_range,
        )
        for step in range(history.shape[1])
    ]
