"""Encoder that maps degraded observations to the latent space of the GAN generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from models.generator import resolve_device


@dataclass(frozen=True)
class EncoderMetrics:
    """Metrics stored with encoder checkpoints."""

    train_loss: float
    val_loss: float
    val_psnr: float
    val_ssim: float


class ObservationEncoder(nn.Module):
    """Convolutional encoder for degraded 32x32 grayscale observations."""

    def __init__(self, latent_dim: int = 128, base_channels: int = 64) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.image_shape = (1, 32, 32)

        hidden_channels = max(base_channels // 2, 32)
        self.features = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * base_channels),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * base_channels * 8 * 8, latent_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 4 or observation.shape[1:] != self.image_shape:
            raise ValueError(
                f"Expected observation shape [B, 1, 32, 32], got {tuple(observation.shape)}."
            )
        return self.projection(self.features(observation))


def save_encoder_checkpoint(
    path: str | Path,
    encoder: ObservationEncoder,
    metrics: EncoderMetrics | None = None,
) -> None:
    """Save encoder weights and metadata to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "latent_dim": encoder.latent_dim,
        "base_channels": encoder.base_channels,
        "encoder_state_dict": encoder.state_dict(),
    }
    if metrics is not None:
        payload["metrics"] = asdict(metrics)

    torch.save(payload, path)


def load_encoder_checkpoint(
    path: str | Path,
    device: str | torch.device = "auto",
) -> ObservationEncoder:
    """Load an encoder checkpoint from disk."""
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=device)

    encoder = ObservationEncoder(
        latent_dim=checkpoint["latent_dim"],
        base_channels=checkpoint["base_channels"],
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.to(device)
    encoder.eval()
    return encoder
