"""GAN-based MNIST generator aligned with the original EADMM framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a device string into a PyTorch device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass(frozen=True)
class GeneratorMetrics:
    """Metrics tracked while training the generator."""

    train_generator_loss: float
    train_discriminator_loss: float
    val_generator_loss: float
    val_discriminator_loss: float


class MNISTGenerator(nn.Module):
    """Convolutional generator matching the EADMM MNIST architecture."""

    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 64,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")
        if activation not in {"relu", "elu"}:
            raise ValueError(f"activation must be 'relu' or 'elu', got {activation!r}.")

        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.activation = activation
        self.image_shape = (1, 32, 32)
        self.output_range = (0.0, 1.0)
        activation_layer: type[nn.Module]
        activation_layer = nn.ELU if activation == "elu" else nn.ReLU

        self.preprocess = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 4 * base_channels),
            activation_layer(inplace=True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * base_channels, 2 * base_channels, kernel_size=4, stride=2, padding=1),
            activation_layer(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            activation_layer(inplace=True),
        )
        self.block3 = nn.ConvTranspose2d(base_channels, 1, kernel_size=4, stride=2, padding=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent vectors with shape [B, {self.latent_dim}], got {tuple(z.shape)}."
            )

        features = self.preprocess(z).view(-1, 4 * self.base_channels, 4, 4)
        features = self.block1(features)
        features = self.block2(features)
        images = self.block3(features)
        return self.output_activation(images)


class MNISTDiscriminator(nn.Module):
    """Convolutional critic used for Wasserstein training on MNIST."""

    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}.")

        self.base_channels = base_channels
        self.image_shape = (1, 32, 32)
        self.main = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 2 * base_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * base_channels, 4 * base_channels, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Linear(4 * base_channels * 4 * 4, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != self.image_shape:
            raise ValueError(
                f"Expected images with shape [B, 1, 32, 32], got {tuple(images.shape)}."
            )
        features = self.main(images)
        return self.output(features.view(images.shape[0], -1))


def save_generator_checkpoint(
    path: str | Path,
    generator: MNISTGenerator,
    discriminator: MNISTDiscriminator | None = None,
    metrics: GeneratorMetrics | None = None,
) -> None:
    """Save generator and optional discriminator weights to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "latent_dim": generator.latent_dim,
        "base_channels": generator.base_channels,
        "activation": generator.activation,
        "image_shape": generator.image_shape,
        "generator_state_dict": generator.state_dict(),
    }
    if discriminator is not None:
        payload["discriminator_base_channels"] = discriminator.base_channels
        payload["discriminator_state_dict"] = discriminator.state_dict()
    if metrics is not None:
        payload["metrics"] = asdict(metrics)

    torch.save(payload, path)


def load_generator_checkpoint(
    path: str | Path,
    device: str | torch.device = "auto",
) -> MNISTGenerator:
    """Load a pretrained GAN generator from disk."""
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=device)
    image_shape = tuple(checkpoint.get("image_shape", (1, 32, 32)))
    if image_shape != (1, 32, 32):
        raise ValueError(
            f"Checkpoint {path} was trained for image_shape={image_shape}, "
            "but the current pipeline expects MNIST resized to (1, 32, 32). "
            "Use a 32x32 checkpoint such as "
            "'checkpoints/generator_wgangp_mnist32_e500_bs128_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu.pt'."
        )

    generator = MNISTGenerator(
        latent_dim=checkpoint["latent_dim"],
        base_channels=checkpoint["base_channels"],
        activation=checkpoint.get("activation", "relu"),
    )
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.to(device)
    generator.eval()
    return generator


def load_discriminator_checkpoint(
    path: str | Path,
    device: str | torch.device = "auto",
) -> MNISTDiscriminator:
    """Load the discriminator stored alongside a generator checkpoint."""
    device = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location=device)
    if "discriminator_state_dict" not in checkpoint:
        raise KeyError(f"No discriminator weights found in checkpoint {path}.")

    discriminator = MNISTDiscriminator(
        base_channels=checkpoint.get("discriminator_base_channels", 64),
    )
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    discriminator.to(device)
    discriminator.eval()
    return discriminator
