"""Recurrent Inference Machine variants for iterative reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ops.forward_models import LinearSensingOperator


class ConvGRUCell(nn.Module):
    """Small ConvGRU cell used to maintain state across RIM iterations."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or hidden_channels <= 0:
            raise ValueError(
                f"input_channels and hidden_channels must be positive, got "
                f"{input_channels=} and {hidden_channels=}."
            )

        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([inputs, hidden], dim=1)
        update_gate, reset_gate = torch.chunk(torch.sigmoid(self.gates(combined)), chunks=2, dim=1)
        candidate_input = torch.cat([inputs, reset_gate * hidden], dim=1)
        candidate = torch.tanh(self.candidate(candidate_input))
        return (1.0 - update_gate) * hidden + update_gate * candidate


class ImageSpaceRIM(nn.Module):
    """Recurrent Inference Machine operating directly in image space."""

    def __init__(
        self,
        hidden_channels: int = 32,
        steps: int = 10,
        step_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}.")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        if step_scale <= 0:
            raise ValueError(f"step_scale must be positive, got {step_scale}.")

        self.hidden_channels = hidden_channels
        self.steps = steps
        self.step_scale = step_scale

        self.input_encoder = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cell = ConvGRUCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )
        self.update_head = nn.Sequential(
            nn.Conv2d(2 * hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def _data_consistency_correction(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        operator: LinearSensingOperator | None,
    ) -> torch.Tensor:
        if operator is None:
            return y - x
        return operator.adjoint(y - operator(x))

    def forward(
        self,
        y: torch.Tensor,
        operator: LinearSensingOperator | None,
        x0: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Unroll the RIM and return the reconstruction history."""
        if x0.ndim != 4 or x0.shape[1] != 1:
            raise ValueError(f"Expected x0 with shape [B, 1, H, W], got {tuple(x0.shape)}.")

        total_steps = self.steps if steps is None else steps
        if total_steps <= 0:
            raise ValueError(f"steps must be positive, got {total_steps}.")
        
        x = x0
        hidden = torch.zeros(
            x.shape[0],
            self.hidden_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )

        history = [x]
        for _ in range(total_steps):
            correction = self._data_consistency_correction(x=x, y=y, operator=operator)
            encoded = self.input_encoder(torch.cat([x, correction], dim=1))
            hidden = self.cell(encoded, hidden)
            delta = self.step_scale * self.update_head(torch.cat([encoded, hidden], dim=1))
            x = x + delta
            history.append(x)

        return torch.stack(history, dim=1)


class LatentImageRIM(nn.Module):
    """RIM variant that jointly updates image and latent states."""

    def __init__(
        self,
        latent_dim: int,
        hidden_channels: int = 32,
        latent_hidden_dim: int | None = None,
        steps: int = 10,
        step_scale: float = 0.1,
        latent_step_scale: float = 0.1,
        lambda_prior: float = 0.1,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}.")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        if step_scale <= 0:
            raise ValueError(f"step_scale must be positive, got {step_scale}.")
        if latent_step_scale <= 0:
            raise ValueError(f"latent_step_scale must be positive, got {latent_step_scale}.")
        if lambda_prior < 0:
            raise ValueError(f"lambda_prior must be non-negative, got {lambda_prior}.")

        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.latent_hidden_dim = max(latent_hidden_dim or latent_dim, 1)
        self.steps = steps
        self.step_scale = step_scale
        self.latent_step_scale = latent_step_scale
        self.lambda_prior = lambda_prior

        self.image_encoder = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.image_cell = ConvGRUCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=3,
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(2 * latent_dim, self.latent_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_hidden_dim, self.latent_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.latent_cell = nn.GRUCell(self.latent_hidden_dim, self.latent_hidden_dim)
        self.image_to_latent = nn.Linear(hidden_channels, self.latent_hidden_dim)
        self.latent_to_image = nn.Linear(self.latent_hidden_dim, hidden_channels)
        self.update_head_x = nn.Sequential(
            nn.Conv2d(3 * hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.update_head_z = nn.Sequential(
            nn.Linear(3 * self.latent_hidden_dim, self.latent_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def _image_correction(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        operator: LinearSensingOperator | None,
        generated: torch.Tensor,
    ) -> torch.Tensor:
        if operator is None:
            measurement_term = y - x
        else:
            measurement_term = operator.adjoint(y - operator(x))
        return measurement_term - self.lambda_prior * (x - generated)

    def _latent_correction(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        generator: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.enable_grad():
            z_var = z.detach().requires_grad_(True)
            generated = generator(z_var)
            prior_energy = 0.5 * self.lambda_prior * torch.sum((x.detach() - generated) ** 2)
            grad_z = torch.autograd.grad(prior_energy, z_var, create_graph=False)[0]
        return -grad_z.detach(), generated.detach()

    def forward(
        self,
        y: torch.Tensor,
        operator: LinearSensingOperator | None,
        x0: torch.Tensor,
        z0: torch.Tensor,
        generator: nn.Module,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Unroll the coupled RIM and return image and latent histories."""
        if x0.ndim != 4 or x0.shape[1] != 1:
            raise ValueError(f"Expected x0 with shape [B, 1, H, W], got {tuple(x0.shape)}.")
        if z0.ndim != 2 or z0.shape[0] != x0.shape[0] or z0.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected z0 with shape [B, {self.latent_dim}], got {tuple(z0.shape)}."
            )

        total_steps = self.steps if steps is None else steps
        if total_steps <= 0:
            raise ValueError(f"steps must be positive, got {total_steps}.")

        lower, upper = (-1.0, 1.0) if x0.min().item() < 0.0 else (0.0, 1.0)
        x = x0
        z = z0
        hidden_x = torch.zeros(
            x.shape[0],
            self.hidden_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )
        hidden_z = torch.zeros(
            z.shape[0],
            self.latent_hidden_dim,
            device=z.device,
            dtype=z.dtype,
        )

        history_x = [x]
        history_z = [z]
        for _ in range(total_steps):
            correction_z, generated = self._latent_correction(x=x, z=z, generator=generator)
            correction_x = self._image_correction(x=x, y=y, operator=operator, generated=generated)

            encoded_x = self.image_encoder(torch.cat([x, correction_x], dim=1))
            hidden_x = self.image_cell(encoded_x, hidden_x)
            pooled_x = F.adaptive_avg_pool2d(hidden_x, output_size=1).flatten(1)

            encoded_z = self.latent_encoder(torch.cat([z, correction_z], dim=1))
            hidden_z = self.latent_cell(encoded_z + self.image_to_latent(pooled_x), hidden_z)

            latent_context = self.latent_to_image(hidden_z).unsqueeze(-1).unsqueeze(-1)
            latent_context = latent_context.expand(-1, -1, x.shape[2], x.shape[3])
            delta_x = self.step_scale * self.update_head_x(
                torch.cat([encoded_x, hidden_x, latent_context], dim=1)
            )
            delta_z = self.latent_step_scale * self.update_head_z(
                torch.cat([encoded_z, hidden_z, self.image_to_latent(pooled_x)], dim=1)
            )

            x = (x + delta_x).clamp(lower, upper)
            z = z + delta_z
            history_x.append(x)
            history_z.append(z)

        return torch.stack(history_x, dim=1), torch.stack(history_z, dim=1)


@dataclass(frozen=True)
class RIMCheckpointMetrics:
    """Metrics stored alongside a trained RIM checkpoint."""

    train_loss: float
    val_loss: float
    val_psnr: float


def save_rim_checkpoint(
    path: str | Path,
    model: ImageSpaceRIM | LatentImageRIM,
    metrics: RIMCheckpointMetrics | None = None,
) -> None:
    """Save RIM weights and metadata to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_class": type(model).__name__,
        "hidden_channels": model.hidden_channels,
        "steps": model.steps,
        "step_scale": model.step_scale,
        "state_dict": model.state_dict(),
    }
    if isinstance(model, LatentImageRIM):
        payload.update(
            {
                "latent_dim": model.latent_dim,
                "latent_hidden_dim": model.latent_hidden_dim,
                "latent_step_scale": model.latent_step_scale,
                "lambda_prior": model.lambda_prior,
            }
        )
    if metrics is not None:
        payload["metrics"] = asdict(metrics)

    torch.save(payload, path)


def load_rim_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> ImageSpaceRIM | LatentImageRIM:
    """Load a trained RIM checkpoint from disk."""
    checkpoint = torch.load(Path(path), map_location=device)
    model_class = checkpoint.get("model_class", "ImageSpaceRIM")
    if model_class == "LatentImageRIM":
        model = LatentImageRIM(
            latent_dim=checkpoint["latent_dim"],
            hidden_channels=checkpoint["hidden_channels"],
            latent_hidden_dim=checkpoint.get("latent_hidden_dim"),
            steps=checkpoint["steps"],
            step_scale=checkpoint["step_scale"],
            latent_step_scale=checkpoint.get("latent_step_scale", checkpoint["step_scale"]),
            lambda_prior=checkpoint.get("lambda_prior", 0.1),
        )
    else:
        model = ImageSpaceRIM(
            hidden_channels=checkpoint["hidden_channels"],
            steps=checkpoint["steps"],
            step_scale=checkpoint["step_scale"],
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
