"""Shared helpers to build degraded observations for different reconstruction tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ops.forward_models import LinearSensingOperator


@dataclass(frozen=True)
class ProblemBatch:
    """Inputs required by reconstruction methods for one degraded batch."""

    observation: torch.Tensor
    fidelity_target: torch.Tensor


def build_observation_batch(
    task: str,
    images: torch.Tensor,
    operator: LinearSensingOperator | None = None,
    measurement_noise_std: float = 0.0,
) -> torch.Tensor:
    """Construct the degraded observation used by a given task."""
    return build_problem_batch(
        task=task,
        images=images,
        operator=operator,
        measurement_noise_std=measurement_noise_std,
    ).observation


def build_problem_batch(
    task: str,
    images: torch.Tensor,
    operator: LinearSensingOperator | None = None,
    measurement_noise_std: float = 0.0,
) -> ProblemBatch:
    """Construct both the observation and the data-fidelity target."""
    if operator is None:
        raise ValueError(f"operator is required for the {task} task.")

    if task == "spi":
        measurements = operator(images)
        if measurement_noise_std > 0:
            measurements = measurements + measurement_noise_std * torch.randn_like(measurements)
        return ProblemBatch(
            observation=operator.backprojection(measurements),
            fidelity_target=measurements,
        )

    if task == "SR":
        measurements = operator(images)
        if measurement_noise_std > 0:
            measurements = measurements + measurement_noise_std * torch.randn_like(measurements)
        return ProblemBatch(
            observation=operator.backprojection(measurements),
            fidelity_target=measurements,
        )
    raise ValueError(f"Unsupported task {task!r}.")


def freeze_module(module: torch.nn.Module) -> None:
    """Disable gradient updates for a module."""
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
