"""Hadamard transform utilities for fast structured sensing operators."""

from __future__ import annotations

import math

import torch


def is_power_of_two(value: int) -> bool:
    """Return True when ``value`` is a strictly positive power of two."""
    return value > 0 and (value & (value - 1)) == 0


def fwht(x: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    """Apply the Fast Walsh-Hadamard Transform along the last dimension.

    The transform is self-inverse. When ``normalized=True`` it becomes
    orthonormal, so the same routine can be used for both forward and inverse
    transforms.
    """

    if x.ndim == 0:
        raise ValueError("fwht expects a tensor with at least one dimension.")

    n = x.shape[-1]
    if not is_power_of_two(n):
        raise ValueError(
            f"fwht requires the last dimension to be a power of two, got {n}."
        )

    original_shape = x.shape
    output = x.reshape(-1, n).contiguous()

    h = 1
    while h < n:
        output = output.view(-1, n // (2 * h), 2, h)
        a = output[:, :, 0, :]
        b = output[:, :, 1, :]
        output = torch.cat((a + b, a - b), dim=2)
        output = output.view(-1, n)
        h *= 2

    if normalized:
        output = output / math.sqrt(n)

    return output.view(*original_shape)
