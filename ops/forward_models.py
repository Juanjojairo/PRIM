"""Simple SPI forward model with zig-zag ordered Hadamard patterns."""

from __future__ import annotations

import argparse
from math import isclose

from einops import rearrange
import torch
from torch import nn

from ops.libs.ordering import get_matrix


VALID_CRS = (1.0, 0.1, 0.05, 0.01, 0.005)


def validate_cr(cr: float) -> float:
    """Keep experiments on the compression ratios used in the project."""
    for valid in VALID_CRS:
        if isclose(cr, valid, rel_tol=0.0, abs_tol=1e-12):
            return valid
    raise ValueError(f"cr must be one of {VALID_CRS}, got {cr}.")


class LinearSensingOperator(nn.Module):
    """Single-pixel camera operator: y = Hx with zig-zag Hadamard rows."""

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (1, 32, 32),
        cr: float | None = None,
        *,
        ordering: str = "zig_zag",
    ) -> None:
        super().__init__()
        channels, height, width = image_shape
        if channels != 1 or height != width:
            raise ValueError(f"Expected grayscale square images, got {image_shape}.")

        cr = validate_cr(cr)
        num_pixels = height * width
        num_measurements = max(1, int(round(num_pixels * cr)))
        matrix = torch.from_numpy(get_matrix(num_pixels, ordering)[:num_measurements]).float()

        self.image_shape = image_shape
        self.image_size = height
        self.cr = cr
        self.ordering = ordering
        self.num_pixels = num_pixels
        self.num_measurements = num_measurements
        self.register_buffer("matrix", matrix)

    def extra_repr(self) -> str:
        return f"image_shape={self.image_shape}, cr={self.cr}, ordering={self.ordering}"

    def image_to_vec(self, x: torch.Tensor) -> torch.Tensor:
        """Match the original optical encoder vectorization: b c m n -> b (c n m)."""
        return rearrange(x, 'b c m n -> b (c n m)')

    def vec_to_image(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, 'b (c n m) -> b c m n', c=1, n=self.image_size, m=self.image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze = x.ndim == 3
        y = self.image_to_vec(x) @ self.matrix.T
        return y.squeeze(0) if squeeze else y

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        squeeze = y.ndim == 1
        if squeeze:
            y = y.unsqueeze(0)
        if y.ndim != 2 or y.shape[1] != self.num_measurements:
            raise ValueError(f"Expected measurements (*, {self.num_measurements}), got {tuple(y.shape)}.")
        x = self.vec_to_image(y @ self.matrix)
        return x.squeeze(0) if squeeze else x

    def gram(self) -> torch.Tensor:
        return self.matrix.T @ self.matrix

    def backprojection(self, y: torch.Tensor) -> torch.Tensor:
        """Return the standard matched-filter backprojection H^T y / N."""
        return self.adjoint(y) / self.num_pixels

    def measurement_error(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        residual = self(x) - y
        return 0.5 * torch.mean(torch.sum(residual.square(), dim=-1))

    def measurement_gradient(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self(x) - y)

    def solve_x_exact(
        self,
        prior: torch.Tensor,
        measurements: torch.Tensor,
        dual: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        """Exact ADMM x-update for 0.5||Hx-y||^2 + beta/2||x-prior+dual/beta||^2."""
        rhs = self.image_to_vec(self.adjoint(measurements)) - self.image_to_vec(dual)
        rhs = rhs + beta * self.image_to_vec(prior)
        system = self.gram() + beta * torch.eye(
            self.num_pixels,
            device=self.matrix.device,
            dtype=self.matrix.dtype,
        )
        x = torch.linalg.solve(system, rhs.T).T
        return self.vec_to_image(x)


SinglePixelHadamardOperator = LinearSensingOperator


def smoke_test(cr: float = 0.1) -> None:
    operator = LinearSensingOperator(cr=cr).double()
    x = torch.rand(4, 1, 32, 32, dtype=torch.float64)
    y = operator(x)
    x_back = operator.adjoint(y)
    x_bp = operator.backprojection(y)

    lhs = torch.sum(operator(x) * y)
    rhs = torch.sum(x * x_back)
    rel_error = abs(lhs.item() - rhs.item()) / max(abs(lhs.item()), abs(rhs.item()), 1.0)

    print(operator)
    print(f"num_measurements={operator.num_measurements}")
    print(f"measurement_shape={tuple(y.shape)}")
    print(f"backprojection_shape={tuple(x_back.shape)}")
    print(f"matched_filter_shape={tuple(x_bp.shape)}")
    print(f"relative_adjoint_error={rel_error:.6e}")
    if cr == 1.0:
        print(f"matched_filter_mse={torch.mean((x_bp - x).square()).item():.6e}")

    if y.shape != (4, operator.num_measurements) or x_back.shape != x.shape or x_bp.shape != x.shape:
        raise RuntimeError("Shape check failed.")
    if rel_error > 1e-10:
        raise RuntimeError("Adjoint check failed.")
    if cr == 1.0 and torch.mean((x_bp - x).square()).item() > 1e-10:
        raise RuntimeError("Matched filter check failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the SPI operator.")
    parser.add_argument("--cr", type=float, default=0.1)
    args = parser.parse_args()
    smoke_test(cr=args.cr)


if __name__ == "__main__":
    main()
