"""EADMM and PEADMM baselines aligned with the original EADMM framework."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from ops.forward_models import LinearSensingOperator
from ops.SR import SR
from utils.observations import ProblemBatch


@dataclass(frozen=True)
class SimplifiedADMMResult:
    """Outputs returned by the EADMM-style baselines."""

    reconstruction: torch.Tensor
    history: torch.Tensor
    latent: torch.Tensor
    penalty_history: torch.Tensor
    auxiliary: torch.Tensor


def _adaptive_penalty_schedule(beta_0: float, sigma_0: float):
    """Match the adaptive beta/sigma updates used in the original implementation."""
    beta, sigma = beta_0, sigma_0
    eta: float | None = None

    def next_values(t: int, infeasibility: float) -> tuple[float, float]:
        nonlocal beta, sigma, eta

        if infeasibility == 0 or t < 2:
            return beta, sigma

        if infeasibility <= beta_0 / sqrt(t):
            next_beta = beta
        else:
            next_beta = beta * sqrt(t / (t - 1))

        if infeasibility <= sigma_0 / t:
            eta = None
            next_sigma = sigma
        elif infeasibility <= sigma_0 / sqrt(t):
            eta = None
            next_sigma = sigma * sqrt((t - 1) / t)
        else:
            if eta is None:
                eta = sigma / max(next_beta - beta, 1e-12)
            next_sigma = eta * (next_beta - beta)

        beta, sigma = next_beta, next_sigma
        return beta, sigma

    return next_values


def _project_onto_l1_ball(v: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """Project each sample in ``v`` onto an L1 ball of the given radius."""
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")

    flat = v.view(v.shape[0], -1)
    abs_flat = flat.abs()
    l1_norm = abs_flat.sum(dim=1, keepdim=True)
    inside = l1_norm <= radius

    sorted_abs, _ = torch.sort(abs_flat, dim=1, descending=True)
    cssv = torch.cumsum(sorted_abs, dim=1) - radius
    arange = torch.arange(1, flat.shape[1] + 1, device=v.device, dtype=v.dtype)
    cond = sorted_abs - cssv / arange > 0
    rho = cond.sum(dim=1).clamp(min=1) - 1
    theta = cssv.gather(1, rho.unsqueeze(1)) / (rho.to(v.dtype).unsqueeze(1) + 1.0)
    projected = torch.sign(flat) * torch.clamp(abs_flat - theta, min=0.0)
    projected[inside.expand_as(projected)] = flat[inside.expand_as(flat)]
    return projected.view_as(v)


def _initialize_latent_code(
    init_mode: str,
    batch_size: int,
    latent_dim: int,
    device: torch.device,
    observation: torch.Tensor,
    encoder: nn.Module | None = None,
    random_init_std: float = 1.0,
) -> torch.Tensor:
    if init_mode == "learned":
        if encoder is None:
            raise ValueError("encoder is required for learned initialization.")
        with torch.no_grad():
            return encoder(observation).detach()
    if init_mode == "random":
        return random_init_std * torch.randn(batch_size, latent_dim, device=device)
    raise ValueError(f"Unsupported init_mode {init_mode!r}.")


def _generator_fidelity(
    reconstruction: torch.Tensor,
    problem: ProblemBatch,
    task: str,
    operator: LinearSensingOperator | SR | None = None,
) -> torch.Tensor:
    if task == "spi":
        if operator is None:
            raise ValueError("operator is required for the SPI task.")
        residual = operator(reconstruction) - problem.fidelity_target
        return 0.5 * torch.mean(torch.sum(residual * residual, dim=-1))
    if task == "SR":
        if operator is None:
            raise ValueError("operator is required for the SR task.")
        residual = operator(reconstruction) - problem.fidelity_target
        return 0.5 * torch.mean(torch.sum(residual.square(), dim=(1, 2, 3)))
    raise ValueError(f"Unsupported task {task!r}.")


def _al_value(
    x: torch.Tensor,
    prior: torch.Tensor,
    problem: ProblemBatch,
    task: str,
    dual: torch.Tensor,
    beta: float,
    operator: LinearSensingOperator | SR | None = None,
) -> torch.Tensor:
    if task == "spi":
        if operator is None:
            raise ValueError("operator is required for the SPI task.")
        fidelity = operator.measurement_error(x, problem.fidelity_target)
    elif task == "SR":
        if operator is None:
            raise ValueError("operator is required for the SR task.")
        residual = operator(x) - problem.fidelity_target
        fidelity = 0.5 * torch.mean(torch.sum(residual.square(), dim=(1, 2, 3)))
    else:
        raise ValueError(f"Unsupported task {task!r}.")

    diff = x - prior
    return fidelity + torch.mean(dual * diff) + 0.5 * beta * torch.mean(diff * diff)


def _exact_x_update(
    prior: torch.Tensor,
    problem: ProblemBatch,
    task: str,
    dual: torch.Tensor,
    beta: float,
    operator: LinearSensingOperator | SR | None = None,
) -> torch.Tensor:
    if task == "spi":
        if operator is None:
            raise ValueError("operator is required for the SPI task.")
        return operator.solve_x_exact(
            prior=prior,
            measurements=problem.fidelity_target,
            dual=dual,
            beta=beta,
        ).detach()
    if task == "SR":
        if operator is None:
            raise ValueError("operator is required for the SR task.")
        if not isinstance(operator, SR):
            raise ValueError("SR task requires an SR operator.")

        block_area = float(operator.s ** 2)
        shifted_prior = prior - dual / beta
        shifted_prior_lr = operator(shifted_prior)
        correction = (problem.fidelity_target - shifted_prior_lr) / (beta * block_area + 1.0)
        return (shifted_prior + operator.backprojection(correction)).detach()
    raise ValueError(f"Unsupported task {task!r}.")


def run_simplified_admm(
    generator: nn.Module,
    problem: ProblemBatch,
    task: str,
    num_iterations: int,
    gamma: float,
    beta: float,
    sigma: float,
    operator: LinearSensingOperator | SR | None = None,
    encoder: nn.Module | None = None,
    init_mode: str = "random",
    random_init_std: float = 1.0,
    record_iterations: set[int] | None = None,
) -> SimplifiedADMMResult:
    """Run an EADMM-style solver with either random or learned latent initialization."""
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}.")
    if gamma <= 0 or beta <= 0 or sigma <= 0:
        raise ValueError(f"gamma, beta and sigma must be positive, got {(gamma, beta, sigma)}.")
    if not hasattr(generator, "latent_dim"):
        raise ValueError("generator must expose a latent_dim attribute.")

    generator.eval()
    device = problem.observation.device
    batch_size = problem.observation.shape[0]
    latent_dim = int(generator.latent_dim)

    z = _initialize_latent_code(
        init_mode=init_mode,
        batch_size=batch_size,
        latent_dim=latent_dim,
        device=device,
        observation=problem.observation,
        encoder=encoder,
        random_init_std=random_init_std,
    )

    with torch.no_grad():
        prior = generator(z)
    x = prior.detach().clone()
    dual = torch.zeros_like(x)
    best_z = z.detach().clone()
    best_value = _generator_fidelity(prior, problem=problem, task=task, operator=operator).item()

    if record_iterations is not None:
        record_iterations = {int(item) for item in record_iterations if 0 <= int(item) <= num_iterations}
        record_iterations.add(0)
        record_iterations.add(num_iterations)

    recorded_iterations: list[int] = []
    history = []
    if record_iterations is None or 0 in record_iterations:
        history.append(prior.detach().cpu())
        recorded_iterations.append(0)
    penalty_history = [torch.norm(x - prior).detach().cpu()]
    next_penalties = _adaptive_penalty_schedule(beta_0=beta, sigma_0=sigma)

    for iteration in range(num_iterations):
        z_var = z.detach().requires_grad_(True)
        prior_var = generator(z_var)
        al_value = _al_value(
            x=x.detach(),
            prior=prior_var,
            problem=problem,
            task=task,
            dual=dual.detach(),
            beta=beta,
            operator=operator,
        )
        grad_z = torch.autograd.grad(al_value, z_var)[0]
        z = (z_var - (gamma / beta) * grad_z).detach()

        with torch.no_grad():
            prior = generator(z)
            x = _exact_x_update(
                prior=prior,
                problem=problem,
                task=task,
                dual=dual,
                beta=beta,
                operator=operator,
            )
            dual = dual + sigma * (x - prior)

            current_value = _generator_fidelity(
                prior,
                problem=problem,
                task=task,
                operator=operator,
            ).item()
            if current_value < best_value:
                best_value = current_value
                best_z = z.detach().clone()

            infeasibility = torch.norm(x - prior).item()
            beta, sigma = next_penalties(iteration + 1, infeasibility)

        step = iteration + 1
        if record_iterations is None or step in record_iterations:
            history.append(prior.detach().cpu())
            recorded_iterations.append(step)
        penalty_history.append(torch.norm(x - prior).detach().cpu())

    with torch.no_grad():
        best_reconstruction = generator(best_z)

    return SimplifiedADMMResult(
        reconstruction=best_reconstruction.detach(),
        history=torch.stack(history, dim=1),
        latent=best_z.detach(),
        penalty_history=torch.stack(penalty_history),
        auxiliary=x.detach(),
    )


def run_eadmm(
    generator: nn.Module,
    problem: ProblemBatch,
    task: str,
    num_iterations: int,
    gamma: float,
    beta: float,
    sigma: float,
    operator: LinearSensingOperator | SR | None = None,
    random_init_std: float = 1.0,
    record_iterations: set[int] | None = None,
) -> SimplifiedADMMResult:
    """Run EADMM with random latent initialization."""
    return run_simplified_admm(
        generator=generator,
        problem=problem,
        task=task,
        num_iterations=num_iterations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        operator=operator,
        encoder=None,
        init_mode="random",
        random_init_std=random_init_std,
        record_iterations=record_iterations,
    )


def run_peadmm(
    generator: nn.Module,
    encoder: nn.Module,
    problem: ProblemBatch,
    task: str,
    num_iterations: int,
    gamma: float,
    beta: float,
    sigma: float,
    operator: LinearSensingOperator | SR | None = None,
    record_iterations: set[int] | None = None,
) -> SimplifiedADMMResult:
    """Run PEADMM with learned latent initialization."""
    return run_simplified_admm(
        generator=generator,
        problem=problem,
        task=task,
        num_iterations=num_iterations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        operator=operator,
        encoder=encoder,
        init_mode="learned",
        record_iterations=record_iterations,
    )
