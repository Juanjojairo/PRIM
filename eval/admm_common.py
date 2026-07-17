"""Shared evaluation helpers for simplified EADMM and PEADMM baselines."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.mnist import get_mnist_dataloaders, to_display_mnist
from models.baselines import SimplifiedADMMResult, run_eadmm, run_peadmm
from models.encoder import load_encoder_checkpoint
from models.generator import load_generator_checkpoint, resolve_device
from ops.forward_models import LinearSensingOperator
from ops.SR import SR
from ops.metrics import ReconstructionMetrics, compute_metrics_curve, compute_reconstruction_metrics
from utils.experiments import format_float_token, join_name_parts
from utils.observations import ProblemBatch, build_problem_batch, freeze_module
from utils.visualization import save_labeled_mnist_grid, save_reconstruction_triplets_svg
from utils.wandb import add_wandb_args, init_wandb_run, namespace_to_config


SolverFn = Callable[..., SimplifiedADMMResult]


def evaluate_admm_solver(
    solver: SolverFn,
    method_name: str,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    task: str,
    num_iterations: int,
    gamma: float,
    beta: float,
    sigma: float,
    operator: LinearSensingOperator | SR | None = None,
    encoder: torch.nn.Module | None = None,
    measurement_noise_std: float = 0.0,
    max_batches: int | None = None,
    max_examples: int | None = None,
    random_init_std: float = 1.0,
    record_iterations: set[int] | None = None,
) -> tuple[
    ReconstructionMetrics,
    list[ReconstructionMetrics],
    ProblemBatch,
    SimplifiedADMMResult,
    torch.Tensor,
    int,
]:
    """Run a baseline solver over a split and aggregate final and per-iteration metrics."""
    if record_iterations is None:
        recorded_iterations = list(range(num_iterations + 1))
        record_set = None
    else:
        record_set = {int(item) for item in record_iterations if 0 <= int(item) <= num_iterations}
        record_set.add(0)
        record_set.add(num_iterations)
        recorded_iterations = sorted(record_set)

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    metrics_curve_sum = {
        metric_name: torch.zeros(len(recorded_iterations), dtype=torch.float64)
        for metric_name in ("mse", "mae", "psnr", "ssim")
    }
    total_examples = 0

    first_problem: ProblemBatch | None = None
    first_result: SimplifiedADMMResult | None = None
    first_targets: torch.Tensor | None = None

    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for images, labels in tqdm(batches, desc=f"eval-{method_name.lower()}", leave=False):
        remaining_examples = None if max_examples is None else max_examples - total_examples
        if remaining_examples is not None and remaining_examples <= 0:
            break
        if remaining_examples is not None and images.shape[0] > remaining_examples:
            images = images[:remaining_examples]
            labels = labels[:remaining_examples]
        if images.shape[0] == 0:
            break

        images = images.to(device)
        labels = labels.to(device)
        problem = build_problem_batch(
            task=task,
            images=images,
            operator=operator,
            measurement_noise_std=measurement_noise_std,
        )

        if solver is run_peadmm:
            result = solver(
                generator=generator,
                encoder=encoder,
                problem=problem,
                task=task,
                num_iterations=num_iterations,
                gamma=gamma,
                beta=beta,
                sigma=sigma,
                operator=operator,
                record_iterations=record_set,
            )
        else:
            result = solver(
                generator=generator,
                problem=problem,
                task=task,
                num_iterations=num_iterations,
                gamma=gamma,
                beta=beta,
                sigma=sigma,
                operator=operator,
                random_init_std=random_init_std,
                record_iterations=record_set,
            )

        batch_size = images.shape[0]
        predictions.append(result.reconstruction.cpu())
        targets.append(images.cpu())
        for index, step_metrics in enumerate(compute_metrics_curve(result.history.cpu(), images.cpu())):
            metrics_curve_sum["mse"][index] += step_metrics.mse * batch_size
            metrics_curve_sum["mae"][index] += step_metrics.mae * batch_size
            metrics_curve_sum["psnr"][index] += step_metrics.psnr * batch_size
            metrics_curve_sum["ssim"][index] += step_metrics.ssim * batch_size
        total_examples += batch_size

        if first_problem is None:
            first_problem = ProblemBatch(
                observation=problem.observation.detach().cpu(),
                fidelity_target=problem.fidelity_target.detach().cpu(),
            )
            first_result = result
            first_targets = images.detach().cpu()

    if first_problem is None or first_result is None or first_targets is None:
        raise RuntimeError("No examples were processed during ADMM evaluation.")

    final_metrics = compute_reconstruction_metrics(
        prediction=torch.cat(predictions, dim=0),
        target=torch.cat(targets, dim=0),
    )
    mean_metrics_curve = [
        ReconstructionMetrics(
            mse=(metrics_curve_sum["mse"][index] / max(total_examples, 1)).item(),
            mse_std=0.0,
            mae=(metrics_curve_sum["mae"][index] / max(total_examples, 1)).item(),
            mae_std=0.0,
            psnr=(metrics_curve_sum["psnr"][index] / max(total_examples, 1)).item(),
            psnr_std=0.0,
            ssim=(metrics_curve_sum["ssim"][index] / max(total_examples, 1)).item(),
            ssim_std=0.0,
        )
        for index in range(len(recorded_iterations))
    ]
    return (
        final_metrics,
        mean_metrics_curve,
        first_problem,
        first_result,
        first_targets,
        total_examples,
    )


def export_admm_qualitative_grid(
    save_path: str | Path,
    targets: torch.Tensor,
    observation: torch.Tensor,
    result: SimplifiedADMMResult,
    num_images: int = 8,
) -> None:
    """Save ground truth, observation, initialization, and final reconstruction."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_images = min(num_images, targets.shape[0], observation.shape[0], result.history.shape[0])
    initial = result.history[:num_images, 0]
    final = result.reconstruction[:num_images].cpu()
    save_labeled_mnist_grid(
        rows=[
            to_display_mnist(targets[:num_images].cpu()),
            to_display_mnist(observation[:num_images].cpu().clamp(0.0, 1.0)),
            to_display_mnist(initial.cpu()),
            to_display_mnist(final),
        ],
        row_labels=["GT", "Obs", "Init", "Final"],
        save_path=save_path,
    )


def write_baseline_metrics_csv(
    path: str | Path,
    method: str,
    split: str,
    task: str,
    metrics: ReconstructionMetrics,
    num_examples: int,
    num_iterations: int,
    gamma: float,
    beta: float,
    sigma: float,
    cr: float | None = None,
    scale: int | None = None,
    measurement_noise_std: float = 0.0,
) -> None:
    """Persist aggregate metrics for an ADMM baseline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "split",
                "task",
                "num_examples",
                "iterations",
                "gamma",
                "beta",
                "sigma",
                "cr",
                "scale",
                "measurement_noise_std",
                "mse",
                "mse_std",
                "mae",
                "mae_std",
                "psnr",
                "psnr_std",
                "ssim",
                "ssim_std",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": method,
                "split": split,
                "task": task,
                "num_examples": num_examples,
                "iterations": num_iterations,
                "gamma": gamma,
                "beta": beta,
                "sigma": sigma,
                "cr": "" if cr is None else f"{cr:g}",
                "scale": "" if scale is None else str(scale),
                "measurement_noise_std": f"{measurement_noise_std:g}",
                "mse": f"{metrics.mse:.6f}",
                "mse_std": f"{metrics.mse_std:.6f}",
                "mae": f"{metrics.mae:.6f}",
                "mae_std": f"{metrics.mae_std:.6f}",
                "psnr": f"{metrics.psnr:.4f}",
                "psnr_std": f"{metrics.psnr_std:.4f}",
                "ssim": f"{metrics.ssim:.4f}",
                "ssim_std": f"{metrics.ssim_std:.4f}",
            }
        )


def write_iteration_metrics_csv(
    path: str | Path,
    method: str,
    split: str,
    task: str,
    metrics_curve: list[ReconstructionMetrics],
    iterations: list[int] | None = None,
) -> None:
    """Persist average reconstruction metrics for every iteration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "split", "task", "iteration", "mse", "mse_std", "mae",  "mae_std", "psnr", "psnr_std", "ssim", "ssim_std"],
        )
        writer.writeheader()
        iteration_labels = iterations if iterations is not None else list(range(len(metrics_curve)))
        for iteration, metrics in zip(iteration_labels, metrics_curve):
            writer.writerow(
                {
                    "method": method,
                    "split": split,
                    "task": task,
                    "iteration": iteration,
                    "mse": f"{metrics.mse:.6f}",
                    "mse_std": f"{metrics.mse_std:.6f}",
                    "mae": f"{metrics.mae:.6f}",
                    "mae_std": f"{metrics.mae_std:.6f}",
                    "psnr": f"{metrics.psnr:.4f}",
                    "psnr_std": f"{metrics.psnr_std:.4f}",
                    "ssim": f"{metrics.ssim:.4f}",
                    "ssim_std": f"{metrics.ssim_std:.4f}",
                }
            )


def save_psnr_plot(
    path: str | Path,
    method: str,
    task: str,
    psnr_curve: torch.Tensor,
    title: str | None = None,
    iterations: list[int] | None = None,
) -> None:
    """Save a publication-style PSNR-vs-iteration plot."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    iterations = iterations if iterations is not None else list(range(len(psnr_curve)))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(iterations, psnr_curve.tolist(), marker="o", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average PSNR over evaluated samples (dB)")
    ax.set_title(title or f"{method} on {task}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_parser(method_name: str) -> argparse.ArgumentParser:
    """Build a CLI parser for ADMM baseline evaluation scripts."""
    parser = argparse.ArgumentParser(description=f"Evaluate the {method_name} baseline.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--generator-checkpoint", type=Path,
                        default=Path("results/generator_wgangp_mnist32_e500_bs128"
                                     "_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt"))
    parser.add_argument("--encoder-checkpoint", type=Path,
                        # default=Path("results/encoder_spi_mnist32_cr_1e-1_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"))
                        default=Path("results/encoder_spi_mnist32_cr_1e-2_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"))
    parser.add_argument("--save-root", type=Path, default=Path("results"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--task", type=str, default="spi", choices=["spi", "SR"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cr", type=float, default=0.01)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--gamma", type=float, default=200)
    parser.add_argument("--beta", type=float, default=0.005)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--random-init-std", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--num-qualitative", type=int, default=8)
    parser.add_argument("--num-visual", type=int, default=8)
    return add_wandb_args(parser, default_job_type=f"eval_{method_name.lower()}")

def run_baseline_cli(method_name: str) -> None:
    """Entry point shared by eval_eadmm.py and eval_peadmm.py."""
    if method_name not in {"EADMM", "PEADMM"}:
        raise ValueError(f"Unsupported method_name {method_name!r}.")

    args = build_parser(method_name).parse_args()
    name_parts: list[object] = [
        "eval",
        method_name.lower(),
        args.task,
        "mnist32",
        args.split,
        f"iter{args.iterations}",
        f"bs{args.batch_size}",
        f"gamma_{format_float_token(args.gamma)}",
        f"beta_{format_float_token(args.beta)}",
        f"sigma_{format_float_token(args.sigma)}",
    ]
    name_parts.extend(
        [
            (
                f"cr_{format_float_token(args.cr)}"
                if args.task == "spi"
                else f"sr_{args.scale}"
            ),
            f"noise_{format_float_token(args.measurement_noise_std)}",
        ]
    )
    if method_name == "EADMM":
        name_parts.append(f"rinit_{format_float_token(args.random_init_std)}")
    if args.max_batches is not None:
        name_parts.append(f"mb{args.max_batches}")
    if args.max_examples is not None:
        name_parts.append(f"n{args.max_examples}")
    experiment_name = join_name_parts(*name_parts)
    args.output_dir = args.save_root / experiment_name
    if args.wandb_run_name is None:
        args.wandb_run_name = experiment_name

    device = resolve_device(args.device)
    wandb_logger = init_wandb_run(
        args,
        config=namespace_to_config(args),
        tags=["eval", method_name.lower(), args.task, experiment_name],
    )

    try:
        generator = load_generator_checkpoint(args.generator_checkpoint, device=device)
        freeze_module(generator)

        encoder = None
        if args.encoder_checkpoint is not None:
            encoder = load_encoder_checkpoint(args.encoder_checkpoint, device=device)
            freeze_module(encoder)

        loaders = get_mnist_dataloaders(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            download=args.download,
        )
        loader = loaders[args.split]

        if args.task == "spi":
            operator = LinearSensingOperator(cr=args.cr).to(device)
        else:
            operator = SR(s=args.scale).to(device)

        solver = run_peadmm if method_name == "PEADMM" else run_eadmm
        metrics, metrics_curve, first_problem, first_result, first_targets, num_examples = evaluate_admm_solver(
            solver=solver,
            method_name=method_name,
            generator=generator,
            loader=loader,
            device=device,
            task=args.task,
            num_iterations=args.iterations,
            gamma=args.gamma,
            beta=args.beta,
            sigma=args.sigma,
            operator=operator,
            encoder=encoder,
            measurement_noise_std=args.measurement_noise_std,
            max_batches=args.max_batches,
            max_examples=args.max_examples,
            random_init_std=args.random_init_std,
        )
        psnr_curve = torch.tensor([step_metrics.psnr for step_metrics in metrics_curve], dtype=torch.float32)
        task_label = (
            f"SPI (CR={args.cr:g})"
            if args.task == "spi"
            else f"SR (x{args.scale})"
        )

        metrics_path = args.output_dir / f"{args.task}_{args.split}_metrics.csv"
        curve_csv_path = args.output_dir / f"{args.task}_{args.split}_iteration_metrics.csv"
        curve_plot_path = args.output_dir / f"{args.task}_{args.split}_psnr_curve.png"
        qualitative_path = args.output_dir / f"{args.task}_{args.split}_qualitative.png"
        visual_svg_path = args.output_dir / f"{args.task}_{args.split}_visual_triplets.svg"

        write_baseline_metrics_csv(
            path=metrics_path,
            method=method_name,
            split=args.split,
            task=args.task,
            metrics=metrics,
            num_examples=num_examples,
            num_iterations=args.iterations,
            gamma=args.gamma,
            beta=args.beta,
            sigma=args.sigma,
            cr=args.cr if args.task == "spi" else None,
            scale=args.scale if args.task == "SR" else None,
            measurement_noise_std=args.measurement_noise_std,
        )
        write_iteration_metrics_csv(
            path=curve_csv_path,
            method=method_name,
            split=args.split,
            task=args.task,
            metrics_curve=metrics_curve,
        )
        save_psnr_plot(
            path=curve_plot_path,
            method=method_name,
            task=args.task,
            psnr_curve=psnr_curve,
            title=f"{method_name} on {task_label}",
        )
        export_admm_qualitative_grid(
            save_path=qualitative_path,
            targets=first_targets,
            observation=first_problem.observation,
            result=first_result,
            num_images=args.num_qualitative,
        )
        save_reconstruction_triplets_svg(
            save_path=visual_svg_path,
            backprojection=first_problem.observation,
            estimate=first_result.reconstruction,
            target=first_targets,
            num_images=args.num_visual,
            title=f"{method_name} {args.task}/{args.split}",
        )

        prefix = f"{method_name.lower()}/{args.task}/{args.split}"
        panel_prefix = method_name.lower()
        wandb_logger.summary(
            {
                "device": str(device),
                f"{prefix}/num_examples": num_examples,
                f"{prefix}/cr": args.cr if args.task == "spi" else None,
                f"{prefix}/scale": args.scale if args.task == "SR" else None,
                "paths/metrics_csv": str(metrics_path),
                "paths/curve_csv": str(curve_csv_path),
                "paths/curve_plot": str(curve_plot_path),
                "paths/reconstructions": str(qualitative_path),
                "paths/visual_svg": str(visual_svg_path),
            }
        )
        for iteration, step_metrics in enumerate(metrics_curve):
            wandb_logger.log(
                {
                    f"{panel_prefix}/mse": step_metrics.mse,
                    f"{panel_prefix}/mae": step_metrics.mae,
                    f"{panel_prefix}/psnr": step_metrics.psnr,
                    f"{panel_prefix}/ssim": step_metrics.ssim,
                    "iter": iteration,
                }
            )
        wandb_logger.log_table(f"{panel_prefix}/metrics_table", curve_csv_path)
        wandb_logger.log_image(
            f"{panel_prefix}/reconstructions",
            qualitative_path,
            caption=f"{method_name} qualitative grid on {args.task}/{args.split}",
        )

        print(f"method={method_name}")
        print(f"split={args.split}")
        print(f"task={args.task}")
        print(f"num_examples={num_examples}")
        print(f"mse={metrics.mse:.6f} ± {metrics.mse_std:.6f}")
        print(f"mae={metrics.mae:.6f} ± {metrics.mae_std:.6f}")
        print(f"psnr={metrics.psnr:.4f} ± {metrics.psnr_std:.4f}")
        print(f"ssim={metrics.ssim:.4f} ± {metrics.ssim_std:.4f}")
        print(f"metrics_path={metrics_path}")
        print(f"curve_csv_path={curve_csv_path}")
        print(f"curve_plot_path={curve_plot_path}")
        print(f"qualitative_path={qualitative_path}")
        print(f"visual_svg_path={visual_svg_path}")
    finally:
        wandb_logger.finish()
