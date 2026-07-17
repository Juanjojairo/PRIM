"""Evaluate the trained RIM only."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.mnist import get_mnist_dataloaders, to_display_mnist
from models.encoder import load_encoder_checkpoint
from models.generator import load_generator_checkpoint, resolve_device
from models.rim import ImageSpaceRIM, LatentImageRIM, load_rim_checkpoint
from ops.forward_models import LinearSensingOperator
from ops.SR import SR
from ops.metrics import ReconstructionMetrics, compute_metrics_curve, compute_reconstruction_metrics
from utils.experiments import format_float_token, join_name_parts
from utils.observations import ProblemBatch, build_problem_batch, freeze_module
from utils.seed import set_seed
from utils.visualization import save_reconstruction_triplets_svg
from utils.wandb import WandbLogger, add_wandb_args, init_wandb_run, namespace_to_config


def build_rim_initialization(
    use_generator_prior: bool,
    init_mode: str,
    generator: torch.nn.Module,
    encoder: torch.nn.Module,
    problem: ProblemBatch,
    task: str,
    random_init_std: float = 1.0,
) -> torch.Tensor:
    """Build the initial reconstruction used by the RIM."""
    if not use_generator_prior:
        return problem.observation
    if init_mode == "learned":
        return generator(encoder(problem.observation))
    if init_mode == "random_latent":
        if not hasattr(generator, "latent_dim"):
            raise ValueError("generator must expose latent_dim for random latent initialization.")
        latent = random_init_std * torch.randn(
            problem.observation.shape[0],
            int(generator.latent_dim),
            device=problem.observation.device,
            dtype=problem.observation.dtype,
        )
        return generator(latent)
    if init_mode == "backprojection":
        return problem.observation
    raise ValueError(f"Unsupported init_mode {init_mode!r}.")


def build_rim_initial_state(
    use_generator_prior: bool,
    init_mode: str,
    generator: torch.nn.Module,
    encoder: torch.nn.Module,
    problem: ProblemBatch,
    task: str,
    random_init_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build the initial reconstruction and optional latent code."""
    if not use_generator_prior:
        return problem.observation, None
    if init_mode == "learned":
        latent = encoder(problem.observation)
        return generator(latent), latent
    if init_mode == "random_latent":
        if not hasattr(generator, "latent_dim"):
            raise ValueError("generator must expose latent_dim for random latent initialization.")
        latent = random_init_std * torch.randn(
            problem.observation.shape[0],
            int(generator.latent_dim),
            device=problem.observation.device,
            dtype=problem.observation.dtype,
        )
        return generator(latent), latent
    if init_mode == "backprojection":
        return problem.observation, None
    raise ValueError(f"Unsupported init_mode {init_mode!r}.")


def run_rim_rollout(
    rim: ImageSpaceRIM | LatentImageRIM,
    y: torch.Tensor,
    operator: LinearSensingOperator | SR | None,
    x0: torch.Tensor,
    generator: torch.nn.Module,
    z0: torch.Tensor | None = None,
    steps: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run either RIM variant through a shared interface."""
    if isinstance(rim, LatentImageRIM):
        if z0 is None:
            raise ValueError("z0 is required for LatentImageRIM.")
        history_x, history_z = rim(
            y=y,
            operator=operator,
            x0=x0,
            z0=z0,
            generator=generator,
            steps=steps,
        )
        return history_x, history_z

    history_x = rim(
        y=y,
        operator=operator,
        x0=x0,
        steps=steps,
    )
    return history_x, None


def evaluate_proposed_rim(
    rim: ImageSpaceRIM | LatentImageRIM,
    encoder: torch.nn.Module,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    task: str,
    operator: LinearSensingOperator | SR | None = None,
    measurement_noise_std: float = 0.0,
    max_batches: int | None = None,
    use_generator_prior: bool = True,
    init_mode: str = "learned",
    steps_override: int | None = None,
    random_init_std: float = 1.0,
    desc: str = "eval-rim",
) -> tuple[
    ReconstructionMetrics,
    list[ReconstructionMetrics],
    ProblemBatch,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    """Evaluate the proposed RIM and return aggregate metrics and a reference batch."""
    rim.eval()
    encoder.eval()
    generator.eval()

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    total_steps = rim.steps if steps_override is None else steps_override
    metrics_curve_sum = {
        metric_name: torch.zeros(total_steps + 1, dtype=torch.float64)
        for metric_name in ("mse", "mae", "psnr", "ssim")
    }
    total_examples = 0

    first_problem: ProblemBatch | None = None
    first_history: torch.Tensor | None = None
    first_targets: torch.Tensor | None = None

    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for images, labels in tqdm(batches, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        problem = build_problem_batch(
            task=task,
            images=images,
            operator=operator,
            measurement_noise_std=measurement_noise_std,
        )

        with torch.no_grad():
            x0, z0 = build_rim_initial_state(
                use_generator_prior=use_generator_prior,
                init_mode=init_mode,
                generator=generator,
                encoder=encoder,
                problem=problem,
                task=task,
                random_init_std=random_init_std,
            )
            history, _ = run_rim_rollout(
                rim=rim,
                y=problem.fidelity_target,
                operator=operator,
                x0=x0,
                z0=z0,
                generator=generator,
                steps=steps_override,
            )

        batch_size = images.shape[0]
        predictions.append(history[:, -1].cpu())
        targets.append(images.cpu())
        for iteration, step_metrics in enumerate(compute_metrics_curve(history.cpu(), images.cpu())):
            metrics_curve_sum["mse"][iteration] += step_metrics.mse * batch_size
            metrics_curve_sum["mae"][iteration] += step_metrics.mae * batch_size
            metrics_curve_sum["psnr"][iteration] += step_metrics.psnr * batch_size
            metrics_curve_sum["ssim"][iteration] += step_metrics.ssim * batch_size
        total_examples += batch_size

        if first_problem is None:
            first_problem = ProblemBatch(
                observation=problem.observation.detach().cpu(),
                fidelity_target=problem.fidelity_target.detach().cpu(),
            )
            first_history = history.detach().cpu()
            first_targets = images.detach().cpu()

    if first_problem is None or first_history is None or first_targets is None:
        raise RuntimeError("No examples were processed during RIM evaluation.")

    metrics = compute_reconstruction_metrics(
        prediction=torch.cat(predictions, dim=0),
        target=torch.cat(targets, dim=0),
    )
    mean_metrics_curve = [
        ReconstructionMetrics(
            mse=(metrics_curve_sum["mse"][iteration] / max(total_examples, 1)).item(),
            mse_std=0.0,
            mae=(metrics_curve_sum["mae"][iteration] / max(total_examples, 1)).item(),
            mae_std=0.0,
            psnr=(metrics_curve_sum["psnr"][iteration] / max(total_examples, 1)).item(),
            psnr_std=0.0,
            ssim=(metrics_curve_sum["ssim"][iteration] / max(total_examples, 1)).item(),
            ssim_std=0.0,
        )
        for iteration in range(total_steps + 1)
    ]
    return metrics, mean_metrics_curve, first_problem, first_history, first_targets, total_examples
def save_combined_psnr_plot(
    save_path: str | Path,
    task: str,
    psnr_curves: dict[str, torch.Tensor],
    title: str | None = None,
) -> None:
    """Save a combined PSNR-vs-iteration plot for all methods."""
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for method, curve in psnr_curves.items():
        values = curve.tolist()
        if len(values) == 1:
            ax.axhline(values[0], linestyle="--", linewidth=2, label=f"{method} (final)")
        else:
            ax.plot(range(len(values)), values, marker="o", linewidth=2, label=method)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average PSNR over evaluated samples (dB)")
    ax.set_title(title or f"Method Comparison on {task}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_aggregate_metrics_csv(
    path: str | Path,
    split: str,
    task: str,
    results: list[dict[str, str | int | float]],
) -> None:
    """Write aggregate method metrics to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["method", "split", "task", "num_examples", "iterations", "mse", "mae", "psnr", "ssim"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def write_psnr_curves_csv(
    path: str | Path,
    task: str,
    psnr_curves: dict[str, torch.Tensor],
) -> None:
    """Write all PSNR curves in long CSV format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "task", "iteration", "psnr"])
        writer.writeheader()
        for method, curve in psnr_curves.items():
            for iteration, psnr in enumerate(curve.tolist()):
                writer.writerow(
                    {
                        "method": method,
                        "task": task,
                        "iteration": iteration,
                        "psnr": f"{psnr:.4f}",
                    }
                )


def write_iteration_metrics_curves_csv(
    path: str | Path,
    task: str,
    metrics_curves: dict[str, list[ReconstructionMetrics]],
) -> None:
    """Write per-iteration average metrics for one or more methods."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "task", "iteration", "mse", "mae", "psnr", "ssim"],
        )
        writer.writeheader()
        for method, curve in metrics_curves.items():
            for iteration, metrics in enumerate(curve):
                writer.writerow(
                    {
                        "method": method,
                        "task": task,
                        "iteration": iteration,
                        "mse": f"{metrics.mse:.6f}",
                        "mae": f"{metrics.mae:.6f}",
                        "psnr": f"{metrics.psnr:.4f}",
                        "ssim": f"{metrics.ssim:.4f}",
                    }
                )


def write_markdown_table(
    path: str | Path,
    split: str,
    task: str,
    results: list[dict[str, str | int | float]],
    title: str | None = None,
) -> None:
    """Write a compact Markdown table for paper-ready summaries."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        title or f"# Results: {task} / {split}",
        "",
        "| Method | Iterations | MSE | MAE | PSNR | SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            "| "
            f"{row['method']} | {row['iterations']} | {row['mse']} | {row['mae']} | {row['psnr']} | {row['ssim']} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RIM.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--generator-checkpoint",type=Path,
                        default=Path("results/generator_wgangp_mnist32_e500_bs128_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt"))
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("results/encoder_spi_mnist32_cr_1e-2_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"))
    parser.add_argument("--rim-checkpoint", type=Path, 
                        default=Path("results/rim/rim_spi_mnist32_gp_init_learned_e100_bs128_lr_1e-3_steps10_hc32_cr_1e-2/rim.pt"))
    parser.add_argument("--save-root", type=Path, default=Path("results"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--task", type=str, default="spi", choices=["spi", "SR"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cr", type=float, default=0.01)
    parser.add_argument("--scale", type=int, default=-1)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument("--use-generator-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rim-init-mode", type=str, default="learned",
                        choices=["learned", "random_latent", "backprojection"])
    parser.add_argument("--random-init-std", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-visual", type=int, default=8)
    args = add_wandb_args(parser, default_job_type="eval_rim").parse_args()

    init_label = args.rim_init_mode if args.use_generator_prior else "observation"
    name_parts: list[object] = [
        "eval",
        "rim",
        args.task,
        "mnist32",
        "gp" if args.use_generator_prior else "nogp",
        f"init_{init_label}",
        args.split,
        f"bs{args.batch_size}",
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
    if init_label == "random_latent":
        name_parts.append(f"rinit_{format_float_token(args.random_init_std)}")
    if args.max_batches is not None:
        name_parts.append(f"mb{args.max_batches}")
    experiment_name = join_name_parts(*name_parts)
    args.output_dir = args.save_root / experiment_name
    if args.wandb_run_name is None:
        args.wandb_run_name = experiment_name

    set_seed(args.seed)
    device = resolve_device(args.device)
    wandb_logger = init_wandb_run(
        args,
        config=namespace_to_config(args),
        tags=["eval", "rim", args.task, "gp" if args.use_generator_prior else "nogp", args.rim_init_mode, experiment_name],
    )

    try:
        if args.use_generator_prior and args.rim_init_mode == "learned":
            encoder = load_encoder_checkpoint(args.encoder_checkpoint, device=device)
            freeze_module(encoder)
        else:
            encoder = torch.nn.Identity().to(device)
        if args.use_generator_prior:
            generator = load_generator_checkpoint(args.generator_checkpoint, device=device)
            freeze_module(generator)
        else:
            generator = torch.nn.Identity().to(device)
        rim = load_rim_checkpoint(args.rim_checkpoint, device=device)
        freeze_module(rim)

        loaders = get_mnist_dataloaders(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            download=args.download,
        )
        loader = loaders[args.split]

        if not args.use_generator_prior and args.rim_init_mode == "random_latent":
            raise ValueError("random_latent initialization requires use_generator_prior=true.")
        if args.task == "spi":
            operator = LinearSensingOperator(cr=args.cr).to(device)
        else:
            operator = SR(s=args.scale).to(device)

        set_seed(args.seed)
        rim_metrics, rim_metrics_curve, first_problem, first_history, first_targets, num_examples = evaluate_proposed_rim(
            rim=rim,
            encoder=encoder,
            generator=generator,
            loader=loader,
            device=device,
            task=args.task,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            max_batches=args.max_batches,
            use_generator_prior=args.use_generator_prior,
            init_mode=args.rim_init_mode,
            random_init_std=args.random_init_std,
        )
        metrics_csv_path = args.output_dir / f"{args.task}_{args.split}_rim_metrics.csv"
        metrics_md_path = args.output_dir / f"{args.task}_{args.split}_rim_metrics.md"
        curves_csv_path = args.output_dir / f"{args.task}_{args.split}_rim_psnr_curves.csv"
        iteration_metrics_csv_path = args.output_dir / f"{args.task}_{args.split}_rim_iteration_metrics.csv"
        plot_path = args.output_dir / f"{args.task}_{args.split}_rim_psnr.png"
        visual_svg_path = args.output_dir / f"{args.task}_{args.split}_rim_visual_triplets.svg"

        results = [
            {
                "method": "RIM",
                "split": args.split,
                "task": args.task,
                "num_examples": num_examples,
                "iterations": rim.steps,
                "mse": f"{rim_metrics.mse:.6f}",
                "mae": f"{rim_metrics.mae:.6f}",
                "psnr": f"{rim_metrics.psnr:.4f}",
                "ssim": f"{rim_metrics.ssim:.4f}",
            }
        ]
        psnr_curves = {"RIM": torch.tensor([metrics.psnr for metrics in rim_metrics_curve], dtype=torch.float32)}
        metrics_curves = {"RIM": rim_metrics_curve}

        write_aggregate_metrics_csv(path=metrics_csv_path, split=args.split, task=args.task, results=results)
        write_markdown_table(path=metrics_md_path, split=args.split, task=args.task, results=results)
        write_psnr_curves_csv(path=curves_csv_path, task=args.task, psnr_curves=psnr_curves)
        write_iteration_metrics_curves_csv(
            path=iteration_metrics_csv_path,
            task=args.task,
            metrics_curves=metrics_curves,
        )
        save_combined_psnr_plot(
            save_path=plot_path,
            task=args.task,
            psnr_curves=psnr_curves,
            title=f"RIM Evaluation on {args.task}",
        )
        save_reconstruction_triplets_svg(
            save_path=visual_svg_path,
            backprojection=first_problem.observation,
            estimate=first_history[:, -1],
            target=first_targets,
            num_images=args.num_visual,
            title=f"RIM {args.task}/{args.split}",
        )

        prefix = f"rim/{args.task}/{args.split}/evaluation"
        wandb_logger.summary(
            {
                "device": str(device),
                f"{prefix}/num_examples": num_examples,
                f"{prefix}/cr": args.cr if args.task == "spi" else None,
                f"{prefix}/scale": args.scale if args.task == "SR" else None,
                f"{prefix}/use_generator_prior": args.use_generator_prior,
                f"{prefix}/init_mode": args.rim_init_mode,
                f"{prefix}/random_init_std": args.random_init_std,
                f"{prefix}/iterations": rim.steps,
                f"{prefix}/mse": rim_metrics.mse,
                f"{prefix}/mae": rim_metrics.mae,
                f"{prefix}/psnr": rim_metrics.psnr,
                f"{prefix}/ssim": rim_metrics.ssim,
                "paths/rim_metrics_csv": str(metrics_csv_path),
                "paths/rim_metrics_md": str(metrics_md_path),
                "paths/rim_curves_csv": str(curves_csv_path),
                "paths/rim_iteration_metrics_csv": str(iteration_metrics_csv_path),
                "paths/rim_psnr_plot": str(plot_path),
                "paths/rim_visual_svg": str(visual_svg_path),
            }
        )
        wandb_logger.log_table(f"{prefix}/metrics_table", metrics_csv_path)

        print(f"split={args.split}")
        print(f"task={args.task}")
        print(
            f"RIM: mse={rim_metrics.mse:.6f} mae={rim_metrics.mae:.6f} "
            f"psnr={rim_metrics.psnr:.4f} ssim={rim_metrics.ssim:.4f}"
        )
        print(f"metrics_csv_path={metrics_csv_path}")
        print(f"metrics_md_path={metrics_md_path}")
        print(f"curves_csv_path={curves_csv_path}")
        print(f"iteration_metrics_csv_path={iteration_metrics_csv_path}")
        print(f"plot_path={plot_path}")
        print(f"visual_svg_path={visual_svg_path}")
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
