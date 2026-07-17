"""Train the image-space RIM with configurable generator initialization."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.mnist import get_mnist_dataloaders, to_display_mnist
from models.encoder import load_encoder_checkpoint
from models.generator import load_generator_checkpoint, resolve_device
from models.rim import (
    ImageSpaceRIM,
    RIMCheckpointMetrics,
    load_rim_checkpoint,
    save_rim_checkpoint,
)
from ops.forward_models import LinearSensingOperator
from ops.SR import SR
from ops.metrics import ReconstructionMetrics, compute_metrics_curve
from utils.observations import build_problem_batch, freeze_module
from utils.experiments import format_float_token, join_name_parts
from utils.seed import set_seed
from utils.visualization import save_labeled_mnist_grid
from utils.wandb import add_wandb_args, init_wandb_run, namespace_to_config


def parse_step_weights(
    steps: int,
    step_weights: str | None = None,
) -> torch.Tensor:
    """Parse step weights for the multistep reconstruction loss."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")

    if step_weights is None:
        return torch.ones(steps, dtype=torch.float32) / steps

    values = [float(item.strip()) for item in step_weights.split(",") if item.strip()]
    if len(values) != steps:
        raise ValueError(
            f"Expected {steps} step weights, got {len(values)} from {step_weights!r}."
        )

    weights = torch.tensor(values, dtype=torch.float32)
    if torch.any(weights < 0):
        raise ValueError("step weights must be non-negative.")
    if torch.sum(weights).item() <= 0:
        raise ValueError("step weights must sum to a positive value.")
    return weights / torch.sum(weights)


def compute_multistep_loss(
    history: torch.Tensor,
    target: torch.Tensor,
    step_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute a weighted multistep MSE loss over the RIM trajectory."""
    if history.shape[1] - 1 != step_weights.numel():
        raise ValueError(
            f"history has {history.shape[1] - 1} updates but received "
            f"{step_weights.numel()} step weights."
        )

    losses = []
    for index in range(step_weights.numel()):
        reconstruction = history[:, index + 1]
        losses.append(step_weights[index] * F.mse_loss(reconstruction, target))
    return torch.stack(losses).sum()


def write_iteration_metrics_csv(
    path: str | Path,
    split: str,
    task: str,
    metrics_curve: list[ReconstructionMetrics],
) -> None:
    """Persist average reconstruction metrics for every RIM iteration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "task", "iteration", "mse", "mae", "psnr", "ssim"],
        )
        writer.writeheader()
        for iteration, metrics in enumerate(metrics_curve):
            writer.writerow(
                {
                    "split": split,
                    "task": task,
                    "iteration": iteration,
                    "mse": f"{metrics.mse:.6f}",
                    "mae": f"{metrics.mae:.6f}",
                    "psnr": f"{metrics.psnr:.4f}",
                    "ssim": f"{metrics.ssim:.4f}",
                }
            )


def build_rim_initialization(
    use_generator_prior: bool,
    init_mode: str,
    generator: torch.nn.Module,
    encoder: torch.nn.Module,
    observation: torch.Tensor,
    task: str,
    random_init_std: float = 1.0,
) -> torch.Tensor:
    """Build the RIM initialization from either a learned or random latent code."""
    if not use_generator_prior:
        return observation
    if init_mode == "learned":
        return generator(encoder(observation))
    if init_mode == "random_latent":
        if not hasattr(generator, "latent_dim"):
            raise ValueError("generator must expose latent_dim for random latent initialization.")
        latent = random_init_std * torch.randn(
            observation.shape[0],
            int(generator.latent_dim),
            device=observation.device,
            dtype=observation.dtype,
        )
        return generator(latent)
    if init_mode == "backprojection":
        return observation
    raise ValueError(f"Unsupported rim init mode {init_mode!r}.")


def train_rim_epoch(
    rim: ImageSpaceRIM,
    encoder: torch.nn.Module,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    step_weights: torch.Tensor,
    operator: LinearSensingOperator | SR | None = None,
    measurement_noise_std: float = 0.0,
    use_generator_prior: bool = True,
    init_mode: str = "learned",
    random_init_std: float = 1.0,
    grad_clip: float | None = None,
    max_batches: int | None = None,
) -> tuple[float, ReconstructionMetrics]:
    """Train the RIM for one epoch."""
    rim.train()
    encoder.eval()
    generator.eval()

    total_loss = 0.0
    total_examples = 0
    train_metric_sums = {
        metric_name: 0.0
        for metric_name in ("mse", "mae", "psnr", "ssim")
    }

    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for images, labels in tqdm(batches, desc="train-rim", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        problem = build_problem_batch(
            task=task,
            images=images,
            operator=operator,
            measurement_noise_std=measurement_noise_std,
        )

        with torch.no_grad():
            x0 = build_rim_initialization(
                use_generator_prior=use_generator_prior,
                init_mode=init_mode,
                generator=generator,
                encoder=encoder,
                observation=problem.observation,
                task=task,
                random_init_std=random_init_std,
            )

        optimizer.zero_grad(set_to_none=True)
        history = rim(
            y=problem.fidelity_target,
            operator=operator,
            x0=x0,
        )
        loss = compute_multistep_loss(
            history=history,
            target=images,
            step_weights=step_weights.to(history.device),
        )
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(rim.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        final_metrics = compute_metrics_curve(history.detach().cpu(), images.detach().cpu())[-1]
        train_metric_sums["mse"] += final_metrics.mse * batch_size
        train_metric_sums["mae"] += final_metrics.mae * batch_size
        train_metric_sums["psnr"] += final_metrics.psnr * batch_size
        train_metric_sums["ssim"] += final_metrics.ssim * batch_size

    train_metrics = ReconstructionMetrics(
        mse=train_metric_sums["mse"] / max(total_examples, 1),
        mae=train_metric_sums["mae"] / max(total_examples, 1),
        psnr=train_metric_sums["psnr"] / max(total_examples, 1),
        ssim=train_metric_sums["ssim"] / max(total_examples, 1),
    )
    return total_loss / max(total_examples, 1), train_metrics


def evaluate_rim(
    rim: ImageSpaceRIM,
    encoder: torch.nn.Module,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    task: str,
    step_weights: torch.Tensor,
    operator: LinearSensingOperator | SR | None = None,
    measurement_noise_std: float = 0.0,
    use_generator_prior: bool = True,
    init_mode: str = "learned",
    random_init_std: float = 1.0,
    max_batches: int | None = None,
) -> tuple[float, list[ReconstructionMetrics]]:
    """Evaluate validation loss and per-iteration metrics for the RIM."""
    rim.eval()
    encoder.eval()
    generator.eval()

    total_loss = 0.0
    total_examples = 0
    metrics_curve_sum = {
        metric_name: torch.zeros(rim.steps + 1, dtype=torch.float64)
        for metric_name in ("mse", "mae", "psnr", "ssim")
    }

    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for images, labels in tqdm(batches, desc="eval-rim", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        problem = build_problem_batch(
            task=task,
            images=images,
            operator=operator,
            measurement_noise_std=measurement_noise_std,
        )

        with torch.no_grad():
            x0 = build_rim_initialization(
                use_generator_prior=use_generator_prior,
                init_mode=init_mode,
                generator=generator,
                encoder=encoder,
                observation=problem.observation,
                task=task,
                random_init_std=random_init_std,
            )
            history = rim(
                y=problem.fidelity_target,
                operator=operator,
                x0=x0,
            )
            loss = compute_multistep_loss(
                history=history,
                target=images,
                step_weights=step_weights.to(history.device),
            )

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        for iteration, metrics in enumerate(compute_metrics_curve(history.cpu(), images.cpu())):
            metrics_curve_sum["mse"][iteration] += metrics.mse * batch_size
            metrics_curve_sum["mae"][iteration] += metrics.mae * batch_size
            metrics_curve_sum["psnr"][iteration] += metrics.psnr * batch_size
            metrics_curve_sum["ssim"][iteration] += metrics.ssim * batch_size

    mean_metrics_curve = [
        ReconstructionMetrics(
            mse=(metrics_curve_sum["mse"][iteration] / max(total_examples, 1)).item(),
            mae=(metrics_curve_sum["mae"][iteration] / max(total_examples, 1)).item(),
            psnr=(metrics_curve_sum["psnr"][iteration] / max(total_examples, 1)).item(),
            ssim=(metrics_curve_sum["ssim"][iteration] / max(total_examples, 1)).item(),
        )
        for iteration in range(rim.steps + 1)
    ]
    return total_loss / max(total_examples, 1), mean_metrics_curve


def export_rim_triplets(
    rim: ImageSpaceRIM,
    encoder: torch.nn.Module,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    task: str,
    save_path: str | Path,
    operator: LinearSensingOperator | SR | None = None,
    measurement_noise_std: float = 0.0,
    use_generator_prior: bool = True,
    init_mode: str = "learned",
    random_init_std: float = 1.0,
    num_images: int = 8,
) -> None:
    """Save ground truth, observation, x0 and final RIM reconstruction."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)
    problem = build_problem_batch(
        task=task,
        images=images,
        operator=operator,
        measurement_noise_std=measurement_noise_std,
    )
    with torch.no_grad():
        x0 = build_rim_initialization(
            use_generator_prior=use_generator_prior,
            init_mode=init_mode,
            generator=generator,
            encoder=encoder,
            observation=problem.observation,
            task=task,
            random_init_std=random_init_std,
        )
        history = rim(
            y=problem.fidelity_target,
            operator=operator,
            x0=x0,
        )
        final = history[:, -1]
    save_labeled_mnist_grid(
        rows=[
            to_display_mnist(images.cpu()),
            to_display_mnist(problem.observation.cpu().clamp(0.0, 1.0)),
            to_display_mnist(x0.cpu()),
            to_display_mnist(final.cpu()),
        ],
        row_labels=["GT", "Obs", "Init", "Final"],
        save_path=save_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the image-space RIM.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--generator-checkpoint", type=Path,
                        default=Path("results/generator_wgangp_mnist32_e500_bs128"
                                     "_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt"))
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("results/encoder_SR_mnist32_sr_4_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"))
    parser.add_argument("--save-root", type=Path, default=Path("results"))
    parser.add_argument("--task", type=str, default="SR", choices=["spi", "SR"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-scale", type=float, default=0.1)
    parser.add_argument("--step-weights", type=str, default=None)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cr", type=float, default=-1)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument("--use-generator-prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rim-init-mode", type=str, default="learned",
                        choices=["learned", "random_latent", "backprojection"])
    parser.add_argument("--random-init-std", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = add_wandb_args(parser, default_job_type="train_rim").parse_args()

    set_seed(args.seed)
    init_label = args.rim_init_mode if args.use_generator_prior else "observation"
    experiment_name = join_name_parts(
        "rim",
        args.task,
        "mnist32",
        "gp" if args.use_generator_prior else "nogp",
        f"init_{init_label}",
        f"e{args.epochs}",
        f"bs{args.batch_size}",
        f"lr_{format_float_token(args.lr)}",
        f"steps{args.steps}",
        f"hc{args.hidden_channels}",
        (
            f"cr_{format_float_token(args.cr)}"
            if args.task == "spi"
            else f"sr_{args.scale}"
        ),
    )
    if init_label == "random_latent":
        experiment_name = join_name_parts(
            experiment_name,
            f"rinit_{format_float_token(args.random_init_std)}",
    )
    args.output_dir = args.save_root / experiment_name
    args.rim_checkpoint = args.output_dir / "rim.pt"
    args.last_rim_checkpoint = args.output_dir / "rim_last.pt"
    if args.wandb_run_name is None:
        args.wandb_run_name = experiment_name

    device = resolve_device(args.device)
    wandb_logger = init_wandb_run(
        args,
        config=namespace_to_config(args),
        tags=["train", "rim", args.task, "gp" if args.use_generator_prior else "nogp", args.rim_init_mode],
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

        rim = ImageSpaceRIM(
            hidden_channels=args.hidden_channels,
            steps=args.steps,
            step_scale=args.step_scale,
        ).to(device)
        optimizer = torch.optim.Adam(
            rim.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        step_weights = parse_step_weights(args.steps, args.step_weights)

        loaders = get_mnist_dataloaders(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            download=args.download,
        )

        operator = None
        if not args.use_generator_prior and args.rim_init_mode == "random_latent":
            raise ValueError("random_latent initialization requires use_generator_prior=true.")
        if args.task == "spi":
            operator = LinearSensingOperator(cr=args.cr).to(device)
        else:
            operator = SR(s=args.scale).to(device)

        best_val_psnr = float("-inf")
        best_val_loss = float("inf")
        best_val_metrics: ReconstructionMetrics | None = None
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_rim_epoch(
                rim=rim,
                encoder=encoder,
                generator=generator,
                loader=loaders["train"],
                optimizer=optimizer,
                device=device,
                task=args.task,
                step_weights=step_weights,
                operator=operator,
                measurement_noise_std=args.measurement_noise_std,
                use_generator_prior=args.use_generator_prior,
                init_mode=args.rim_init_mode,
                random_init_std=args.random_init_std,
                grad_clip=args.grad_clip,
                max_batches=args.max_train_batches,
            )
            val_loss, val_metrics_curve = evaluate_rim(
                rim=rim,
                encoder=encoder,
                generator=generator,
                loader=loaders["val"],
                device=device,
                task=args.task,
                step_weights=step_weights,
                operator=operator,
                measurement_noise_std=args.measurement_noise_std,
                use_generator_prior=args.use_generator_prior,
                init_mode=args.rim_init_mode,
                random_init_std=args.random_init_std,
                max_batches=args.max_val_batches,
            )
            val_psnr_curve = [round(metrics.psnr, 4) for metrics in val_metrics_curve]

            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.6f} "
                f"train_psnr={train_metrics.psnr:.4f} "
                f"train_ssim={train_metrics.ssim:.4f} "
                f"val_loss={val_loss:.6f} "
                f"val_psnr_curve={val_psnr_curve}"
            )

            triplet_path = args.output_dir / f"{args.task}_triplets_epoch_{epoch:03d}.png"
            val_iteration_metrics_path = args.output_dir / f"{args.task}_val_iteration_metrics_epoch_{epoch:03d}.csv"
            export_rim_triplets(
                rim=rim,
                encoder=encoder,
                generator=generator,
                loader=loaders["val"],
                device=device,
                task=args.task,
                save_path=triplet_path,
                operator=operator,
                measurement_noise_std=args.measurement_noise_std,
                use_generator_prior=args.use_generator_prior,
                init_mode=args.rim_init_mode,
                random_init_std=args.random_init_std,
            )
            write_iteration_metrics_csv(
                path=val_iteration_metrics_path,
                split="val",
                task=args.task,
                metrics_curve=val_metrics_curve,
            )

            final_val_psnr = val_metrics_curve[-1].psnr
            final_val_metrics = val_metrics_curve[-1]
            is_best = final_val_psnr >= best_val_psnr
            if is_best:
                best_val_psnr = final_val_psnr
                best_val_loss = val_loss
                best_val_metrics = final_val_metrics
                save_rim_checkpoint(
                    path=args.rim_checkpoint,
                    model=rim,
                    metrics=RIMCheckpointMetrics(
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_psnr=final_val_psnr,
                    ),
                )

            wandb_logger.log(
                {
                    "epoch": epoch,
                    "rim/train_loss": train_loss,
                    "rim/train_mse": train_metrics.mse,
                    "rim/train_mae": train_metrics.mae,
                    "rim/train_psnr": train_metrics.psnr,
                    "rim/train_ssim": train_metrics.ssim,
                    "rim/val_loss": val_loss,
                    "rim/val_mse": final_val_metrics.mse,
                    "rim/val_mae": final_val_metrics.mae,
                    "rim/val_psnr": final_val_metrics.psnr,
                    "rim/val_ssim": final_val_metrics.ssim,
                    "rim/best_val_loss": best_val_loss,
                    "rim/best_val_mse": best_val_metrics.mse if best_val_metrics is not None else None,
                    "rim/best_val_mae": best_val_metrics.mae if best_val_metrics is not None else None,
                    "rim/best_val_psnr": best_val_psnr,
                    "rim/best_val_ssim": best_val_metrics.ssim if best_val_metrics is not None else None,
                    "rim/is_best_checkpoint": float(is_best),
                    "rim/cr": args.cr if args.task == "spi" else None,
                    "rim/scale": args.scale if args.task == "SR" else None,
                },
                step=epoch,
            )
            wandb_logger.log_image(
                "rim/val_triplets",
                triplet_path,
                step=epoch,
                caption=f"{args.task} RIM triplets at epoch {epoch}",
            )

        if best_val_metrics is None:
            raise RuntimeError("No best validation metrics were recorded during training.")

        save_rim_checkpoint(
            path=args.last_rim_checkpoint,
            model=rim,
            metrics=RIMCheckpointMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                val_psnr=final_val_psnr,
            ),
        )

        test_last_loss, test_last_metrics_curve = evaluate_rim(
            rim=rim,
            encoder=encoder,
            generator=generator,
            loader=loaders["test"],
            device=device,
            task=args.task,
            step_weights=step_weights,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            use_generator_prior=args.use_generator_prior,
            init_mode=args.rim_init_mode,
            random_init_std=args.random_init_std,
            max_batches=args.max_val_batches,
        )
        best_rim = load_rim_checkpoint(args.rim_checkpoint, device=device)
        test_best_loss, test_best_metrics_curve = evaluate_rim(
            rim=best_rim,
            encoder=encoder,
            generator=generator,
            loader=loaders["test"],
            device=device,
            task=args.task,
            step_weights=step_weights,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            use_generator_prior=args.use_generator_prior,
            init_mode=args.rim_init_mode,
            random_init_std=args.random_init_std,
            max_batches=args.max_val_batches,
        )
        test_last_psnr_curve = [round(metrics.psnr, 4) for metrics in test_last_metrics_curve]
        test_best_psnr_curve = [round(metrics.psnr, 4) for metrics in test_best_metrics_curve]
        test_iteration_metrics_path = args.output_dir / f"{args.task}_test_iteration_metrics.csv"
        write_iteration_metrics_csv(
            path=test_iteration_metrics_path,
            split="test",
            task=args.task,
            metrics_curve=test_best_metrics_curve,
        )
        final_test_last_metrics = test_last_metrics_curve[-1]
        final_test_best_metrics = test_best_metrics_curve[-1]
        wandb_logger.summary(
            {
                "device": str(device),
                "rim/best_val_loss": best_val_loss,
                "rim/best_val_mse": best_val_metrics.mse if best_val_metrics is not None else None,
                "rim/best_val_mae": best_val_metrics.mae if best_val_metrics is not None else None,
                "rim/best_val_psnr": best_val_psnr,
                "rim/best_val_ssim": best_val_metrics.ssim if best_val_metrics is not None else None,
                "rim/test_last_loss": test_last_loss,
                "rim/test_last_mse": final_test_last_metrics.mse,
                "rim/test_last_mae": final_test_last_metrics.mae,
                "rim/test_last_psnr": final_test_last_metrics.psnr,
                "rim/test_last_ssim": final_test_last_metrics.ssim,
                "rim/test_best_loss": test_best_loss,
                "rim/test_best_mse": final_test_best_metrics.mse,
                "rim/test_best_mae": final_test_best_metrics.mae,
                "rim/test_best_psnr": final_test_best_metrics.psnr,
                "rim/test_best_ssim": final_test_best_metrics.ssim,
                "rim/test_loss": test_best_loss,
                "rim/test_mse": final_test_best_metrics.mse,
                "rim/test_mae": final_test_best_metrics.mae,
                "rim/test_psnr": final_test_best_metrics.psnr,
                "rim/test_ssim": final_test_best_metrics.ssim,
                "rim/cr": args.cr if args.task == "spi" else None,
                "rim/scale": args.scale if args.task == "SR" else None,
                "rim/use_generator_prior": args.use_generator_prior,
                "rim/init_mode": args.rim_init_mode,
                "rim/random_init_std": args.random_init_std,
                "paths/rim_checkpoint": str(args.rim_checkpoint),
                "paths/last_rim_checkpoint": str(args.last_rim_checkpoint),
                "paths/output_dir": str(args.output_dir),
                "paths/test_iteration_metrics_csv": str(test_iteration_metrics_path),
            }
        )
        wandb_logger.log_artifact(
            args.rim_checkpoint,
            artifact_name=f"rim-{args.task}-checkpoint",
            artifact_type="model",
            aliases=["best"],
        )
        wandb_logger.log_artifact(
            args.last_rim_checkpoint,
            artifact_name=f"rim-{args.task}-last-checkpoint",
            artifact_type="model",
            aliases=["last"],
        )

        print(f"best_val_psnr={best_val_psnr:.4f}")
        print(f"test_last_loss={test_last_loss:.6f}")
        print(f"test_last_psnr_curve={test_last_psnr_curve}")
        print(f"test_best_loss={test_best_loss:.6f}")
        print(f"test_best_psnr_curve={test_best_psnr_curve}")
        print(f"test_iteration_metrics_path={test_iteration_metrics_path}")
        print(f"rim_checkpoint={args.rim_checkpoint}")
        print(f"last_rim_checkpoint={args.last_rim_checkpoint}")
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
