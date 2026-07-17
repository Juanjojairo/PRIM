"""Train the latent-initialization encoder against a frozen pretrained generator."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from datasets.mnist import get_mnist_dataloaders, to_display_mnist
from models.encoder import (
    EncoderMetrics,
    ObservationEncoder,
    load_encoder_checkpoint,
    save_encoder_checkpoint,
)
from models.generator import load_generator_checkpoint, resolve_device
from ops.forward_models import LinearSensingOperator
from ops.metrics import ReconstructionMetrics, compute_reconstruction_metrics
from utils.observations import build_observation_batch, freeze_module
from utils.experiments import format_float_token, join_name_parts
from utils.seed import set_seed
from utils.wandb import add_wandb_args, init_wandb_run, namespace_to_config
from ops.SR import SR

def train_encoder_epoch(
    encoder: ObservationEncoder,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task: str,
    operator: LinearSensingOperator | None = None,
    measurement_noise_std: float = 0.0,
    grad_clip: float | None = None,
    max_batches: int | None = None,
) -> tuple[float, ReconstructionMetrics]:
    """Train the encoder for one epoch with the generator frozen."""
    encoder.train()
    generator.eval()

    total_loss = 0.0
    total_examples = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for images, labels in tqdm(batches, desc="train-enc", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        observations = build_observation_batch(
            task=task,
            images=images,
            operator=operator,
            measurement_noise_std=measurement_noise_std,
        )

        optimizer.zero_grad(set_to_none=True)
        latent = encoder(observations)
        reconstructions = generator(latent)
        loss = F.mse_loss(reconstructions, images)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=grad_clip)

        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        predictions.append(reconstructions.detach().cpu())
        targets.append(images.detach().cpu())

    prediction_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    metrics = compute_reconstruction_metrics(prediction_tensor, target_tensor)
    return total_loss / max(total_examples, 1), metrics


def evaluate_encoder(
    encoder: ObservationEncoder,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    task: str,
    operator: LinearSensingOperator | None = None,
    measurement_noise_std: float = 0.0,
    max_batches: int | None = None,
) -> tuple[float, ReconstructionMetrics]:
    """Evaluate encoder reconstruction quality on a dataset split."""
    encoder.eval()
    generator.eval()

    total_loss = 0.0
    total_examples = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for images, labels in tqdm(batches, desc="eval-enc", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        observations = build_observation_batch(
            task=task,
            images=images,
            operator=operator,
            measurement_noise_std=measurement_noise_std,
        )

        with torch.no_grad():
            latent = encoder(observations)
            reconstructions = generator(latent)
            loss = F.mse_loss(reconstructions, images)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        predictions.append(reconstructions.cpu())
        targets.append(images.cpu())

    prediction_tensor = torch.cat(predictions, dim=0)
    target_tensor = torch.cat(targets, dim=0)
    metrics = compute_reconstruction_metrics(prediction_tensor, target_tensor)
    return total_loss / max(total_examples, 1), metrics


def export_encoder_triplets(
    encoder: ObservationEncoder,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    task: str,
    save_path: str | Path,
    operator: LinearSensingOperator | None = None,
    measurement_noise_std: float = 0.0,
    num_images: int = 8,
    scale: int = 2,
) -> None:
    """Save ground truth, observation, and reconstruction triplets."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].to(device)
    observations = build_observation_batch(
        task=task,
        images=images,
        operator=operator,
        measurement_noise_std=measurement_noise_std,
    )
    with torch.no_grad():
        reconstructions = generator(encoder(observations))

    if task == "spi":
        observation_display = observations.detach().cpu()
        mins = observation_display.amin(dim=(1, 2, 3), keepdim=True)
        maxs = observation_display.amax(dim=(1, 2, 3), keepdim=True)
        observation_display = (observation_display - mins) / (maxs - mins).clamp_min(1e-8)

    observation_display = observations.detach().cpu().clamp(0.0, 1.0)

    comparison = torch.cat(
        [images.cpu(), observation_display, reconstructions.cpu()],
        dim=0,
    )
    save_image(to_display_mnist(comparison), save_path, nrow=num_images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the latent initialization encoder.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--generator-checkpoint", type=Path,
                        default=Path("results/generator_wgangp_mnist32_e500_bs128"
                                     "_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt"))
    parser.add_argument("--save-root", type=Path, default=Path("results"))
    parser.add_argument("--task", type=str, default="SR", choices=["spi", "SR"])
    parser.add_argument("--scale", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cr", type=float, default=-1)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = add_wandb_args(parser, default_job_type="train_encoder").parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    generator = load_generator_checkpoint(args.generator_checkpoint, device=device)
    latent_dim = generator.latent_dim if args.latent_dim is None else args.latent_dim
    base_channels = generator.base_channels if args.base_channels is None else args.base_channels
    experiment_name = join_name_parts(
        "encoder",
        args.task,
        "mnist32",
        (
            f"cr_{format_float_token(args.cr)}"
            if args.task == "spi"
            else f"sr_{args.scale}"
        ),
        f"e{args.epochs}",
        f"bs{args.batch_size}",
        f"lr_{format_float_token(args.lr)}",
        f"z{latent_dim}",
        f"ch{base_channels}",
    )
    args.output_dir = args.save_root / experiment_name
    args.encoder_checkpoint = args.output_dir / "encoder.pt"
    args.last_encoder_checkpoint = args.output_dir / "encoder_last.pt"
    if args.wandb_run_name is None:
        args.wandb_run_name = experiment_name

    wandb_logger = init_wandb_run(
        args,
        config=namespace_to_config(args),
        tags=["train", "encoder", args.task],
    )

    try:
        freeze_module(generator)
        encoder = ObservationEncoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
        ).to(device)
        optimizer = torch.optim.Adam(
            encoder.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        loaders = get_mnist_dataloaders(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            download=args.download,
        )

        operator = None
        if args.task == "spi":
            operator = LinearSensingOperator(cr=args.cr).to(device)
        elif args.task == "SR":
            operator = SR(s=args.scale).to(device)

        best_val_psnr = float("-inf")
        best_val_loss = float("inf")
        best_val_metrics: ReconstructionMetrics | None = None
        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_encoder_epoch(
                encoder=encoder,
                generator=generator,
                loader=loaders["train"],
                optimizer=optimizer,
                device=device,
                task=args.task,
                operator=operator,
                measurement_noise_std=args.measurement_noise_std,
                grad_clip=args.grad_clip,
                max_batches=args.max_train_batches,
            )
            val_loss, val_metrics = evaluate_encoder(
                encoder=encoder,
                generator=generator,
                loader=loaders["val"],
                device=device,
                task=args.task,
                operator=operator,
                measurement_noise_std=args.measurement_noise_std,
                max_batches=args.max_val_batches,
            )

            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.6f} "
                f"train_psnr={train_metrics.psnr:.4f} "
                f"train_ssim={train_metrics.ssim:.4f} "
                f"val_loss={val_loss:.6f} "
                f"val_psnr={val_metrics.psnr:.4f} "
                f"val_ssim={val_metrics.ssim:.4f}"
            )

            triplet_path = args.output_dir / f"{args.task}_triplets_epoch_{epoch:03d}.png"
            export_encoder_triplets(
                encoder=encoder,
                generator=generator,
                loader=loaders["val"],
                device=device,
                task=args.task,
                save_path=triplet_path,
                operator=operator,
                measurement_noise_std=args.measurement_noise_std,
                scale=args.scale,
            )

            is_best = val_metrics.psnr >= best_val_psnr
            if is_best:
                best_val_psnr = val_metrics.psnr
                best_val_loss = val_loss
                best_val_metrics = val_metrics
                metrics = EncoderMetrics(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_psnr=val_metrics.psnr,
                    val_ssim=val_metrics.ssim,
                )
                save_encoder_checkpoint(args.encoder_checkpoint, encoder=encoder, metrics=metrics)

            wandb_logger.log(
                {
                    "epoch": epoch,
                    "encoder/train_loss": train_loss,
                    "encoder/train_mse": train_metrics.mse,
                    "encoder/train_mae": train_metrics.mae,
                    "encoder/train_psnr": train_metrics.psnr,
                    "encoder/train_ssim": train_metrics.ssim,
                    "encoder/val_loss": val_loss,
                    "encoder/val_mse": val_metrics.mse,
                    "encoder/val_mae": val_metrics.mae,
                    "encoder/val_psnr": val_metrics.psnr,
                    "encoder/val_ssim": val_metrics.ssim,
                    "encoder/best_val_loss": best_val_loss,
                    "encoder/best_val_mse": best_val_metrics.mse if best_val_metrics is not None else None,
                    "encoder/best_val_mae": best_val_metrics.mae if best_val_metrics is not None else None,
                    "encoder/best_val_psnr": best_val_psnr,
                    "encoder/best_val_ssim": best_val_metrics.ssim if best_val_metrics is not None else None,
                    "encoder/is_best_checkpoint": float(is_best),
                },
                step=epoch,
            )
            wandb_logger.log_image(
                "encoder/val_triplets",
                triplet_path,
                step=epoch,
                caption=f"{args.task} validation triplets at epoch {epoch}",
            )

        if best_val_metrics is None:
            raise RuntimeError("No best validation metrics were recorded during training.")

        save_encoder_checkpoint(
            args.last_encoder_checkpoint,
            encoder=encoder,
            metrics=EncoderMetrics(
                train_loss=train_loss,
                val_loss=val_loss,
                val_psnr=val_metrics.psnr,
                val_ssim=val_metrics.ssim,
            ),
        )

        test_last_loss, test_last_metrics = evaluate_encoder(
            encoder=encoder,
            generator=generator,
            loader=loaders["test"],
            device=device,
            task=args.task,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            max_batches=args.max_val_batches,
        )
        best_encoder = load_encoder_checkpoint(args.encoder_checkpoint, device=device)
        test_best_loss, test_best_metrics = evaluate_encoder(
            encoder=best_encoder,
            generator=generator,
            loader=loaders["test"],
            device=device,
            task=args.task,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            max_batches=args.max_val_batches,
        )
        wandb_logger.summary(
            {
                "device": str(device),
                "encoder/best_val_loss": best_val_loss,
                "encoder/best_val_mse": best_val_metrics.mse if best_val_metrics is not None else None,
                "encoder/best_val_mae": best_val_metrics.mae if best_val_metrics is not None else None,
                "encoder/best_val_psnr": best_val_psnr,
                "encoder/best_val_ssim": best_val_metrics.ssim if best_val_metrics is not None else None,
                "encoder/test_last_loss": test_last_loss,
                "encoder/test_last_mse": test_last_metrics.mse,
                "encoder/test_last_mae": test_last_metrics.mae,
                "encoder/test_last_psnr": test_last_metrics.psnr,
                "encoder/test_last_ssim": test_last_metrics.ssim,
                "encoder/test_best_loss": test_best_loss,
                "encoder/test_best_mse": test_best_metrics.mse,
                "encoder/test_best_mae": test_best_metrics.mae,
                "encoder/test_best_psnr": test_best_metrics.psnr,
                "encoder/test_best_ssim": test_best_metrics.ssim,
                "encoder/test_loss": test_best_loss,
                "encoder/test_mse": test_best_metrics.mse,
                "encoder/test_mae": test_best_metrics.mae,
                "encoder/test_psnr": test_best_metrics.psnr,
                "encoder/test_ssim": test_best_metrics.ssim,
                "paths/encoder_checkpoint": str(args.encoder_checkpoint),
                "paths/last_encoder_checkpoint": str(args.last_encoder_checkpoint),
                "paths/output_dir": str(args.output_dir),
            }
        )
        wandb_logger.log_artifact(
            args.encoder_checkpoint,
            artifact_name=f"encoder-{args.task}-checkpoint",
            artifact_type="model",
            aliases=["best"],
        )
        wandb_logger.log_artifact(
            args.last_encoder_checkpoint,
            artifact_name=f"encoder-{args.task}-last-checkpoint",
            artifact_type="model",
            aliases=["last"],
        )

        print(f"best_val_psnr={best_val_psnr:.4f}")
        print(f"test_last_loss={test_last_loss:.6f}")
        print(f"test_last_psnr={test_last_metrics.psnr:.4f}")
        print(f"test_last_ssim={test_last_metrics.ssim:.4f}")
        print(f"test_best_loss={test_best_loss:.6f}")
        print(f"test_best_psnr={test_best_metrics.psnr:.4f}")
        print(f"test_best_ssim={test_best_metrics.ssim:.4f}")
        print(f"encoder_checkpoint={args.encoder_checkpoint}")
        print(f"last_encoder_checkpoint={args.last_encoder_checkpoint}")
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
