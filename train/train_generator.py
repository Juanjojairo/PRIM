"""Train the MNIST GAN generator used as the EADMM generative prior."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import autograd
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from datasets.mnist import get_mnist_dataloaders, to_display_mnist
from models.generator import (
    GeneratorMetrics,
    MNISTDiscriminator,
    MNISTGenerator,
    load_discriminator_checkpoint,
    load_generator_checkpoint,
    resolve_device,
    save_generator_checkpoint,
)
from utils.seed import set_seed
from utils.experiments import format_float_token, join_name_parts
from utils.wandb import add_wandb_args, init_wandb_run, namespace_to_config


def train_gan_epoch(
    generator: MNISTGenerator,
    discriminator: MNISTDiscriminator,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_penalty_weight: float,
    critic_iterations: int,
) -> tuple[float, float, float, float]:
    """Run one WGAN-GP training epoch."""
    generator.train()
    discriminator.train()

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_wasserstein = 0.0
    total_gp = 0.0
    generator_updates = 0
    total_examples = 0

    for batch_index, (images, _) in enumerate(tqdm(loader, desc="train-wgan-gp", leave=False), start=1):
        images = images.to(device)
        batch_size = images.shape[0]

        discriminator_optimizer.zero_grad(set_to_none=True)
        latent = torch.randn(batch_size, generator.latent_dim, device=device)
        fake_images = generator(latent)
        real_scores = discriminator(images)
        fake_scores = discriminator(fake_images.detach())
        wasserstein_distance = real_scores.mean() - fake_scores.mean()
        gradient_penalty = compute_gradient_penalty(
            discriminator=discriminator,
            real_images=images,
            fake_images=fake_images.detach(),
            device=device,
        )
        discriminator_loss = -wasserstein_distance + gradient_penalty_weight * gradient_penalty
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_loss_value = 0.0
        if batch_index % critic_iterations == 0:
            generator_optimizer.zero_grad(set_to_none=True)
            latent = torch.randn(batch_size, generator.latent_dim, device=device)
            fake_images = generator(latent)
            generator_loss = -discriminator(fake_images).mean()
            generator_loss.backward()
            generator_optimizer.step()
            generator_loss_value = generator_loss.item()
            total_g_loss += generator_loss_value * batch_size
            generator_updates += batch_size

        total_d_loss += discriminator_loss.item() * batch_size
        total_wasserstein += wasserstein_distance.item() * batch_size
        total_gp += gradient_penalty.item() * batch_size
        total_examples += batch_size

    return (
        total_g_loss / max(generator_updates, 1),
        total_d_loss / max(total_examples, 1),
        total_wasserstein / max(total_examples, 1),
        total_gp / max(total_examples, 1),
    )


def evaluate_gan(
    generator: MNISTGenerator,
    discriminator: MNISTDiscriminator,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    gradient_penalty_weight: float,
) -> tuple[float, float, float, float]:
    """Evaluate WGAN-GP losses on a held-out split."""
    generator.eval()
    discriminator.eval()

    total_g_loss = 0.0
    total_d_loss = 0.0
    total_wasserstein = 0.0
    total_gp = 0.0
    total_examples = 0

    for images, _ in tqdm(loader, desc="eval-wgan-gp", leave=False):
        images = images.to(device)
        batch_size = images.shape[0]
        latent = torch.randn(batch_size, generator.latent_dim, device=device)
        with torch.enable_grad():
            fake_images = generator(latent)
            real_scores = discriminator(images)
            fake_scores = discriminator(fake_images)
            wasserstein_distance = real_scores.mean() - fake_scores.mean()
            gradient_penalty = compute_gradient_penalty(
                discriminator=discriminator,
                real_images=images,
                fake_images=fake_images,
                device=device,
            )
            discriminator_loss = -wasserstein_distance + gradient_penalty_weight * gradient_penalty
            generator_loss = -fake_scores.mean()

        total_g_loss += generator_loss.item() * batch_size
        total_d_loss += discriminator_loss.item() * batch_size
        total_wasserstein += wasserstein_distance.item() * batch_size
        total_gp += gradient_penalty.item() * batch_size
        total_examples += batch_size

    return (
        total_g_loss / max(total_examples, 1),
        total_d_loss / max(total_examples, 1),
        total_wasserstein / max(total_examples, 1),
        total_gp / max(total_examples, 1),
    )


def compute_gradient_penalty(
    discriminator: MNISTDiscriminator,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute the WGAN-GP gradient penalty."""
    batch_size = real_images.shape[0]
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, dtype=real_images.dtype)
    interpolated = epsilon * real_images + (1.0 - epsilon) * fake_images
    interpolated.requires_grad_(True)

    interpolated_scores = discriminator(interpolated)
    gradients = autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()


@torch.no_grad()
def export_real_fake_grid(
    generator: MNISTGenerator,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    save_path: str | Path,
    fixed_latent: torch.Tensor | None = None,
    true_rows: int = 2,
    fake_rows: int = 8,
    cols: int = 10,
) -> None:
    """Export a grid mixing real and generated digits, following the original script."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    real_batches = []
    required = true_rows * cols
    for images, _ in loader:
        real_batches.append(images)
        if sum(batch.shape[0] for batch in real_batches) >= required:
            break

    real_images = torch.cat(real_batches, dim=0)[:required]
    latent = fixed_latent
    if latent is None:
        latent = torch.randn(fake_rows * cols, generator.latent_dim, device=device)
    fake_images = generator(latent).cpu()
    grid = torch.cat([real_images, fake_images], dim=0)
    save_image(
        to_display_mnist(grid),
        save_path,
        nrow=cols,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MNIST generator with WGAN-GP.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--save-root", type=Path, default=Path("results"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--generator-lr", type=float, default=1e-4)
    parser.add_argument("--discriminator-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--activation", type=str, default="elu", choices=["relu", "elu"])
    parser.add_argument("--critic-iterations", type=int, default=5)
    parser.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    args = add_wandb_args(parser, default_job_type="train_generator").parse_args()

    set_seed(args.seed)
    experiment_name = join_name_parts(
        "generator",
        "wgangp",
        "mnist32",
        f"e{args.epochs}",
        f"bs{args.batch_size}",
        f"glr_{format_float_token(args.generator_lr)}",
        f"dlr_{format_float_token(args.discriminator_lr)}",
        f"z{args.latent_dim}",
        f"ch{args.base_channels}",
        f"gp{format_float_token(args.gradient_penalty_weight)}",
        f"crit{args.critic_iterations}",
        args.activation,
    )
    args.output_dir = args.save_root / experiment_name
    args.checkpoint = args.output_dir / "generator.pt"
    if args.wandb_run_name is None:
        args.wandb_run_name = experiment_name

    device = resolve_device(args.device)
    wandb_logger = init_wandb_run(
        args,
        config=namespace_to_config(args),
        tags=["train", "generator", "gan", "mnist"],
    )

    try:
        loaders = get_mnist_dataloaders(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            normalize=False,
            download=args.download,
        )

        generator = MNISTGenerator(args.latent_dim, args.base_channels, args.activation).to(device)
        discriminator = MNISTDiscriminator(args.base_channels).to(device)
        generator_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=args.generator_lr,
            betas=(0.0, 0.9),
            weight_decay=args.weight_decay,
        )
        discriminator_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.discriminator_lr,
            betas=(0.0, 0.9),
            weight_decay=args.weight_decay,
        )
        fixed_latent = torch.randn(80, generator.latent_dim, device=device)

        best_val_generator_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_g_loss, train_d_loss, train_w_distance, train_gp = train_gan_epoch(
                generator=generator,
                discriminator=discriminator,
                loader=loaders["train"],
                generator_optimizer=generator_optimizer,
                discriminator_optimizer=discriminator_optimizer,
                device=device,
                gradient_penalty_weight=args.gradient_penalty_weight,
                critic_iterations=args.critic_iterations,
            )
            val_g_loss, val_d_loss, val_w_distance, val_gp = evaluate_gan(
                generator=generator,
                discriminator=discriminator,
                loader=loaders["val"],
                device=device,
                gradient_penalty_weight=args.gradient_penalty_weight,
            )
            metrics = GeneratorMetrics(
                train_generator_loss=train_g_loss,
                train_discriminator_loss=train_d_loss,
                val_generator_loss=val_g_loss,
                val_discriminator_loss=val_d_loss,
            )

            print(
                f"epoch={epoch} "
                f"train_g_loss={train_g_loss:.6f} "
                f"train_d_loss={train_d_loss:.6f} "
                f"train_w_distance={train_w_distance:.6f} "
                f"train_gp={train_gp:.6f} "
                f"val_g_loss={val_g_loss:.6f} "
                f"val_d_loss={val_d_loss:.6f} "
                f"val_w_distance={val_w_distance:.6f} "
                f"val_gp={val_gp:.6f}"
            )

            sample_path = args.output_dir / f"samples_epoch_{epoch:03d}.png"
            export_real_fake_grid(
                generator=generator,
                loader=loaders["test"],
                device=device,
                save_path=sample_path,
                fixed_latent=fixed_latent,
            )

            is_best = val_g_loss <= best_val_generator_loss
            if is_best:
                best_val_generator_loss = val_g_loss
                save_generator_checkpoint(
                    args.checkpoint,
                    generator=generator,
                    discriminator=discriminator,
                    metrics=metrics,
                )

            wandb_logger.log(
                {
                    "epoch": epoch,
                    "generator/train_generator_loss": train_g_loss,
                    "generator/train_critic_loss": train_d_loss,
                    "generator/train_wasserstein_distance": train_w_distance,
                    "generator/train_gradient_penalty": train_gp,
                    "generator/val_generator_loss": val_g_loss,
                    "generator/val_critic_loss": val_d_loss,
                    "generator/val_wasserstein_distance": val_w_distance,
                    "generator/val_gradient_penalty": val_gp,
                    "generator/best_val_generator_loss": best_val_generator_loss,
                    "generator/is_best_checkpoint": float(is_best),
                },
                step=epoch,
            )
            wandb_logger.log_image(
                "generator/samples",
                sample_path,
                step=epoch,
                caption=f"Real/fake MNIST grid at epoch {epoch}",
            )

        best_generator = load_generator_checkpoint(args.checkpoint, device=device)
        best_discriminator = load_discriminator_checkpoint(args.checkpoint, device=device)
        test_g_loss, test_d_loss, test_w_distance, test_gp = evaluate_gan(
            generator=best_generator,
            discriminator=best_discriminator,
            loader=loaders["test"],
            device=device,
            gradient_penalty_weight=args.gradient_penalty_weight,
        )
        wandb_logger.summary(
            {
                "device": str(device),
                "generator/best_val_generator_loss": best_val_generator_loss,
                "generator/test_generator_loss": test_g_loss,
                "generator/test_critic_loss": test_d_loss,
                "generator/test_wasserstein_distance": test_w_distance,
                "generator/test_gradient_penalty": test_gp,
                "paths/checkpoint": str(args.checkpoint),
                "paths/output_dir": str(args.output_dir),
            }
        )
        wandb_logger.log_artifact(
            args.checkpoint,
            artifact_name="generator-checkpoint",
            artifact_type="model",
            aliases=["best"],
        )

        print(f"best_val_generator_loss={best_val_generator_loss:.6f}")
        print(f"test_generator_loss={test_g_loss:.6f}")
        print(f"test_critic_loss={test_d_loss:.6f}")
        print(f"test_wasserstein_distance={test_w_distance:.6f}")
        print(f"test_gradient_penalty={test_gp:.6f}")
        print(f"checkpoint={args.checkpoint}")
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
