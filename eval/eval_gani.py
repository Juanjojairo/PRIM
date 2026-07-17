"""Evaluate the encoder-plus-generator (GANI) baseline."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm.auto import tqdm

from datasets.mnist import get_mnist_dataloaders, to_display_mnist
from models.encoder import load_encoder_checkpoint
from models.generator import load_generator_checkpoint, resolve_device
from ops.forward_models import LinearSensingOperator
from ops.SR import SR
from ops.metrics import ReconstructionMetrics, compute_reconstruction_metrics
from utils.experiments import format_float_token, join_name_parts
from utils.observations import build_observation_batch, freeze_module
from utils.visualization import save_reconstruction_triplets_svg
from utils.wandb import add_wandb_args, init_wandb_run, namespace_to_config


def evaluate_gani(
    encoder: torch.nn.Module,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    task: str,
    operator: LinearSensingOperator | SR | None = None,
    measurement_noise_std: float = 0.0,
    max_batches: int | None = None,
) -> tuple[ReconstructionMetrics, int]:
    """Compute aggregate reconstruction metrics for the GANI baseline."""
    encoder.eval()
    generator.eval()

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    total_examples = 0

    batches = loader if max_batches is None else itertools.islice(loader, max_batches)
    for images, labels in tqdm(batches, desc="eval-gani", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        observations = build_observation_batch(
            task=task,
            images=images,
            operator=operator,
            measurement_noise_std=measurement_noise_std,
        )

        with torch.no_grad():
            reconstructions = generator(encoder(observations))

        predictions.append(reconstructions.cpu())
        targets.append(images.cpu())
        total_examples += images.shape[0]

    metrics = compute_reconstruction_metrics(
        prediction=torch.cat(predictions, dim=0),
        target=torch.cat(targets, dim=0),
    )
    return metrics, total_examples


def export_qualitative_grid(
    encoder: torch.nn.Module,
    generator: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    task: str,
    save_path: str | Path,
    operator: LinearSensingOperator | SR | None = None,
    measurement_noise_std: float = 0.0,
    num_images: int = 8,
) -> None:
    """Save ground-truth, observation, and GANI reconstruction triplets."""
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

    grid = torch.cat(
        [images.cpu(), observations.cpu().clamp(0.0, 1.0), reconstructions.cpu()],
        dim=0,
    )
    save_image(to_display_mnist(grid), save_path, nrow=num_images)


def write_metrics_csv(
    path: str | Path,
    split: str,
    task: str,
    metrics: ReconstructionMetrics,
    num_examples: int,
    cr: float | None = None,
    scale: int | None = None,
    measurement_noise_std: float = 0.0,
) -> None:
    """Persist aggregate metrics in CSV format."""
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
                "method": "GANI",
                "split": split,
                "task": task,
                "num_examples": num_examples,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the GANI baseline.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--generator-checkpoint", type=Path,
                        default=Path("results/generator_wgangp_mnist32_e500_bs128"
                                     "_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt"))
    parser.add_argument("--encoder-checkpoint", type=Path,
                        default=Path("results/encoder_spi_mnist32_cr_1e-1_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"))
    parser.add_argument("--save-root", type=Path, default=Path("results"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--task", type=str, default="spi", choices=["spi", "SR"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cr", type=float, default=0.1)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-qualitative", type=int, default=8)
    parser.add_argument("--num-visual", type=int, default=8)
    args = add_wandb_args(parser, default_job_type="eval_gani").parse_args()

    name_parts: list[object] = [
        "eval",
        "gani",
        args.task,
        "mnist32",
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
    if args.max_batches is not None:
        name_parts.append(f"mb{args.max_batches}")
    experiment_name = join_name_parts(*name_parts)
    args.output_dir = args.save_root / experiment_name
    if args.wandb_run_name is None:
        args.wandb_run_name = experiment_name

    device = resolve_device(args.device)
    wandb_logger = init_wandb_run(
        args,
        config=namespace_to_config(args),
        tags=["eval", "gani", args.task, experiment_name],
    )

    try:
        encoder = load_encoder_checkpoint(args.encoder_checkpoint, device=device)
        generator = load_generator_checkpoint(args.generator_checkpoint, device=device)
        freeze_module(encoder)
        freeze_module(generator)

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

        metrics, num_examples = evaluate_gani(
            encoder=encoder,
            generator=generator,
            loader=loader,
            device=device,
            task=args.task,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            max_batches=args.max_batches,
        )

        qualitative_path = args.output_dir / f"{args.task}_{args.split}_qualitative.png"
        visual_svg_path = args.output_dir / f"{args.task}_{args.split}_visual_triplets.svg"
        export_qualitative_grid(
            encoder=encoder,
            generator=generator,
            loader=loader,
            device=device,
            task=args.task,
            save_path=qualitative_path,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            num_images=args.num_qualitative,
        )
        visual_images, _ = next(iter(loader))
        visual_images = visual_images[: args.num_visual].to(device)
        visual_observation = build_observation_batch(
            task=args.task,
            images=visual_images,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
        )
        with torch.no_grad():
            visual_estimate = generator(encoder(visual_observation))
        save_reconstruction_triplets_svg(
            save_path=visual_svg_path,
            backprojection=visual_observation.cpu(),
            estimate=visual_estimate.cpu(),
            target=visual_images.cpu(),
            num_images=args.num_visual,
            title=f"GANI {args.task}/{args.split}",
        )

        csv_path = args.output_dir / f"{args.task}_{args.split}_metrics.csv"
        write_metrics_csv(
            path=csv_path,
            split=args.split,
            task=args.task,
            metrics=metrics,
            num_examples=num_examples,
            cr=args.cr if args.task == "spi" else None,
            scale=args.scale if args.task == "SR" else None,
            measurement_noise_std=args.measurement_noise_std,
        )

        prefix = f"gani/{args.task}/{args.split}"
        wandb_logger.summary(
            {
                "device": str(device),
                f"{prefix}/num_examples": num_examples,
                f"{prefix}/mse": metrics.mse,
                f"{prefix}/mse_std": metrics.mse_std,
                f"{prefix}/mae": metrics.mae,
                f"{prefix}/mae_std": metrics.mae_std,
                f"{prefix}/psnr": metrics.psnr,
                f"{prefix}/psnr_std": metrics.psnr_std,
                f"{prefix}/ssim": metrics.ssim,
                f"{prefix}/ssim_std": metrics.ssim_std,
                f"{prefix}/cr": args.cr if args.task == "spi" else None,
                f"{prefix}/scale": args.scale if args.task == "SR" else None,
                "paths/metrics_csv": str(csv_path),
                "paths/qualitative_image": str(qualitative_path),
                "paths/visual_svg": str(visual_svg_path),
            }
        )
        wandb_logger.log_table(f"{prefix}/metrics_table", csv_path)
        wandb_logger.log_image(
            f"{prefix}/qualitative",
            qualitative_path,
            caption=f"GANI qualitative results for {args.task}/{args.split}",
        )
        wandb_logger.log_artifact(csv_path, artifact_name=f"gani-{args.task}-{args.split}-metrics")

        print(f"split={args.split}")
        print(f"task={args.task}")
        print(f"num_examples={num_examples}")
        print(f"mse={metrics.mse:.6f} ± {metrics.mse_std:.6f}")
        print(f"mae={metrics.mae:.6f} ± {metrics.mae_std:.6f}")
        print(f"psnr={metrics.psnr:.4f} ± {metrics.psnr_std:.4f}")
        print(f"ssim={metrics.ssim:.4f} ± {metrics.ssim_std:.4f}")
        print(f"csv_path={csv_path}")
        print(f"qualitative_path={qualitative_path}")
        print(f"visual_svg_path={visual_svg_path}")
    finally:
        wandb_logger.finish()


if __name__ == "__main__":
    main()
