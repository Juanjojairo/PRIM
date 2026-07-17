"""Evaluate GANI on the requested MNIST test conditions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from utils.experiments import format_float_token, join_name_parts


GENERATOR_CHECKPOINT = Path(
    "results/generator_wgangp_mnist32_e500_bs128"
    "_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt"
)


@dataclass(frozen=True)
class GaniExperiment:
    task: str
    value: float | int
    encoder_checkpoint: Path

    @property
    def condition_token(self) -> str:
        if self.task == "spi":
            return f"cr_{format_float_token(float(self.value))}"
        return f"sr_{int(self.value)}"

    @property
    def output_name(self) -> str:
        return join_name_parts("eval", "gani", self.task, "mnist32", "test", self.condition_token)


EXPERIMENTS = [
    GaniExperiment(
        task="spi",
        value=1e-2,
        encoder_checkpoint=Path("results/encoder_spi_mnist32_cr_1e-2_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"),
    ),
    GaniExperiment(
        task="spi",
        value=5e-2,
        encoder_checkpoint=Path("results/encoder_spi_mnist32_cr_5e-2_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"),
    ),
    GaniExperiment(
        task="SR",
        value=8,
        encoder_checkpoint=Path("results/encoder_SR_mnist32_sr_8_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"),
    ),
    GaniExperiment(
        task="SR",
        value=4,
        encoder_checkpoint=Path("results/encoder_SR_mnist32_sr_4_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"),
    ),
]


def validate_checkpoints(experiments: list[GaniExperiment]) -> None:
    missing: list[Path] = []
    if not GENERATOR_CHECKPOINT.exists():
        missing.append(GENERATOR_CHECKPOINT)
    for experiment in experiments:
        if not experiment.encoder_checkpoint.exists():
            missing.append(experiment.encoder_checkpoint)

    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required checkpoints:\n{formatted}")


def synchronize_device(device: object) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "task",
        "condition",
        "split",
        "num_examples",
        "mse",
        "mse_std",
        "mae",
        "mae_std",
        "psnr",
        "psnr_std",
        "ssim",
        "ssim_std",
        "elapsed_seconds",
        "ms_per_sample",
        "samples_per_second",
        "encoder_checkpoint",
        "generator_checkpoint",
        "metrics_csv",
        "qualitative_path",
        "visual_svg_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate requested GANI experiments on test.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--save-root", type=Path, default=Path("results/eval_gani_experiments"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit. Leave unset to evaluate the full test dataset.",
    )
    parser.add_argument("--num-qualitative", type=int, default=8)
    parser.add_argument("--num-visual", type=int, default=8)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate the expected checkpoint paths and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_checkpoints(EXPERIMENTS)
    if args.check_only:
        print("All required checkpoints were found.")
        for experiment in EXPERIMENTS:
            print(f"{experiment.output_name}: encoder={experiment.encoder_checkpoint}")
        print(f"generator={GENERATOR_CHECKPOINT}")
        return

    from datasets.mnist import get_mnist_dataloaders
    from eval_gani import evaluate_gani, export_qualitative_grid, write_metrics_csv
    from models.encoder import load_encoder_checkpoint
    from models.generator import load_generator_checkpoint, resolve_device
    from ops.SR import SR
    from ops.forward_models import LinearSensingOperator
    from utils.observations import build_observation_batch, freeze_module
    from utils.seed import set_seed
    from utils.visualization import save_reconstruction_triplets_svg

    import torch

    set_seed(args.seed)
    device = resolve_device(args.device)
    loaders = get_mnist_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=args.download,
    )
    test_loader = loaders["test"]

    generator = load_generator_checkpoint(GENERATOR_CHECKPOINT, device=device)
    freeze_module(generator)

    summary_rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        print(f"Evaluating {experiment.output_name}")
        set_seed(args.seed)

        encoder = load_encoder_checkpoint(experiment.encoder_checkpoint, device=device)
        freeze_module(encoder)
        operator = (
            LinearSensingOperator(cr=float(experiment.value)).to(device)
            if experiment.task == "spi"
            else SR(s=int(experiment.value)).to(device)
        )

        synchronize_device(device)
        start_time = perf_counter()
        metrics, num_examples = evaluate_gani(
            encoder=encoder,
            generator=generator,
            loader=test_loader,
            device=device,
            task=experiment.task,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            max_batches=args.max_batches,
        )
        synchronize_device(device)
        elapsed_seconds = perf_counter() - start_time
        ms_per_sample = 1000.0 * elapsed_seconds / max(num_examples, 1)
        samples_per_second = num_examples / max(elapsed_seconds, 1e-12)

        output_dir = args.save_root / experiment.output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = output_dir / f"{experiment.task}_test_metrics.csv"
        qualitative_path = output_dir / f"{experiment.task}_test_qualitative.png"
        visual_svg_path = output_dir / f"{experiment.task}_test_visual_triplets.svg"

        write_metrics_csv(
            path=metrics_csv,
            split="test",
            task=experiment.task,
            metrics=metrics,
            num_examples=num_examples,
            cr=float(experiment.value) if experiment.task == "spi" else None,
            scale=int(experiment.value) if experiment.task == "SR" else None,
            measurement_noise_std=args.measurement_noise_std,
        )
        export_qualitative_grid(
            encoder=encoder,
            generator=generator,
            loader=test_loader,
            device=device,
            task=experiment.task,
            save_path=qualitative_path,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            num_images=args.num_qualitative,
        )
        visual_images, _ = next(iter(test_loader))
        visual_images = visual_images[: args.num_visual].to(device)
        visual_observation = build_observation_batch(
            task=experiment.task,
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
            title=f"GANI {experiment.task} {experiment.condition_token}",
        )

        summary_rows.append(
            {
                "method": "GANI",
                "task": experiment.task,
                "condition": experiment.condition_token,
                "split": "test",
                "num_examples": num_examples,
                "mse": f"{metrics.mse:.6f}",
                "mse_std": f"{metrics.mse_std:.6f}",
                "mae": f"{metrics.mae:.6f}",
                "mae_std": f"{metrics.mae_std:.6f}",
                "psnr": f"{metrics.psnr:.4f}",
                "psnr_std": f"{metrics.psnr_std:.4f}",
                "ssim": f"{metrics.ssim:.4f}",
                "ssim_std": f"{metrics.ssim_std:.4f}",
                "elapsed_seconds": f"{elapsed_seconds:.3f}",
                "ms_per_sample": f"{ms_per_sample:.6f}",
                "samples_per_second": f"{samples_per_second:.6f}",
                "encoder_checkpoint": str(experiment.encoder_checkpoint),
                "generator_checkpoint": str(GENERATOR_CHECKPOINT),
                "metrics_csv": str(metrics_csv),
                "qualitative_path": str(qualitative_path),
                "visual_svg_path": str(visual_svg_path),
            }
        )
        print(
            f"  num_examples={num_examples} "
            f"mse={metrics.mse:.6f} ± {metrics.mse_std:.6f} "
            f"mae={metrics.mae:.6f} ± {metrics.mae_std:.6f} "
            f"psnr={metrics.psnr:.4f} ± {metrics.psnr_std:.4f} "
            f"ssim={metrics.ssim:.4f} ± {metrics.ssim_std:.4f} "
            f"elapsed={elapsed_seconds:.2f}s "
            f"ms/sample={ms_per_sample:.4f}"
        )

    summary_path = args.save_root / "gani_test_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"summary_csv_path={summary_path}")


if __name__ == "__main__":
    main()
