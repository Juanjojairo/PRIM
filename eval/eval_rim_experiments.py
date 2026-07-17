"""Evaluate the requested RIM checkpoints on the full MNIST test split.

This script covers:
  - SPI at CR 1e-2 and 5e-2
  - SR at x8 and x4
  - RIM without generator prior and RIM with the proposed generator prior
"""

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
class RimExperiment:
    variant: str
    task: str
    value: float | int
    encoder_checkpoint: Path | None
    rim_checkpoint: Path

    @property
    def uses_generator_prior(self) -> bool:
        return self.variant == "rim_prop"

    @property
    def init_mode(self) -> str:
        return "learned" if self.uses_generator_prior else "backprojection"

    @property
    def condition_token(self) -> str:
        if self.task == "spi":
            return f"cr_{format_float_token(float(self.value))}"
        return f"sr_{int(self.value)}"

    @property
    def output_name(self) -> str:
        return join_name_parts("eval", self.variant, self.task, "mnist32", "test", self.condition_token)


EXPERIMENTS = [
    RimExperiment(
        variant="rim_prop",
        task="spi",
        value=1e-2,
        encoder_checkpoint=Path("results/encoder_spi_mnist32_cr_1e-2_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"),
        rim_checkpoint=Path("results/rim/rim_spi_mnist32_gp_init_learned_e100_bs128_lr_1e-3_steps10_hc32_cr_1e-2/rim.pt"),
    ),
    RimExperiment(
        variant="rim_prop",
        task="spi",
        value=5e-2,
        encoder_checkpoint=Path("results/encoder_spi_mnist32_cr_5e-2_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"),
        rim_checkpoint=Path("results/rim/rim_spi_mnist32_gp_init_learned_e300_bs128_lr_1e-3_steps10_hc32_cr_5e-2/rim.pt"),
    ),
    RimExperiment(
        variant="rim_prop",
        task="SR",
        value=8,
        encoder_checkpoint=Path("results/encoder_SR_mnist32_sr_8_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"),
        rim_checkpoint=Path("results/rim/rim_SR_mnist32_gp_init_learned_e100_bs128_lr_1e-3_steps10_hc32_sr_8/rim.pt"),
    ),
    RimExperiment(
        variant="rim_prop",
        task="SR",
        value=4,
        encoder_checkpoint=Path("results/encoder_SR_mnist32_sr_4_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"),
        rim_checkpoint=Path("results/rim/rim_SR_mnist32_gp_init_learned_e300_bs128_lr_1e-3_steps10_hc32_sr_4/rim.pt"),
    ),
    RimExperiment(
        variant="rim_nogp",
        task="spi",
        value=1e-2,
        encoder_checkpoint=None,
        rim_checkpoint=Path("results/rim/rim_spi_mnist32_nogp_init_observation_e300_bs128_lr_1e-3_steps10_hc32_cr_1e-2/rim.pt"),
    ),
    RimExperiment(
        variant="rim_nogp",
        task="spi",
        value=5e-2,
        encoder_checkpoint=None,
        rim_checkpoint=Path("results/rim/rim_spi_mnist32_nogp_init_observation_e300_bs128_lr_1e-3_steps10_hc32_cr_5e-2/rim.pt"),
    ),
    RimExperiment(
        variant="rim_nogp",
        task="SR",
        value=8,
        encoder_checkpoint=None,
        rim_checkpoint=Path("results/rim/rim_SR_mnist32_nogp_init_observation_e500_bs128_lr_1e-3_steps10_hc32_sr_8/rim.pt"),
    ),
    RimExperiment(
        variant="rim_nogp",
        task="SR",
        value=4,
        encoder_checkpoint=None,
        rim_checkpoint=Path("results/rim/rim_SR_mnist32_nogp_init_observation_e500_bs128_lr_1e-3_steps10_hc32_sr_4/rim.pt"),
    ),
]


def validate_checkpoints(experiments: list[RimExperiment]) -> None:
    missing: list[Path] = []
    for path in [GENERATOR_CHECKPOINT]:
        if not path.exists():
            missing.append(path)
    for experiment in experiments:
        if not experiment.rim_checkpoint.exists():
            missing.append(experiment.rim_checkpoint)
        if experiment.encoder_checkpoint is not None and not experiment.encoder_checkpoint.exists():
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
        "variant",
        "task",
        "condition",
        "split",
        "num_examples",
        "iterations",
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
        "rim_checkpoint",
        "encoder_checkpoint",
        "visual_svg_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all requested RIM experiments on test.")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--save-root", type=Path, default=Path("results/eval_rim_experiments"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional smoke-test limit. Leave unset to evaluate the full test dataset.",
    )
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
            encoder = experiment.encoder_checkpoint or Path("<not used>")
            print(f"{experiment.output_name}: rim={experiment.rim_checkpoint} encoder={encoder}")
        return

    import torch

    from datasets.mnist import get_mnist_dataloaders
    from eval_rim_prop_spi_cr001 import (
        evaluate_proposed_rim,
        save_combined_psnr_plot,
        write_iteration_metrics_curves_csv,
    )
    from models.encoder import load_encoder_checkpoint
    from models.generator import load_generator_checkpoint, resolve_device
    from models.rim import load_rim_checkpoint
    from ops.SR import SR
    from ops.forward_models import LinearSensingOperator
    from utils.observations import freeze_module
    from utils.seed import set_seed
    from utils.visualization import save_reconstruction_triplets_svg

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
    identity = torch.nn.Identity().to(device)

    summary_rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        print(f"Evaluating {experiment.output_name}")
        set_seed(args.seed)

        rim = load_rim_checkpoint(experiment.rim_checkpoint, device=device)
        freeze_module(rim)
        if experiment.uses_generator_prior:
            if experiment.encoder_checkpoint is None:
                raise ValueError(f"{experiment.output_name} requires an encoder checkpoint.")
            encoder = load_encoder_checkpoint(experiment.encoder_checkpoint, device=device)
            freeze_module(encoder)
            eval_generator = generator
        else:
            encoder = identity
            eval_generator = identity

        operator = (
            LinearSensingOperator(cr=float(experiment.value)).to(device)
            if experiment.task == "spi"
            else SR(s=int(experiment.value)).to(device)
        )

        synchronize_device(device)
        start_time = perf_counter()
        metrics, metrics_curve, first_problem, first_history, first_targets, num_examples = evaluate_proposed_rim(
            rim=rim,
            encoder=encoder,
            generator=eval_generator,
            loader=test_loader,
            device=device,
            task=experiment.task,
            operator=operator,
            measurement_noise_std=args.measurement_noise_std,
            max_batches=args.max_batches,
            use_generator_prior=experiment.uses_generator_prior,
            init_mode=experiment.init_mode,
            desc=experiment.output_name,
        )
        synchronize_device(device)
        elapsed_seconds = perf_counter() - start_time
        ms_per_sample = 1000.0 * elapsed_seconds / max(num_examples, 1)
        samples_per_second = num_examples / max(elapsed_seconds, 1e-12)

        output_dir = args.save_root / experiment.output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        iteration_metrics_path = output_dir / f"{experiment.task}_test_iteration_metrics.csv"
        plot_path = output_dir / f"{experiment.task}_test_psnr.png"
        visual_svg_path = output_dir / f"{experiment.task}_test_visual_triplets.svg"
        write_iteration_metrics_curves_csv(
            path=iteration_metrics_path,
            task=experiment.task,
            metrics_curves={experiment.variant: metrics_curve},
        )
        save_combined_psnr_plot(
            save_path=plot_path,
            task=experiment.task,
            psnr_curves={
                experiment.variant: torch.tensor([item.psnr for item in metrics_curve], dtype=torch.float32)
            },
            title=f"{experiment.variant} {experiment.task} {experiment.condition_token}",
        )
        save_reconstruction_triplets_svg(
            save_path=visual_svg_path,
            backprojection=first_problem.observation,
            estimate=first_history[:, -1],
            target=first_targets,
            num_images=args.num_visual,
            title=f"{experiment.variant} {experiment.task} {experiment.condition_token}",
        )

        summary_rows.append(
            {
                "variant": experiment.variant,
                "task": experiment.task,
                "condition": experiment.condition_token,
                "split": "test",
                "num_examples": num_examples,
                "iterations": rim.steps,
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
                "rim_checkpoint": str(experiment.rim_checkpoint),
                "encoder_checkpoint": str(experiment.encoder_checkpoint or ""),
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

    summary_path = args.save_root / "rim_test_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"summary_csv_path={summary_path}")


if __name__ == "__main__":
    main()
