"""Evaluate EADMM and PEADMM using selected sweep parameters."""

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
DEFAULT_PARAMS_PATH = Path("results/eadmm_peadmm_sweep_params")


@dataclass(frozen=True)
class AdmmExperiment:
    sweep_name: str
    method: str
    task: str
    value: float | int
    iterations: int
    gamma: float
    beta: float
    sigma: float

    @property
    def condition_token(self) -> str:
        if self.task == "spi":
            return f"cr_{format_float_token(float(self.value))}"
        return f"sr_{int(self.value)}"

    @property
    def output_name(self) -> str:
        return join_name_parts(
            "eval",
            self.method.lower(),
            self.task,
            "mnist32",
            "test",
            f"iter{self.iterations}",
            f"gamma_{format_float_token(self.gamma)}",
            f"beta_{format_float_token(self.beta)}",
            f"sigma_{format_float_token(self.sigma)}",
            self.condition_token,
        )


def infer_experiment_identity(sweep_name: str) -> tuple[str, str, float | int, int]:
    tokens = sweep_name.split("-")
    if len(tokens) < 5 or tokens[0] != "sweep":
        raise ValueError(f"Unsupported sweep name: {sweep_name!r}")

    method = tokens[1].upper()
    task_token = tokens[2]
    condition = tokens[3]
    if method not in {"EADMM", "PEADMM"}:
        raise ValueError(f"Unsupported method in sweep name: {sweep_name!r}")
    if task_token == "spi":
        task = "spi"
        if condition == "cr001":
            value: float | int = 1e-2
        elif condition == "cr005":
            value = 5e-2
        else:
            raise ValueError(f"Unsupported SPI condition in sweep name: {sweep_name!r}")
    elif task_token == "sr":
        task = "SR"
        if not condition.startswith("x"):
            raise ValueError(f"Unsupported SR condition in sweep name: {sweep_name!r}")
        value = int(condition[1:])
    else:
        raise ValueError(f"Unsupported task in sweep name: {sweep_name!r}")

    iterations = 10000 if method == "EADMM" else 2500
    return method, task, value, iterations


def parse_sweep_params(path: Path) -> list[AdmmExperiment]:
    experiments: list[AdmmExperiment] = []
    current_name: str | None = None
    current_values: dict[str, float] = {}

    def flush() -> None:
        nonlocal current_name, current_values
        if current_name is None:
            return
        missing = {"gamma", "beta", "sigma"} - set(current_values)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing {missing_text} for {current_name}.")
        method, task, value, default_iterations = infer_experiment_identity(current_name)
        iterations = int(current_values.get("iterations", default_iterations))
        experiments.append(
            AdmmExperiment(
                sweep_name=current_name,
                method=method,
                task=task,
                value=value,
                iterations=iterations,
                gamma=current_values["gamma"],
                beta=current_values["beta"],
                sigma=current_values["sigma"],
            )
        )
        current_name = None
        current_values = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("sweep-"):
            flush()
            current_name = line
            current_values = {}
            continue
        if "=" not in line or current_name is None:
            raise ValueError(f"Malformed line in {path}: {raw_line!r}")
        key, value = [item.strip() for item in line.split("=", 1)]
        if key not in {"gamma", "beta", "sigma", "iterations"}:
            raise ValueError(f"Unsupported parameter {key!r} in {path}.")
        current_values[key] = float(value)

    flush()
    return experiments


def encoder_checkpoint_for(experiment: AdmmExperiment) -> Path | None:
    if experiment.method != "PEADMM":
        return None
    if experiment.task == "spi":
        return Path(
            "results"
            f"/encoder_spi_mnist32_cr_{format_float_token(float(experiment.value))}"
            "_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"
        )
    return Path(
        "results"
        f"/encoder_SR_mnist32_sr_{int(experiment.value)}"
        "_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"
    )


def validate_checkpoints(experiments: list[AdmmExperiment]) -> None:
    missing: list[Path] = []
    if not GENERATOR_CHECKPOINT.exists():
        missing.append(GENERATOR_CHECKPOINT)
    for experiment in experiments:
        encoder_checkpoint = encoder_checkpoint_for(experiment)
        if encoder_checkpoint is not None and not encoder_checkpoint.exists():
            missing.append(encoder_checkpoint)
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required checkpoints:\n{formatted}")


def parse_csv_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


def filter_experiments(
    experiments: list[AdmmExperiment],
    only_sweeps: set[str] | None,
    only_methods: set[str] | None,
) -> list[AdmmExperiment]:
    selected = experiments
    if only_sweeps is not None:
        selected = [experiment for experiment in selected if experiment.sweep_name in only_sweeps]
    if only_methods is not None:
        normalized_methods = {method.upper() for method in only_methods}
        selected = [experiment for experiment in selected if experiment.method in normalized_methods]
    if not selected:
        raise ValueError("No ADMM experiments matched the provided filters.")
    return selected


def build_recorded_iterations(num_iterations: int, stride: int | None) -> list[int]:
    if stride is None or stride <= 0:
        return list(range(num_iterations + 1))
    iterations = list(range(0, num_iterations + 1, stride))
    if iterations[-1] != num_iterations:
        iterations.append(num_iterations)
    return iterations


def synchronize_device(device: object) -> None:
    if getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "sweep_name",
        "task",
        "condition",
        "split",
        "num_examples",
        "iterations",
        "gamma",
        "beta",
        "sigma",
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
        "metrics_csv",
        "iteration_metrics_csv",
        "plot_path",
        "qualitative_path",
        "visual_svg_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EADMM/PEADMM sweep-selected params on test.")
    parser.add_argument("--params-path", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--save-root", type=Path, default=Path("results/eval_admm_experiments"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument("--random-init-std", type=float, default=1.0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--expected-test-examples",
        type=int,
        default=10000,
        help="Fail if a full test run processes a different number of examples.",
    )
    parser.add_argument("--num-qualitative", type=int, default=8)
    parser.add_argument("--num-visual", type=int, default=8)
    parser.add_argument(
        "--only-sweeps",
        type=str,
        default=None,
        help="Comma-separated sweep names to evaluate.",
    )
    parser.add_argument(
        "--only-methods",
        type=str,
        default=None,
        help="Comma-separated method names to evaluate, e.g. EADMM,PEADMM.",
    )
    parser.add_argument(
        "--curve-stride",
        type=int,
        default=100,
        help="Record iteration metrics every N iterations plus the final iterate. Use 0 to record all.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = parse_sweep_params(args.params_path)
    experiments = filter_experiments(
        experiments=experiments,
        only_sweeps=parse_csv_filter(args.only_sweeps),
        only_methods=parse_csv_filter(args.only_methods),
    )
    validate_checkpoints(experiments)
    if args.check_only:
        print("Parsed experiments:")
        for experiment in experiments:
            encoder = encoder_checkpoint_for(experiment) or Path("<not used>")
            print(
                f"{experiment.sweep_name}: method={experiment.method} task={experiment.task} "
                f"{experiment.condition_token} iter={experiment.iterations} "
                f"gamma={experiment.gamma:g} beta={experiment.beta:g} sigma={experiment.sigma:g} "
                f"encoder={encoder}"
            )
        print(f"generator={GENERATOR_CHECKPOINT}")
        return

    import torch

    from admm_common import (
        evaluate_admm_solver,
        export_admm_qualitative_grid,
        save_psnr_plot,
        write_baseline_metrics_csv,
        write_iteration_metrics_csv,
    )
    from datasets.mnist import get_mnist_dataloaders
    from models.baselines import run_eadmm, run_peadmm
    from models.encoder import load_encoder_checkpoint
    from models.generator import load_generator_checkpoint, resolve_device
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

    summary_rows: list[dict[str, object]] = []
    for experiment in experiments:
        print(f"Evaluating {experiment.output_name}")
        set_seed(args.seed)
        encoder = None
        encoder_checkpoint = encoder_checkpoint_for(experiment)
        if encoder_checkpoint is not None:
            encoder = load_encoder_checkpoint(encoder_checkpoint, device=device)
            freeze_module(encoder)

        operator = (
            LinearSensingOperator(cr=float(experiment.value)).to(device)
            if experiment.task == "spi"
            else SR(s=int(experiment.value)).to(device)
        )
        solver = run_peadmm if experiment.method == "PEADMM" else run_eadmm
        recorded_iterations = build_recorded_iterations(experiment.iterations, args.curve_stride)

        synchronize_device(device)
        start_time = perf_counter()
        metrics, metrics_curve, first_problem, first_result, first_targets, num_examples = evaluate_admm_solver(
            solver=solver,
            method_name=experiment.method,
            generator=generator,
            loader=test_loader,
            device=device,
            task=experiment.task,
            num_iterations=experiment.iterations,
            gamma=experiment.gamma,
            beta=experiment.beta,
            sigma=experiment.sigma,
            operator=operator,
            encoder=encoder,
            measurement_noise_std=args.measurement_noise_std,
            max_batches=args.max_batches,
            max_examples=args.max_examples,
            random_init_std=args.random_init_std,
            record_iterations=set(recorded_iterations),
        )
        synchronize_device(device)
        elapsed_seconds = perf_counter() - start_time
        ms_per_sample = 1000.0 * elapsed_seconds / max(num_examples, 1)
        samples_per_second = num_examples / max(elapsed_seconds, 1e-12)
        if (
            args.max_batches is None
            and args.max_examples is None
            and num_examples != args.expected_test_examples
        ):
            raise RuntimeError(
                f"Expected {args.expected_test_examples} test examples, got {num_examples} "
                f"for {experiment.output_name}."
            )

        output_dir = args.save_root / experiment.output_name
        metrics_csv = output_dir / f"{experiment.task}_test_metrics.csv"
        iteration_csv = output_dir / f"{experiment.task}_test_iteration_metrics.csv"
        plot_path = output_dir / f"{experiment.task}_test_psnr_curve.png"
        qualitative_path = output_dir / f"{experiment.task}_test_qualitative.png"
        visual_svg_path = output_dir / f"{experiment.task}_test_visual_triplets.svg"

        write_baseline_metrics_csv(
            path=metrics_csv,
            method=experiment.method,
            split="test",
            task=experiment.task,
            metrics=metrics,
            num_examples=num_examples,
            num_iterations=experiment.iterations,
            gamma=experiment.gamma,
            beta=experiment.beta,
            sigma=experiment.sigma,
            cr=float(experiment.value) if experiment.task == "spi" else None,
            scale=int(experiment.value) if experiment.task == "SR" else None,
            measurement_noise_std=args.measurement_noise_std,
        )
        write_iteration_metrics_csv(
            path=iteration_csv,
            method=experiment.method,
            split="test",
            task=experiment.task,
            metrics_curve=metrics_curve,
            iterations=recorded_iterations,
        )
        save_psnr_plot(
            path=plot_path,
            method=experiment.method,
            task=experiment.task,
            psnr_curve=torch.tensor([item.psnr for item in metrics_curve], dtype=torch.float32),
            title=f"{experiment.method} {experiment.task} {experiment.condition_token}",
            iterations=recorded_iterations,
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
            title=f"{experiment.method} {experiment.task} {experiment.condition_token}",
        )

        summary_rows.append(
            {
                "method": experiment.method,
                "sweep_name": experiment.sweep_name,
                "task": experiment.task,
                "condition": experiment.condition_token,
                "split": "test",
                "num_examples": num_examples,
                "iterations": experiment.iterations,
                "gamma": f"{experiment.gamma:g}",
                "beta": f"{experiment.beta:g}",
                "sigma": f"{experiment.sigma:g}",
                "mse": f"{metrics.mse:.6f}",
                "mse_std": f"{metrics.mse:.6f}",
                "mae": f"{metrics.mae:.6f}",
                "mae_std": f"{metrics.mae:.6f}",
                "psnr": f"{metrics.psnr:.4f}",
                "psnr_std": f"{metrics.psnr_std:.4f}",
                "ssim": f"{metrics.ssim:.4f}",
                "ssim_std": f"{metrics.ssim:.4f}",
                "elapsed_seconds": f"{elapsed_seconds:.3f}",
                "ms_per_sample": f"{ms_per_sample:.6f}",
                "samples_per_second": f"{samples_per_second:.6f}",
                "metrics_csv": str(metrics_csv),
                "iteration_metrics_csv": str(iteration_csv),
                "plot_path": str(plot_path),
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

    summary_path = args.save_root / "admm_test_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"summary_csv_path={summary_path}")


if __name__ == "__main__":
    main()
