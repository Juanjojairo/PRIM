"""Select ADMM iteration counts from short validation runs.

The input params file supplies gamma/beta/sigma. This script evaluates each
experiment on a small split subset, picks the iteration with best PSNR, and
writes a params file that also includes iterations=<best_iteration>.
"""
from tqdm import tqdm
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from datasets.mnist import get_mnist_dataloaders
from eval_admm_experiments import (
    DEFAULT_PARAMS_PATH,
    GENERATOR_CHECKPOINT,
    AdmmExperiment,
    encoder_checkpoint_for,
    parse_sweep_params,
    validate_checkpoints,
)
from models.baselines import run_eadmm, run_peadmm
from models.encoder import load_encoder_checkpoint
from models.generator import load_generator_checkpoint, resolve_device
from ops.SR import SR
from ops.metrics import ReconstructionMetrics, compute_reconstruction_metrics
from ops.forward_models import LinearSensingOperator
from utils.observations import build_problem_batch, freeze_module
from utils.seed import set_seed


DEFAULT_OUTPUT_PARAMS_PATH = Path("results/eadmm_peadmm_sweep_params_with_iterations")


def choose_best_iteration(
    recorded_iterations: list[int],
    metrics_curve: list[ReconstructionMetrics],
) -> tuple[int, float]:
    best_iteration = recorded_iterations[0]
    best_psnr = float("-inf")
    for iteration, metrics in zip(recorded_iterations, metrics_curve):
        if metrics.psnr > best_psnr:
            best_iteration = iteration
            best_psnr = float(metrics.psnr)
    return best_iteration, best_psnr


def build_recorded_iterations(num_iterations: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError(f"selection stride must be positive, got {stride}.")
    iterations = list(range(0, num_iterations + 1, stride))
    if iterations[-1] != num_iterations:
        iterations.append(num_iterations)
    return iterations


def evaluate_admm_checkpoints(
    *,
    experiment: AdmmExperiment,
    generator,
    encoder,
    loader,
    device,
    operator,
    recorded_iterations: list[int],
    num_examples: int,
    measurement_noise_std: float,
    random_init_std: float,
) -> tuple[list[ReconstructionMetrics], int]:
    metric_sums = {
        metric_name: [0.0 for _ in recorded_iterations]
        for metric_name in (
            "mse", "mse_std",
            "mae", "mae_std",
            "psnr", "psnr_std",
            "ssim", "ssim_std",
        )
    }
    total_examples = 0
    record_set = set(recorded_iterations)

    for images, _ in tqdm(loader):
        remaining_examples = num_examples - total_examples
        if remaining_examples <= 0:
            break
        if images.shape[0] > remaining_examples:
            images = images[:remaining_examples]
        images = images.to(device)
        problem = build_problem_batch(
            task=experiment.task,
            images=images,
            operator=operator,
            measurement_noise_std=measurement_noise_std,
        )

        if experiment.method == "PEADMM":
            result = run_peadmm(
                generator=generator,
                encoder=encoder,
                problem=problem,
                task=experiment.task,
                num_iterations=recorded_iterations[-1],
                gamma=experiment.gamma,
                beta=experiment.beta,
                sigma=experiment.sigma,
                operator=operator,
                record_iterations=record_set,
            )
        else:
            result = run_eadmm(
                generator=generator,
                problem=problem,
                task=experiment.task,
                num_iterations=recorded_iterations[-1],
                gamma=experiment.gamma,
                beta=experiment.beta,
                sigma=experiment.sigma,
                operator=operator,
                random_init_std=random_init_std,
                record_iterations=record_set,
            )

        batch_size = images.shape[0]
        for index in range(result.history.shape[1]):
            metrics = compute_reconstruction_metrics(result.history[:, index].cpu(), images.cpu())
            metric_sums["mse"][index] += metrics.mse * batch_size
            metric_sums["mae"][index] += metrics.mae * batch_size
            metric_sums["psnr"][index] += metrics.psnr * batch_size
            metric_sums["ssim"][index] += metrics.ssim * batch_size
            metric_sums["mse_std"][index] += metrics.mse_std * batch_size
            metric_sums["mae_std"][index] += metrics.mae_std * batch_size
            metric_sums["psnr_std"][index] += metrics.psnr_std * batch_size
            metric_sums["ssim_std"][index] += metrics.ssim_std * batch_size
        total_examples += batch_size

    if total_examples <= 0:
        raise RuntimeError("No examples were processed during ADMM checkpoint evaluation.")

    metrics_curve = [
        ReconstructionMetrics(
            mse=metric_sums["mse"][index] / total_examples,
            mse_std=metric_sums["mse_std"][index] / total_examples,
            mae=metric_sums["mae"][index] / total_examples,
            mae_std=metric_sums["mae_std"][index] / total_examples,
            psnr=metric_sums["psnr"][index] / total_examples,
            psnr_std=metric_sums["psnr_std"][index] / total_examples,
            ssim=metric_sums["ssim"][index] / total_examples,
            ssim_std=metric_sums["ssim_std"][index] / total_examples,
        )
        for index in range(len(recorded_iterations))
    ]
    return metrics_curve, total_examples


def write_checkpoint_metrics_csv(
    path: Path,
    *,
    method: str,
    split: str,
    task: str,
    recorded_iterations: list[int],
    metrics_curve: list[ReconstructionMetrics],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "split", "task", "iteration", "mse", "mse_std", "mae", "mae_std", "psnr", "psnr_std", "ssim", "ssim_std"],
        )
        writer.writeheader()
        for iteration, metrics in zip(recorded_iterations, metrics_curve):
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


def write_params_file(path: Path, experiments: list[AdmmExperiment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for experiment in experiments:
        lines.extend(
            [
                experiment.sweep_name,
                f"gamma={experiment.gamma:g}",
                f"beta={experiment.beta:g}",
                f"sigma={experiment.sigma:g}",
                f"iterations={experiment.iterations}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "sweep_name",
        "task",
        "condition",
        "num_examples",
        "budget_iterations",
        "optimal_iterations",
        "hit_budget_limit",
        "best_psnr",
        "final_psnr",
        "gamma",
        "beta",
        "sigma",
        "iteration_metrics_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Determine optimal ADMM iteration counts on 100 examples.")
    parser.add_argument("--params-path", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--output-params-path", type=Path, default=DEFAULT_OUTPUT_PARAMS_PATH)
    parser.add_argument("--save-root", type=Path, default=Path("results/admm_iteration_selection"))
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measurement-noise-std", type=float, default=0.0)
    parser.add_argument("--random-init-std", type=float, default=1.0)
    parser.add_argument(
        "--selection-stride",
        type=int,
        default=100,
        help="Evaluate PSNR/SSIM every N iterations while searching the optimum.",
    )
    parser.add_argument(
        "--eadmm-budget-iterations",
        type=int,
        default=None,
        help="Override the iteration budget used to search EADMM convergence.",
    )
    parser.add_argument(
        "--peadmm-budget-iterations",
        type=int,
        default=None,
        help="Override the iteration budget used to search PEADMM convergence.",
    )
    parser.add_argument(
        "--only-sweeps",
        type=str,
        default="",
        help="Comma-separated sweep names to rerun. Empty means all experiments.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def search_budget_for(experiment: AdmmExperiment, args: argparse.Namespace) -> int:
    """Return the number of iterations to run while selecting the optimum."""
    if experiment.method == "EADMM" and args.eadmm_budget_iterations is not None:
        return args.eadmm_budget_iterations
    if experiment.method == "PEADMM" and args.peadmm_budget_iterations is not None:
        return args.peadmm_budget_iterations
    return experiment.iterations


def main() -> None:
    args = parse_args()
    experiments = parse_sweep_params(args.params_path)
    only_sweeps = {item.strip() for item in args.only_sweeps.split(",") if item.strip()}
    if only_sweeps:
        known_sweeps = {experiment.sweep_name for experiment in experiments}
        unknown_sweeps = only_sweeps - known_sweeps
        if unknown_sweeps:
            unknown_text = ", ".join(sorted(unknown_sweeps))
            raise ValueError(f"Unknown sweeps in --only-sweeps: {unknown_text}")
    experiments_to_run = [
        experiment for experiment in experiments
        if not only_sweeps or experiment.sweep_name in only_sweeps
    ]
    validate_checkpoints(experiments)

    if args.check_only:
        for experiment in experiments_to_run:
            budget = search_budget_for(experiment, args)
            print(
                f"{experiment.sweep_name}: budget_iterations={budget} "
                f"gamma={experiment.gamma:g} beta={experiment.beta:g} sigma={experiment.sigma:g}"
            )
        return

    set_seed(args.seed)
    device = resolve_device(args.device)
    loaders = get_mnist_dataloaders(
        root=args.root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        download=args.download,
    )
    loader = loaders[args.split]

    generator = load_generator_checkpoint(GENERATOR_CHECKPOINT, device=device)
    freeze_module(generator)

    selected_by_sweep: dict[str, AdmmExperiment] = {}
    summary_rows: list[dict[str, object]] = []
    for experiment in experiments_to_run:
        budget_iterations = search_budget_for(experiment, args)
        print(
            f"Selecting iterations for {experiment.method} {experiment.task} "
            f"{experiment.condition_token} with budget {budget_iterations}"
        )
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
        recorded_iterations = build_recorded_iterations(budget_iterations, args.selection_stride)
        metrics_curve, num_examples = evaluate_admm_checkpoints(
            experiment=experiment,
            generator=generator,
            encoder=encoder,
            loader=loader,
            device=device,
            operator=operator,
            recorded_iterations=recorded_iterations,
            num_examples=args.num_examples,
            measurement_noise_std=args.measurement_noise_std,
            random_init_std=args.random_init_std,
        )
        best_iteration, best_psnr = choose_best_iteration(recorded_iterations, metrics_curve)
        final_psnr = metrics_curve[-1].psnr
        hit_budget_limit = best_iteration == budget_iterations
        selected = replace(experiment, iterations=best_iteration)
        selected_by_sweep[experiment.sweep_name] = selected

        output_dir = args.save_root / selected.output_name
        iteration_csv = output_dir / f"{experiment.task}_{args.split}_iteration_metrics.csv"
        write_checkpoint_metrics_csv(
            path=iteration_csv,
            method=experiment.method,
            split=args.split,
            task=experiment.task,
            recorded_iterations=recorded_iterations,
            metrics_curve=metrics_curve,
        )

        summary_rows.append(
            {
                "method": experiment.method,
                "sweep_name": experiment.sweep_name,
                "task": experiment.task,
                "condition": experiment.condition_token,
                "num_examples": num_examples,
                "budget_iterations": budget_iterations,
                "optimal_iterations": best_iteration,
                "hit_budget_limit": int(hit_budget_limit),
                "best_psnr": f"{best_psnr:.4f}",
                "final_psnr": f"{final_psnr:.4f}",
                "gamma": f"{experiment.gamma:g}",
                "beta": f"{experiment.beta:g}",
                "sigma": f"{experiment.sigma:g}",
                "iteration_metrics_csv": str(iteration_csv),
            }
        )
        print(
            f"  optimal_iterations={best_iteration} best_psnr={best_psnr:.4f} "
            f"final_psnr={final_psnr:.4f} num_examples={num_examples}"
        )
        if hit_budget_limit:
            print("  WARNING: best iteration is at the budget limit; increase the budget and rerun.")

    merged_experiments = [
        selected_by_sweep.get(experiment.sweep_name, experiment)
        for experiment in experiments
    ]
    write_params_file(args.output_params_path, merged_experiments)
    summary_path = args.save_root / "admm_iteration_selection_summary.csv"
    write_summary_csv(summary_path, summary_rows)
    print(f"output_params_path={args.output_params_path}")
    print(f"summary_csv_path={summary_path}")


if __name__ == "__main__":
    main()
