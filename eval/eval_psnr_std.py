from __future__ import annotations
from pathlib import Path
from admm_common import evaluate_admm_solver
from datasets.mnist import get_mnist_dataloaders
from models.baselines import run_peadmm
from models.encoder import load_encoder_checkpoint
from models.generator import (
    load_generator_checkpoint,
    resolve_device,
)
from ops.forward_models import LinearSensingOperator
from utils.observations import freeze_module
from utils.seed import set_seed

def main():
    method = "PEADMM"

    task = "spi"

    cr = 1e-2

    iterations = 100

    gamma = 200

    beta = 1

    sigma = 0.005
    

    GENERATOR_CHECKPOINT = Path(
        "results/generator_wgangp_mnist32_e500_bs128"
        "_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt"
    )

    ENCODER_CHECKPOINT = Path(
        "results/encoder_spi_mnist32_cr_1e-2_e500_bs128_lr_1e-3_z128_ch64/encoder.pt"
    )

    set_seed(42)

    device = resolve_device("auto")

    generator = load_generator_checkpoint(
        GENERATOR_CHECKPOINT,
        device=device,
    )
    freeze_module(generator)

    encoder = load_encoder_checkpoint(
        ENCODER_CHECKPOINT,
        device=device,
    )
    freeze_module(encoder)

    loaders = get_mnist_dataloaders(
        root=Path("data"),
        batch_size=128,
        num_workers=0,
        seed=42,
        download=True,
    )

    test_loader = loaders["test"]

    operator = LinearSensingOperator(cr=cr).to(device)

    metrics, *_ = evaluate_admm_solver(
        solver=run_peadmm,
        method_name=method,
        generator=generator,
        loader=test_loader,
        device=device,
        task=task,
        num_iterations=iterations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        operator=operator,
        encoder=encoder,
    )

    print(f"MSE  = {metrics.mse:.6f}")
    print(f"MAE  = {metrics.mae:.6f}")
    print(f"PSNR = {metrics.psnr:.4f} ± {metrics.psnr_std:.4f}")
    print(f"SSIM = {metrics.ssim:.4f}")
    

if __name__ == "__main__":
    main()