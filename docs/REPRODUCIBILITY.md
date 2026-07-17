# Reproducibility guide

This guide describes the executable workflow represented by the current source tree. Commands are intended to run from the repository root. The CLI definitions in each Python module are authoritative; the YAML files under `configs/` are reference configurations and W&B sweep definitions.

## Experimental protocol

| Component | Setting |
|:--|:--|
| Dataset | MNIST, resized to 32×32 grayscale |
| Split | 55,000 train / 5,000 validation / 10,000 test |
| Pixel range | `[0, 1]` |
| Seed | 42 by default |
| SPI operator | Row-truncated Hadamard matrix, zig-zag row ordering |
| SPI conditions in the manuscript | κ=0.01 and κ=0.05 |
| SR operator | `s×s` average filter followed by stride `s` |
| SR conditions in the manuscript | ×8 and ×4 |
| RIM unrolling | 10 steps, 32 hidden channels, step scale 0.1 |
| Metrics | PSNR, SSIM, MSE, MAE |

The operator implementations are in `ops/forward_models.py` and `ops/SR.py`. Degraded observations and fidelity targets are assembled in `utils/observations.py`.

## 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use `.venv\Scripts\Activate.ps1` instead of the `source` command in Windows PowerShell.

Authenticate only when using online W&B tracking:

```bash
wandb login
```

Offline runs use `--wandb-mode offline` and do not require login or network access. The default project is `prim-inverse-problems`; it can be overridden with `--wandb-project` or `WANDB_PROJECT`.

The current `requirements.txt` contains package names without exact versions. For an archival experiment, capture the resolved environment after installation:

```bash
python --version
python -m pip freeze > environment-lock.txt
```

Treat `environment-lock.txt` as experiment metadata. Review platform-specific CUDA packages before using it as a cross-platform lock file.

## 2. Validate the forward operators

The SPI module includes shape, adjoint, and full-rate matched-filter checks:

```bash
python -m ops.forward_models --cr 0.1
python -m ops.forward_models --cr 0.01
```

## 3. Train the WGAN-GP prior

```bash
python -m train_generator \
  --device cuda \
  --epochs 500 \
  --batch-size 128 \
  --latent-dim 128 \
  --base-channels 64 \
  --activation elu \
  --generator-lr 1e-4 \
  --discriminator-lr 1e-4 \
  --gradient-penalty-weight 10 \
  --critic-iterations 5 \
  --wandb-project prim-inverse-problems \
  --wandb-mode online
```

With these arguments, the expected best checkpoint is:

```text
results/generator_wgangp_mnist32_e500_bs128_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt
```

The generator checkpoint stores the generator architecture, weights, optional discriminator weights, and training metrics.

## 4. Train task-specific encoders

The two encoder files share the same training pipeline; their filenames and defaults encode the condition used when each copy was created. Explicit CLI flags can select the task condition.

Set the generator path once in your shell or substitute it directly in each command:

```bash
GENERATOR=results/generator_wgangp_mnist32_e500_bs128_glr_1e-4_dlr_1e-4_z128_ch64_gp10_crit5_elu/generator.pt
```

### SPI, κ=0.01

```bash
python -m train_encoder_spi0005 \
  --task spi --cr 0.01 --device cuda --epochs 500 --batch-size 128 \
  --generator-checkpoint "$GENERATOR" \
  --wandb-group encoder-spi-cr001 --wandb-mode online
```

### SPI, κ=0.05

```bash
python -m train_encoder_spi0005 \
  --task spi --cr 0.05 --device cuda --epochs 500 --batch-size 128 \
  --generator-checkpoint "$GENERATOR" \
  --wandb-group encoder-spi-cr005 --wandb-mode online
```

### SR ×8

```bash
python -m train_encoder_sr16 \
  --task SR --scale 8 --device cuda --epochs 500 --batch-size 128 \
  --generator-checkpoint "$GENERATOR" \
  --wandb-group encoder-sr-x8 --wandb-mode online
```

### SR ×4

```bash
python -m train_encoder_sr16 \
  --task SR --scale 4 --device cuda --epochs 500 --batch-size 128 \
  --generator-checkpoint "$GENERATOR" \
  --wandb-group encoder-sr-x4 --wandb-mode online
```

Expected best-checkpoint directories follow these patterns:

```text
results/encoder_spi_mnist32_cr_<ratio>_e500_bs128_lr_1e-3_z128_ch64/encoder.pt
results/encoder_SR_mnist32_sr_<scale>_e500_bs128_lr_1e-3_z128_ch64/encoder.pt
```

## 5. Train RIM and PRIM

The standard RIM and PRIM use the same `ImageSpaceRIM`. Their key difference is initialization:

| Variant | Initialization | Generator prior |
|:--|:--|:--|
| Standard RIM (`nogp`) | Normalized backprojection | Disabled |
| PRIM (`prop`) | `G*(E*(Hᵀy))` | Frozen encoder-generator pair |

The condition-specific entry points used for the manuscript are:

| Condition | Standard RIM | PRIM |
|:--|:--|:--|
| SPI, κ=0.01 | `train_rim_nogp_spi_cr001` | `train_rim_prop_spi_cr001` |
| SPI, κ=0.05 | `train_rim_nogp_spi_cr005` | `train_rim_prop_spi_cr005` |
| SR ×8 | `train_rim_nogp_sr_x8` | `train_rim_prop_sr_x8` |
| SR ×4 | `train_rim_nogp_sr_x4` | `train_rim_prop_sr_x4` |

For example:

```bash
python -m train_rim_nogp_spi_cr001 \
  --device cuda --steps 10 --hidden-channels 32 \
  --wandb-group rim-spi-cr001 --wandb-mode online

python -m train_rim_prop_spi_cr001 \
  --device cuda --steps 10 --hidden-channels 32 \
  --wandb-group prim-spi-cr001 --wandb-mode online
```

Each training run writes:

- `rim.pt`: checkpoint selected by validation PSNR;
- `rim_last.pt`: final-epoch checkpoint;
- per-epoch qualitative grids;
- per-iteration validation CSV files;
- final test trajectory metrics;
- W&B scalars, images, tables, and checkpoint artifacts.

Use `--max-train-batches` and `--max-val-batches` for a short integration run before launching full training.

## 6. Evaluate learned methods

The aggregate evaluators expect the checkpoint paths encoded in their `EXPERIMENTS` constants.

Validate paths without running inference:

```bash
python -m eval_gani_experiments --check-only
python -m eval_rim_experiments --check-only
```

Evaluate the four manuscript conditions:

```bash
python -m eval_gani_experiments --device cuda
python -m eval_rim_experiments --device cuda
```

The RIM evaluator covers eight runs: standard RIM and PRIM for SPI κ∈{0.01, 0.05} and SR scales {×8, ×4}. It writes aggregate CSV summaries, per-step curves, plots, and editable SVG triplets.

## 7. Tune and evaluate EADMM/PEADMM

Bayesian W&B sweep templates live under `configs/sweeps/` for each method and condition. For example:

```bash
wandb sweep configs/sweeps/eadmm_spi_cr001_bayes.yaml
wandb agent <SWEEP_ID>
```

The sweep templates search discrete values of `gamma`, `beta`, and `sigma` on the validation split. After selecting those values, `determine_admm_iterations.py` can choose an iteration count from validation trajectories.

The aggregate ADMM evaluator reads a plain-text parameter file with blank-line-separated blocks:

```text
sweep-eadmm-spi-cr001-val
gamma=100
beta=0.1
sigma=0.05
iterations=5000

sweep-peadmm-spi-cr001-val
gamma=200
beta=1
sigma=0.005
iterations=100
```

Run selection and evaluation with:

```bash
python -m determine_admm_iterations \
  --params-path results/eadmm_peadmm_sweep_params \
  --output-params-path results/eadmm_peadmm_sweep_params_with_iterations \
  --device cuda

python -m eval_admm_experiments \
  --params-path results/eadmm_peadmm_sweep_params_with_iterations \
  --device cuda
```

The parameters reported in the current experiment summaries are:

| Method | Condition | Iterations | γ | β | σ |
|:--|:--|--:|--:|--:|--:|
| EADMM | SPI κ=0.01 | 5,000 | 100 | 0.1 | 0.05 |
| PEADMM | SPI κ=0.01 | 100 | 200 | 1 | 0.005 |
| EADMM | SPI κ=0.05 | 10,000 | 200 | 0.02 | 0.1 |
| PEADMM | SPI κ=0.05 | 18,000 | 200 | 0.01 | 0.05 |
| EADMM | SR ×8 | 5,000 | 200 | 0.005 | 0.05 |
| PEADMM | SR ×8 | 500 | 200 | 0.01 | 0.02 |
| EADMM | SR ×4 | 10,000 | 200 | 0.005 | 0.01 |
| PEADMM | SR ×4 | 4,500 | 200 | 0.005 | 0.1 |

## 8. Runtime and rollout figures

Once every checkpoint and the ADMM parameter file are present:

```bash
python -m time_inference_experiments \
  --params-path results/eadmm_peadmm_sweep_params_with_iterations \
  --device cuda

python -m visualize_rim_prim_rollouts \
  --device cuda \
  --only-conditions spi_cr_1e-2,sr_8
```

Runtime measurements synchronize CUDA around each call. Record the GPU model, PyTorch/CUDA versions, warm-up count, and sample selection whenever publishing new measurements.

## Artifact policy

The repository ignores the following local products:

- downloaded MNIST files under `data/`;
- model checkpoints (`*.pt`, `*.pth`, `*.ckpt`);
- `results/`, `history/`, and local W&B directories;
- LaTeX build intermediates;
- archives.

For a public reproducibility release, publish essential checkpoints through a versioned GitHub Release, W&B Artifact collection, or an archival data repository, and record checksums plus download instructions. Do not place multi-gigabyte experiment directories in Git.

## Verification checklist

Before reporting a run:

1. Record the commit SHA and resolved dependency versions.
2. Confirm the seed, task condition, operator, checkpoint paths, and W&B mode.
3. Run the SPI operator smoke test.
4. Verify that evaluation used all 10,000 test images unless explicitly labeled as a subset.
5. Keep hyperparameter selection on validation data and reserve test data for final reporting.
6. Archive aggregate metrics, per-step curves, and the exact checkpoint metadata.
7. State the hardware and synchronization procedure for runtime measurements.
