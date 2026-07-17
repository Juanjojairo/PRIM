# Project map

PRIM is a staged research pipeline rather than a single monolithic application. Keeping these stages explicit makes model provenance and experimental comparisons easier to audit.

## End-to-end data flow

```text
MNIST image x
    │
    ├── SPI: y = Hx, Hadamard rows ordered in zig-zag
    └── SR:  y = downsample(average_filter(x))
             │
             ▼
      normalized backprojection Hᵀy
             │
       ┌─────┴──────────────┐
       │                    │
       ▼                    ▼
 standard RIM        encoder E* → latent z₀ → generator G* → PRIM x₀
       │                    │
       └────── recurrent refinement with Hᵀ(y − Hxₜ) ──────┘
                            │
                            ▼
                      reconstruction x̂
```

## Module ownership

| Area | Main files | Responsibility |
|:--|:--|:--|
| Data | `datasets/mnist.py` | Resize, normalize, split, and load MNIST reproducibly. |
| SPI | `ops/forward_models.py`, `ops/libs/ordering/` | Hadamard sensing, zig-zag ordering, adjoint, backprojection, and exact x-update. |
| SR | `ops/SR.py` | Average-filter degradation, adjoint, and normalized backprojection. |
| Metrics | `ops/metrics.py` | MSE, MAE, PSNR, SSIM, and trajectory aggregation. |
| Prior | `models/generator.py` | WGAN-GP generator/discriminator and checkpoint I/O. |
| Initialization | `models/encoder.py` | Observation-to-latent encoder and checkpoint I/O. |
| Learned solver | `models/rim.py` | ConvGRU-based image-space and latent-image RIM implementations. |
| Baselines | `models/baselines.py`, `admm_common.py` | GANI, EADMM, and PEADMM reconstruction paths. |
| Experiment utilities | `utils/` | Seeds, observations, visualization, naming, and W&B integration. |

## Training stages

1. `train_generator.py` learns the MNIST WGAN-GP prior.
2. `train_encoder_*.py` learns condition-specific latent initializers while freezing the generator.
3. `train_rim_nogp_*.py` trains standard RIM from a backprojection.
4. `train_rim_prop_*.py` trains PRIM from the frozen encoder-generator initialization.

Condition-specific files mainly encode defaults and checkpoint paths. CLI flags remain the executable source of truth.

## Evaluation families

| Family | Entry points | Output |
|:--|:--|:--|
| Single condition | `eval_gani.py`, `eval_eadmm.py`, `eval_peadmm.py`, `eval_rim_prop_spi_cr001.py` | Metrics, curves, grids, and SVG triplets. |
| Full manuscript matrix | `eval_gani_experiments.py`, `eval_rim_experiments.py`, `eval_admm_experiments.py` | Aggregate test summaries across conditions. |
| Hyperparameter selection | W&B configs plus `determine_admm_iterations.py` | Validation-selected ADMM parameters and iteration counts. |
| Runtime | `time_inference_experiments.py` | Synchronized single-image timing table. |
| Trajectory visualization | `visualize_rim_prim_rollouts.py` | Step-by-step RIM versus PRIM SVG figures. |

## Artifact boundaries

The Git repository contains source, lightweight configs, manuscript material, and curated benchmark tables. Downloaded datasets, training runs, caches, and model binaries are external artifacts. This boundary keeps cloning fast while allowing checkpoints to be versioned independently through a release or artifact registry.
