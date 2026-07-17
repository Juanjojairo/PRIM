# Contributing

Thank you for helping improve PRIM. This is research software, so reproducibility and a clear experimental scope matter as much as implementation quality.

## Before opening an issue

- Search existing issues for the same question or failure.
- Identify the task (`spi` or `SR`), condition, command, and commit SHA.
- Include the Python, PyTorch, torchvision, CUDA, and GPU versions when relevant.
- Reduce runtime failures with `--max-train-batches`, `--max-val-batches`, `--max-batches`, or `--max-examples` when the entry point supports them.
- Never attach API keys, private W&B links, credentials, or proprietary data.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Making a change

1. Create a focused branch from the intended base branch.
2. Keep unrelated refactors out of the same pull request.
3. Preserve deterministic data splits and seed handling.
4. Document new CLI flags and output artifacts.
5. Add or update a smoke test when changing an operator, metric, checkpoint format, or reconstruction path.
6. Do not commit generated datasets, checkpoints, W&B state, result directories, or LaTeX build products.

## Minimum validation

Run syntax compilation for the source tree:

```bash
python -m compileall -q \
  -x '(^|/)(results|wandb|history|data|docs|\.git)(/|$)' .
```

Validate the SPI forward/adjoint implementation:

```bash
python -m ops.forward_models --cr 0.1
python -m ops.forward_models --cr 0.01
```

For training or evaluation changes, also run the smallest relevant command using the available batch/example limiting flags. State exactly what you ran in the pull request.

## Experimental changes

A result-changing pull request should describe:

- the hypothesis;
- the baseline and changed configuration;
- the data split and number of examples;
- the seed or seeds;
- checkpoint selection criteria;
- PSNR, SSIM, MSE, and MAE where applicable;
- hardware and timing protocol for performance claims;
- links to non-sensitive experiment artifacts.

Do not select hyperparameters on the test split. If a value comes from an exploratory subset, label it clearly and do not present it as a full-test result.

## Pull request checklist

- [ ] The change has one clear purpose.
- [ ] Source files compile.
- [ ] Relevant smoke/integration checks pass.
- [ ] New behavior and CLI options are documented.
- [ ] Generated or sensitive files are not included.
- [ ] Result claims state the split, sample count, seed, and hardware.
- [ ] Checkpoint compatibility or migration concerns are documented.

By contributing, you agree that your contribution may be distributed under the repository's eventual project license. The repository owner must select and add that license before accepting external contributions for redistribution.
