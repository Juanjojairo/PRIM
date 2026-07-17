# Benchmark archive

This directory contains lightweight, byte-for-byte copies of aggregate experimental records from the local research workspace. It preserves the evidence behind the README tables without committing the multi-gigabyte training and W&B directories.

## Contents

```text
benchmarks/
├── metrics/
│   ├── rim_test_summary.csv
│   ├── gani_test_summary.csv
│   └── admm_group{1,2,3}_test_summary.csv
├── runtime/
│   ├── single_sample_inference_timing.csv
│   ├── single_sample_inference_timing.md
│   └── paper_inference_runtime_table.md
└── parameters/
    ├── eadmm_peadmm_sweep_params.txt
    └── eadmm_peadmm_sweep_params_with_iterations.txt
```

## Interpretation

- Metric summaries report results on the full 10,000-image MNIST test split.
- The RIM summary contains standard RIM and PRIM conditions.
- ADMM results are split across three files because the local evaluation was executed in groups.
- Runtime values are hardware-specific measurements and must be reported with the GPU/software environment.
- Parameter files preserve validation-selected `gamma`, `beta`, `sigma`, and iteration counts used by aggregate ADMM evaluation.

These files are reports, not test fixtures. Regenerated values may differ across dependency versions, hardware, or nondeterministic GPU kernels. New reports should include the source commit, resolved environment, seed, sample count, and checkpoint checksums.

## Provenance

The files were exported from their corresponding experiment-output paths without editing their contents. `MANIFEST.sha256` at the repository root provides integrity hashes for the complete clean distribution.
