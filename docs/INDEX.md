# Documentation center

Use this page as the navigation hub for the PRIM repository.

| Document | Audience | Purpose |
|:--|:--|:--|
| [Project README](../README.md) | Everyone | Method overview, reported results, installation, and quick start. |
| [Reproducibility guide](REPRODUCIBILITY.md) | Researchers | Complete training/evaluation order, sweeps, parameters, and artifact policy. |
| [Project map](PROJECT_MAP.md) | Developers | Architecture, ownership of modules, data flow, and entry-point families. |
| [Benchmark archive](benchmarks/README.md) | Reviewers | Curated copies of aggregate metrics, timing results, and ADMM parameters. |
| [Release checklist](RELEASE_CHECKLIST.md) | Maintainers | Security, licensing, validation, and publication gates. |
| [Contributing guide](../CONTRIBUTING.md) | Contributors | Development practices and pull request validation. |
| [Security policy](../SECURITY.md) | Users and maintainers | Private reporting and credential handling. |
| [Citation metadata](../CITATION.cff) | Researchers | Machine-readable software citation. |

## Research manuscript

The current manuscript source, bibliography, and curated figures are under [`manuscript/`](manuscript/). LaTeX build intermediates and stale compiled PDFs are intentionally excluded from the clean distribution.

## Fast navigation by goal

- **Understand PRIM:** README → Project map → manuscript.
- **Reproduce a table:** Reproducibility guide → benchmark archive → relevant evaluator.
- **Train a new condition:** Reproducibility guide → condition-specific entry point → W&B sweep config.
- **Contribute code:** Contributing guide → issue template → pull request template.
- **Publish a release:** Release checklist → security policy → citation metadata.
