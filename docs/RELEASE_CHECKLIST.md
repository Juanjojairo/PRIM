# Public release checklist

Complete every blocking item before changing the GitHub repository visibility to public or publishing a release.

## Security

- [ ] All credentials are active only where required and have an established rotation procedure.
- [ ] The complete Git history has been scanned, not only the latest working tree.
- [ ] `git grep`, GitHub secret scanning, and an independent secret scanner report no credentials.
- [ ] W&B, CI, and deployment credentials are stored as protected secrets.
- [ ] Logs, metadata, PDFs, and notebooks have been checked for personal or sensitive information.

## Legal and attribution

- [ ] The repository owner has selected and added a `LICENSE` file.
- [ ] Dataset, dependency, figure, and borrowed-code licenses permit the intended distribution.
- [ ] `CITATION.cff`, manuscript authors, affiliations, and repository URL are correct.
- [ ] Funding and third-party acknowledgements are complete.

## Reproducibility

- [ ] Python and dependency versions are captured in a release environment lock.
- [ ] The SPI smoke test passes for the reported compression ratios.
- [ ] Training/evaluation commands match the current CLI.
- [ ] Full-test tables use 10,000 examples and hyperparameters were selected on validation data.
- [ ] Checkpoints include architecture metadata, source commit, condition, seed, and checksum.
- [ ] Runtime claims identify GPU, CUDA, PyTorch, warm-up, synchronization, and sample protocol.
- [ ] External artifact links are durable and access permissions have been tested anonymously.

## Repository quality

- [ ] `README.md`, `docs/INDEX.md`, and `docs/REPRODUCIBILITY.md` render correctly on GitHub.
- [ ] Local links and figures resolve with case-sensitive paths.
- [ ] No dataset downloads, checkpoints, caches, W&B runs, or LaTeX intermediates are staged accidentally.
- [ ] Issue templates, pull request template, contributing guide, and security policy are present.
- [ ] `MANIFEST.sha256` matches the distributed files.
- [ ] The default branch and branch-protection rules are configured.

## Release artifacts

- [ ] Essential checkpoints are attached through GitHub Releases, Git LFS, W&B Artifacts, or an archival repository.
- [ ] Every artifact has a SHA-256 checksum and a short model/data card.
- [ ] The release notes describe experimental scope, known limitations, and compatibility.
- [ ] A clean clone can follow the documented quick start without access to private local paths.

The lack of a license and any active credential exposure are release blockers, not optional improvements.
