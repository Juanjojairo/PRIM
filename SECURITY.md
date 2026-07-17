# Security policy

## Supported code

Security fixes are applied to the active development branch. Research checkpoints, historical experiment artifacts, and archived manuscript builds are not treated as maintained software releases unless explicitly attached to a versioned release.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting feature when available. Do not disclose credentials, private W&B links, access tokens, sensitive datasets, or exploit details in a public issue.

A useful report includes:

- the affected commit and file;
- a minimal reproduction;
- expected impact;
- environment details;
- a proposed mitigation, if known.

## Credential handling

This project must not contain embedded credentials. Weights & Biases authentication is expected to come from one of these standard mechanisms:

```bash
wandb login
```

or, in automated environments, a protected `WANDB_API_KEY` secret. Never place a real value in `.env.example`, source files, notebooks, logs, issue reports, or pull requests.

If a credential is committed accidentally:

1. revoke or rotate it immediately;
2. remove it from the current tree;
3. purge it from Git history before public release;
4. inspect forks, caches, CI logs, and experiment artifacts;
5. enable secret scanning on the GitHub repository.

Removing a key from the latest commit does not invalidate a credential that has already been exposed.

## Research artifacts

Only publish datasets and checkpoints for which redistribution is permitted. Record provenance, license, checksums, model architecture metadata, and the exact source commit used to generate each artifact.
