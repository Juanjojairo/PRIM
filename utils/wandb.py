"""Weights & Biases helpers shared across training and evaluation scripts."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import wandb


_WANDB_ARG_NAMES = {
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_group",
    "wandb_job_type",
    "wandb_tags",
    "wandb_dir",
    "wandb_mode",
}
_WANDB_MAX_TAG_LENGTH = 64


def add_wandb_args(
    parser: argparse.ArgumentParser,
    *,
    default_project: str = "peadmm-rim",
    default_job_type: str | None = None,
) -> argparse.ArgumentParser:
    """Attach a consistent set of wandb CLI flags to a parser."""
    parser.add_argument("--wandb-project", type=str, default=default_project)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default=default_job_type)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-dir", type=Path, default=Path("results/wandb"))
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline"],
        help="Use offline mode to cache runs locally when validating without internet.",
    )
    return parser


def _to_serializable(value: Any) -> Any:
    """Convert common Python values into JSON-friendly config entries."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    return value


def namespace_to_config(
    args: argparse.Namespace,
    *,
    exclude: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Serialize argparse arguments into a wandb config dictionary."""
    excluded = set(exclude or [])
    excluded.update(_WANDB_ARG_NAMES)

    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in excluded:
            continue
        config[key] = _to_serializable(value)
    return config


def parse_wandb_tags(
    raw_tags: str | None,
    *,
    extra_tags: Sequence[str] | None = None,
) -> list[str]:
    """Merge comma-separated CLI tags with extra inferred tags."""
    tags = [item.strip() for item in (raw_tags or "").split(",") if item.strip()]
    if extra_tags is not None:
        tags.extend(item for item in extra_tags if item)
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in tags:
        normalized = _slugify(tag)
        if len(normalized) > _WANDB_MAX_TAG_LENGTH:
            normalized = normalized[:_WANDB_MAX_TAG_LENGTH].rstrip("-") or normalized[:_WANDB_MAX_TAG_LENGTH]
        if not normalized:
            continue
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def _slugify(value: str) -> str:
    """Generate a safe wandb artifact name."""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower())
    slug = slug.strip("-")
    return slug or "artifact"
class WandbLogger:
    """Thin wrapper around an active wandb run."""

    def __init__(self, *, run: Any | None = None, wandb_module: Any | None = None) -> None:
        self.run = run
        self._wandb = wandb_module

    @property
    def enabled(self) -> bool:
        """Return whether a real wandb run is active."""
        return self.run is not None and self._wandb is not None

    def log(self, values: Mapping[str, Any], *, step: int | None = None) -> None:
        """Log scalar values to the current run."""
        if not self.enabled:
            return
        payload = {key: value for key, value in values.items() if value is not None}
        if not payload:
            return
        self.run.log(payload, step=step)

    def summary(self, values: Mapping[str, Any]) -> None:
        """Store final metrics in the run summary."""
        if not self.enabled:
            return
        for key, value in values.items():
            if value is not None:
                self.run.summary[key] = value

    def log_image(
        self,
        key: str,
        path: str | Path,
        *,
        step: int | None = None,
        caption: str | None = None,
    ) -> None:
        """Upload an image file to wandb if it exists."""
        if not self.enabled:
            return
        image_path = Path(path)
        if not image_path.exists():
            return
        self.run.log({key: self._wandb.Image(str(image_path), caption=caption)}, step=step)

    def log_table(
        self,
        key: str,
        path: str | Path,
        *,
        step: int | None = None,
    ) -> None:
        """Upload a CSV file as a wandb table."""
        if not self.enabled:
            return
        table_path = Path(path)
        if not table_path.exists():
            return

        import pandas as pd

        dataframe = pd.read_csv(table_path)
        self.run.log({key: self._wandb.Table(dataframe=dataframe)}, step=step)

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_name: str | None = None,
        artifact_type: str = "output",
        aliases: Sequence[str] | None = None,
    ) -> None:
        """Persist a file or directory as a named wandb artifact."""
        if not self.enabled:
            return
        artifact_path = Path(path)
        if not artifact_path.exists():
            return

        name = artifact_name or _slugify(f"{self.run.project}-{self.run.id}-{artifact_path.stem}")
        artifact = self._wandb.Artifact(name=name, type=artifact_type)
        if artifact_path.is_dir():
            artifact.add_dir(str(artifact_path), name=artifact_path.name)
        else:
            artifact.add_file(str(artifact_path), name=artifact_path.name)
        self.run.log_artifact(artifact, aliases=list(aliases or []))

    def finish(self) -> None:
        """Close the active wandb run."""
        if self.enabled:
            self.run.finish()


def init_wandb_run(
    args: argparse.Namespace,
    *,
    config: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
) -> WandbLogger:
    """Initialize wandb and return a logger wrapper."""
    wandb_dir = Path(getattr(args, "wandb_dir", Path("results/wandb")))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = wandb_dir / "cache"
    config_dir = wandb_dir / "config"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_DIR"] = str(wandb_dir)
    os.environ["WANDB_CACHE_DIR"] = str(cache_dir)
    os.environ["WANDB_CONFIG_DIR"] = str(config_dir)

    wandb.login(key="8024b52ebbe0c11ede163101eb790705ca7880e6")
    run = wandb.init(
        project=getattr(args, "wandb_project", None),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None),
        group=getattr(args, "wandb_group", None),
        job_type=getattr(args, "wandb_job_type", None),
        tags=parse_wandb_tags(getattr(args, "wandb_tags", ""), extra_tags=tags),
        mode=getattr(args, "wandb_mode", "online"),
        dir=str(wandb_dir),
        config=dict(config or {}),
    )
    return WandbLogger(run=run, wandb_module=wandb)
