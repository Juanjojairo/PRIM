"""Generate or verify the SHA-256 manifest for a clean repository release."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


MANIFEST_NAME = "MANIFEST.sha256"
IGNORED_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def iter_release_files(root: Path) -> list[Path]:
    """Return every release file except the manifest and ignored local state."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.as_posix() == MANIFEST_NAME:
            continue
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def sha256(path: Path) -> str:
    """Hash a file without loading it completely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> str:
    """Build deterministic GNU-compatible checksum lines."""
    return "".join(
        f"{sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in iter_release_files(root)
    )


def write_manifest(root: Path) -> None:
    """Write a fresh manifest at the repository root."""
    manifest_path = root / MANIFEST_NAME
    manifest_path.write_text(build_manifest(root), encoding="utf-8", newline="\n")
    print(f"Wrote {manifest_path}")


def verify_manifest(root: Path) -> None:
    """Verify hashes and ensure the manifest covers the full release tree."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    expected = manifest_path.read_text(encoding="utf-8")
    actual = build_manifest(root)
    if expected != actual:
        raise RuntimeError(
            "Manifest verification failed. Regenerate it with "
            "`python tools/release_manifest.py`."
        )
    print(f"Verified {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the existing manifest instead of regenerating it.",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.check:
        verify_manifest(root)
    else:
        write_manifest(root)


if __name__ == "__main__":
    main()
