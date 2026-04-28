#!/usr/bin/env python3
"""Verify ``tests/fixtures/MANIFEST.toml`` matches the on-disk fixtures.

The GEMMA reference fixtures in ``tests/fixtures/`` are the load-bearing
parity baseline for JAMMA. They were generated from specific GEMMA binary
versions with specific commands; silent edits or bit-rot would invalidate
every parity assertion downstream.

This gate hashes every git-tracked fixture file under ``tests/fixtures/``
and compares against the recorded SHA-256s in ``MANIFEST.toml``. It fails
on:

* Hash mismatch (file edited after manifest generation)
* File present on disk but missing from manifest (untracked addition)
* File in manifest but missing from disk

To intentionally update fixtures: regenerate them, then run
``python scripts/regenerate_fixture_manifest.py`` to refresh the manifest,
and commit both the new fixture files and the manifest in the same commit.

Usage::

    python scripts/check_fixture_manifest.py        # check
    python scripts/check_fixture_manifest.py --list # show all tracked files
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"
MANIFEST_PATH = FIXTURES_DIR / "MANIFEST.toml"


def sha256_of(path: Path) -> str:
    """Stream-hash a file in 1 MiB chunks (works for large fixtures)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_fixtures() -> list[Path]:
    """Return git-tracked files under ``tests/fixtures/`` (excluding the manifest)."""
    result = subprocess.run(
        ["git", "ls-files", "tests/fixtures/"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        rel = line.strip()
        if not rel or rel.endswith("MANIFEST.toml"):
            continue
        paths.append(REPO_ROOT / rel)
    return sorted(paths)


def load_manifest() -> dict[str, dict[str, str]]:
    """Return ``{relative_path: {sha256, ...}}`` from MANIFEST.toml."""
    if not MANIFEST_PATH.exists():
        return {}
    with MANIFEST_PATH.open("rb") as f:
        data = tomllib.load(f)
    return data.get("file", {})


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for p in tracked_fixtures():
            print(p.relative_to(REPO_ROOT))
        return 0

    manifest = load_manifest()
    if not manifest:
        sys.stderr.write(
            f"FAIL: {MANIFEST_PATH.relative_to(REPO_ROOT)} is missing or empty.\n"
            "Run `python scripts/regenerate_fixture_manifest.py` to create it.\n"
        )
        return 1

    on_disk = {p.relative_to(REPO_ROOT).as_posix(): p for p in tracked_fixtures()}

    failures: list[str] = []

    # Manifest entries that no longer exist on disk.
    for rel in manifest:
        if rel not in on_disk:
            failures.append(f"  - {rel}: in manifest but not in tests/fixtures/")

    # On-disk files missing from the manifest.
    for rel in on_disk:
        if rel not in manifest:
            failures.append(f"  + {rel}: on disk but absent from manifest")

    # Hash mismatches.
    for rel, path in on_disk.items():
        if rel not in manifest:
            continue
        expected = manifest[rel].get("sha256", "")
        actual = sha256_of(path)
        if expected != actual:
            failures.append(
                f"  ! {rel}: sha256 drift\n"
                f"      expected={expected}\n"
                f"      actual  ={actual}"
            )

    if failures:
        sys.stderr.write(
            "Fixture manifest check FAILED.\n\n" + "\n".join(failures) + "\n\n"
            "If you intentionally regenerated fixtures, run:\n"
            "    python scripts/regenerate_fixture_manifest.py\n"
            "and commit the updated MANIFEST.toml together with the new files.\n"
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
