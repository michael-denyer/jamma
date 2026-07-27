"""Self-test for the GEMMA fixture manifest gate.

The fixtures in ``tests/fixtures/`` are the load-bearing GEMMA parity
baseline. ``scripts/check_fixture_manifest.py`` enforces that on-disk
files match the recorded SHA-256s in ``MANIFEST.toml``. This test runs
the gate in-process for fast feedback (the pre-commit hook is the slow
path that catches incoming drift).

Failure here means one of:

* A fixture was edited but ``MANIFEST.toml`` wasn't regenerated.
* A new fixture was added without an entry in ``MANIFEST.toml``.
* A manifest entry refers to a fixture that no longer exists.

Fix: run ``python scripts/regenerate_fixture_manifest.py`` and commit
both the fixture changes and the regenerated manifest in the same commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier0


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = _REPO_ROOT / "scripts"


def _load_checker():
    """Import the checker from ``scripts/``, which is not a package.

    Same sys.path dance as tests/test_check_c_extension_freshness.py, rather
    than spec_from_file_location, so the module's attributes stay visible to
    the type checker (``scripts`` is on pyrefly's search-path).
    """
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        import check_fixture_manifest
    finally:
        if sys.path and sys.path[0] == str(_SCRIPT_DIR):
            sys.path.pop(0)
    return check_fixture_manifest


def test_fixture_manifest_matches_disk(capsys: pytest.CaptureFixture[str]) -> None:
    """Repository state must satisfy the gate (no drift, no missing entries)."""
    checker = _load_checker()
    rc = checker.main([])
    captured = capsys.readouterr()
    assert rc == 0, (
        f"Fixture manifest check failed.\n"
        f"stdout={captured.out!r}\nstderr={captured.err!r}"
    )


def test_manifest_file_exists() -> None:
    """``MANIFEST.toml`` exists; the gate would refuse to run otherwise."""
    checker = _load_checker()
    assert checker.MANIFEST_PATH.exists(), (
        f"Missing manifest at {checker.MANIFEST_PATH}; run "
        "scripts/regenerate_fixture_manifest.py to create it."
    )


def test_every_tracked_fixture_has_manifest_entry() -> None:
    """No silent drift: a new fixture without a manifest entry must fail."""
    checker = _load_checker()
    manifest = checker.load_manifest()
    on_disk = {
        p.relative_to(checker.REPO_ROOT).as_posix() for p in checker.tracked_fixtures()
    }
    missing = sorted(on_disk - manifest.keys())
    assert not missing, (
        "Fixtures committed without manifest entries:\n  "
        + "\n  ".join(missing)
        + "\n\nRun scripts/regenerate_fixture_manifest.py."
    )


def test_drift_is_detected(tmp_path: Path) -> None:
    """End-to-end check: a hash-mismatched manifest must make the gate fail.

    Validates the gate's *failure* path, complementing the success-path test
    above. We synthesize a manifest pointing at a real fixture but with the
    wrong sha256 and confirm the checker exits non-zero.
    """
    checker = _load_checker()
    fixtures = checker.tracked_fixtures()
    if not fixtures:
        pytest.skip("no tracked fixtures to drift-test against")
    target = fixtures[0]
    rel = target.relative_to(checker.REPO_ROOT).as_posix()

    bogus_manifest = tmp_path / "MANIFEST.toml"
    bogus_manifest.write_text(
        f'[file."{rel}"]\nsha256 = "0" * 64\n'.replace('"0" * 64', '"' + "0" * 64 + '"')
    )

    # Patch MANIFEST_PATH to point at the bogus file for one call.
    real_path = checker.MANIFEST_PATH
    checker.MANIFEST_PATH = bogus_manifest
    try:
        rc = checker.main([])
    finally:
        checker.MANIFEST_PATH = real_path
        # Restore stderr — checker writes to it on failure.
        sys.stderr.flush()

    assert rc != 0, "Drifted manifest should make the gate fail (returncode 0 = bug)"
