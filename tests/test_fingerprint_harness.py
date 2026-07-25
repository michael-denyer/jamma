"""The fingerprint harness records what it claims to record.

``scripts/lmm_accel_fingerprint.py`` is the pin the whole ``_lmm_accel.c``
refactoring programme rests on, and a pin that quietly stops covering things
looks exactly like a clean refactor. Its own docstring names the subtle failure:
``jamma.lmm.compute_numpy`` copies each C function into a module-level alias at
import, so wrapping only the extension's attributes misses almost every call
site. If that alias sweep broke, the harness would still produce a file, still
exit zero, and still compare equal.

Runs pytest in a subprocess because the recorder installs itself at import time
off an environment variable, and it rebinds module globals process-wide.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.tier1, pytest.mark.slow]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> list[tuple[str, str, str]]:
    """Drive the whole accel suite under the recorder, as the CI gate does."""
    out = tmp_path_factory.mktemp("fingerprint") / "records.txt"
    env = {**os.environ, "JAMMA_FINGERPRINT_OUT": str(out)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/lmm_accel/",
            "-n0",
            "--randomly-seed=1234",
            "-p",
            "scripts.lmm_accel_fingerprint",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if not out.exists():
        pytest.fail(
            f"recorder wrote no file (exit {result.returncode}).\n"
            f"stdout tail:\n{result.stdout[-2000:]}"
        )
    return [
        tuple(line.split("\t"))  # type: ignore[misc]
        for line in out.read_text().splitlines()
        if line.strip()
    ]


def test_the_recorder_produces_records(recorded):
    assert recorded, "the recorder installed but captured nothing"


def test_every_record_is_well_formed(recorded):
    for record in recorded:
        assert len(record) == 3, f"malformed record: {record!r}"
        name, args_digest, result = record
        assert name
        assert not name.startswith("_")
        assert args_digest
        assert result


def test_calls_routed_through_compute_numpy_aliases_are_captured(recorded):
    """The alias sweep is what catches nearly every real call site.

    ``compute_numpy`` binds its own module-level name to each C function at
    import. Wrapping only the extension's attributes leaves those bindings
    pointing at the unwrapped originals, so the calls that matter go unrecorded
    while the harness still reports success.
    """
    import jamma.lmm.compute_numpy as cn

    aliased = {
        name.lstrip("_")
        for name, value in vars(cn).items()
        if name.startswith("_compute_") and callable(value)
    }
    if not aliased:
        pytest.skip("compute_numpy exposes no C aliases in this build")

    recorded_names = {name for name, _, _ in recorded}
    assert recorded_names & aliased, (
        "no recorded call went through a compute_numpy alias. Either the "
        f"alias sweep in _install() broke, or none of {sorted(aliased)} was "
        f"exercised. Recorded: {sorted(recorded_names)}"
    )


def test_the_suite_covers_most_of_the_extension(recorded):
    """A collapse to a handful of entry points is the signature of a broken sweep.

    The suite drives roughly 30 of the extension's public callables. The bar is
    deliberately well under that, so adding or retiring a kernel family does not
    fail this, but losing the alias sweep does.
    """
    import jamma.lmm._lmm_accel as accel

    public = {
        name
        for name in dir(accel)
        if not name.startswith("_") and callable(getattr(accel, name))
    }
    covered = {name for name, _, _ in recorded}
    assert len(covered) >= 10, (
        f"only {len(covered)} of {len(public)} public callables recorded: "
        f"{sorted(covered)}. The alias sweep in _install() has likely broken."
    )
