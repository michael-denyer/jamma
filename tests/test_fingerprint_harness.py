"""The fingerprint harness records what it claims to record.

``scripts/lmm_accel_fingerprint.py`` is the pin the whole ``_lmm_accel.c``
refactoring programme rests on, and a pin that quietly stops covering things
looks exactly like a clean refactor. The subtle failure is a module that binds
a C function to its own global at import: the recorder wraps the attribute on
the extension, the copy keeps the unwrapped original, and calls through it go
unrecorded while the harness still produces a file, still exits zero, and still
compares equal.

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


def test_no_jamma_module_holds_a_raw_c_callable():
    """Wrapping the extension's attributes is only enough while nothing copies them.

    ``_install`` replaces each callable on ``_lmm_accel`` itself, which covers
    every caller that looks the symbol up at call time. A module that binds one
    to its own global at import keeps the unwrapped original, and its calls go
    unrecorded while the harness still writes a file and still exits zero.

    ``compute_numpy`` used to do exactly that for thirty symbols, which is why
    ``_install`` carried a second pass sweeping ``sys.modules`` by ``id()``.
    Reintroducing such a copy must fail here rather than silently shrink what
    the fingerprint gate compares.
    """
    import sys

    import jamma.lmm._lmm_accel as accel
    import jamma.lmm.chunk_dispatch
    import jamma.lmm.chunk_workspaces
    import jamma.lmm.compute_numpy
    import jamma.lmm.loco
    import jamma.lmm.runner_numpy
    import jamma.lmm.runner_numpy_streaming  # noqa: F401

    by_id = {
        id(attr): name
        for name in dir(accel)
        if not name.startswith("_")
        and callable(attr := getattr(accel, name))
        and not isinstance(attr, type)
    }
    copies = [
        f"{modname}.{attr} aliases {by_id[id(value)]}"
        for modname, module in list(sys.modules.items())
        if modname.startswith("jamma.") and modname != accel.__name__
        for attr, value in list(vars(module).items())
        if id(value) in by_id
    ]
    assert not copies, (
        "a module global holds a raw C callable, so the fingerprint recorder "
        "cannot see calls through it. Either stop copying the symbol and read "
        "it off the extension at call time, or restore the sys.modules sweep "
        f"in _install(). Copies: {sorted(copies)}"
    )


def test_the_suite_covers_most_of_the_extension(recorded):
    """A collapse to a handful of entry points is the signature of a broken recorder.

    The suite drives roughly 30 of the extension's public callables. The bar is
    deliberately well under that, so adding or retiring a kernel family does not
    fail this, but losing most of the wrapping does.
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
        f"{sorted(covered)}. The wrapping in _install() has likely broken."
    )
