"""Self-tests for the requires_c marker and no_c_kernels fixture.

D3 replaced 26 `skipif(compute_numpy._accel is None, ...)` decorators, five
module-level availability flags, fourteen inline `pytest.skip` calls, and
roughly two dozen hand-rolled `orig = ...; try: ... finally:` hold-outs with
one marker and one fixture. These pin the two halves of that seam: the
marker skips exactly when the extension is unavailable, and the fixture
actually flips `accel.available()` for the span of the test that uses it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jamma.lmm import accel
from tests.conftest import requires_c

pytestmark = pytest.mark.tier0

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requires_c_marker_reflects_accel_available() -> None:
    """The marker's skip condition is exactly `not accel.available()`.

    `requires_c` is built once at collection from `accel.available()`, the
    same call a test would make by hand — this pins that pytest actually
    receives that boolean rather than a stale or inverted one.
    """
    skipif_mark = requires_c.mark
    assert skipif_mark.name == "skipif"
    condition = skipif_mark.args[0]
    assert condition is (not accel.available())


def test_no_c_kernels_fixture_clears_availability(no_c_kernels: None) -> None:
    """Inside the fixture, `accel.available()` reports False regardless of build."""
    assert accel.available() is False


def test_no_c_kernels_restores_after_the_test() -> None:
    """The fixture's teardown restores whatever `accel._accel` held before it ran.

    Runs the fixture-consuming case in a fresh subprocess so this test's own
    result is not sensitive to xdist worker ordering, then checks the marker
    behaves identically before and after: `requires_c` on a probe function
    collected before and after the fixtured test see the same availability.
    """
    script = textwrap.dedent("""
        import pytest
        from jamma.lmm import accel

        pytestmark = pytest.mark.tier0

        _BEFORE = accel.available()

        def test_a(no_c_kernels):
            assert accel.available() is False

        def test_b():
            assert accel.available() is _BEFORE
        """)
    test_dir = Path(__file__).resolve().parent
    tmp_test = test_dir / "test_planted_no_c_kernels_restore.py"
    tmp_test.write_text(script)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(tmp_test), "-o", "addopts="],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        tmp_test.unlink()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout, result.stdout + result.stderr


def test_requires_c_skips_when_extension_unavailable() -> None:
    """A `@requires_c` test collected under a forced-absent extension skips.

    Drives the marker through a real pytest subprocess with
    JAMMA_FORCE_NUMPY_FALLBACK set, rather than monkeypatching `accel` in
    this process, because `requires_c` is built once at collection from
    `accel.available()` and this process already imported it.
    """
    script = textwrap.dedent("""
        import pytest
        from tests.conftest import requires_c

        pytestmark = pytest.mark.tier0

        @requires_c
        def test_needs_c():
            assert False, "should have skipped"
        """)
    test_dir = Path(__file__).resolve().parent
    tmp_test = test_dir / "test_planted_requires_c_forced_absent.py"
    tmp_test.write_text(script)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(tmp_test), "-o", "addopts="],
            cwd=_REPO_ROOT,
            env={**os.environ, "JAMMA_FORCE_NUMPY_FALLBACK": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        tmp_test.unlink()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout, result.stdout + result.stderr
