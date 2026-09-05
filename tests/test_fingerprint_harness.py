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

import numpy as np
import pytest

from scripts import compare_fingerprints
from scripts import lmm_accel_fingerprint as fingerprint

_REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [pytest.mark.tier0, pytest.mark.slow]


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
    import jamma.lmm.chunk_kernel
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


def test_the_suite_covers_the_whole_extension(recorded):
    """Every public callable the accel suite can reach has a record.

    The bar is the extension's own export set, not a number, so retiring a
    kernel family cannot make this pass vacuously and losing the wrapping
    cannot hide behind a generous margin. ``jamma_sentinel_oob`` exists only
    in sanitizer builds and is driven by tests/test_sanitizer_sentinel.py,
    outside the recorded suite.
    """
    import jamma.lmm._lmm_accel as accel

    public = {
        name
        for name in dir(accel)
        if not name.startswith("_") and callable(getattr(accel, name))
    } - {"jamma_sentinel_oob"}
    covered = {name.split(".", 1)[0] for name, _, _ in recorded}
    assert covered == public, (
        f"recorded {sorted(covered)}, extension exports {sorted(public)}. "
        "The wrapping in _install() has likely broken."
    )


def _record(fn):
    fingerprint._records.clear()
    result = fingerprint._wrap("probe", fn)(np.array([1.0]))
    records = list(fingerprint._records)
    fingerprint._records.clear()
    return result, records


def _as_comparison_input(records):
    grouped = {}
    for line in records:
        name, args_digest, result = line.split("\t")
        grouped.setdefault((name, args_digest), []).append(result)
    return {key: tuple(sorted(values)) for key, values in grouped.items()}


def test_dictionary_fields_are_independent_comparison_keys():
    _, base_records = _record(
        lambda _: {"shared": np.array([1.0]), "removed": np.array([2.0])}
    )
    changed = np.array([1.0])
    changed.view(np.uint64)[0] ^= np.uint64(1)
    _, head_records = _record(lambda _: {"shared": changed, "added": np.array([3.0])})

    base = _as_comparison_input(base_records)
    head = _as_comparison_input(head_records)
    drifted, added, removed = compare_fingerprints.compare(base, head)

    args_digest = fingerprint._digest((np.array([1.0]),), {})
    assert drifted == [("probe.shared", args_digest)]
    assert added == [("probe.added", args_digest)]
    assert removed == [("probe.removed", args_digest)]
    assert all(key[0] != "probe" for key in base | head)


def test_non_dictionary_and_exception_records_keep_their_identity():
    value = np.array([4.0])
    _, records = _record(lambda _: value)
    name, _, result = records[0].split("\t")
    assert name == "probe"
    assert result == fingerprint._digest(value)

    fingerprint._records.clear()

    def fail(_):
        raise ValueError("expected")

    with pytest.raises(ValueError, match="expected"):
        fingerprint._wrap("probe", fail)(np.array([1.0]))
    name, _, result = fingerprint._records.pop().split("\t")
    assert name == "probe"
    assert result == "raise:ValueError"


def test_digest_preserves_dtype_shape_signed_zero_and_nan_payload_bits():
    assert fingerprint._digest(np.array([1.0], dtype=np.float64)) != (
        fingerprint._digest(np.array([1.0], dtype=np.float32))
    )
    assert fingerprint._digest(np.array([1.0, 2.0])) != fingerprint._digest(
        np.array([[1.0, 2.0]])
    )
    assert fingerprint._digest(np.array([0.0])) != fingerprint._digest(np.array([-0.0]))
    nan_a = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    nan_b = np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64)
    assert fingerprint._digest(nan_a) != fingerprint._digest(nan_b)
