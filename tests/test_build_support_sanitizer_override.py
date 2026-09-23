"""Tests for the JAMMA_SANITIZE flags ``resolve_flags`` resolves and their
forwarding through ``resolve_cflags_for`` and ``execute_build``.

Covers the sanitizer flag injection seam.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from jamma._build_support.build_execution import Toolchain, execute_build
from jamma._build_support.build_models import (
    BASE_CFLAGS,
    JLINALG_SPEC,
    LAPACK_CFLAGS,
    LINK_LIBS,
    LMM_ACCEL_SPEC,
    BuildReport,
    ResolvedFlags,
    resolve_cflags_for,
    resolve_flags,
)

pytestmark = pytest.mark.tier0

_SAN_CFLAGS = ("-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-O1")


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _wheel_flags(env: dict[str, str]) -> ResolvedFlags:
    return resolve_flags(LMM_ACCEL_SPEC, dev_mode=False, system="Linux", env=env)


# ---------------------------------------------------------------------------
# resolve_flags — JAMMA_SANITIZE injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "  ", "0"])
def test_sanitizer_off_values_add_nothing(value):
    """Unset, empty, whitespace, and "0" all leave the flags unsanitized."""
    env = (
        {"CFLAGS": "-DUSER"}
        if value is None
        else {"CFLAGS": "-DUSER", "JAMMA_SANITIZE": value}
    )
    flags = _wheel_flags(env)
    assert flags.base_extra == ("-DUSER",)
    assert flags.lapack_extra == ()
    assert flags.link_libs == LINK_LIBS


def test_address_undefined_appends_sanitizer_flags():
    """address,undefined: base and LAPACK cflags get -fsanitize=...,
    -fno-omit-frame-pointer, -O1 after any user flags; the link gets the same
    -fsanitize=... after -lm.
    """
    flags = _wheel_flags({"CFLAGS": "-DUSER", "JAMMA_SANITIZE": "address,undefined"})
    assert flags.base_extra == ("-DUSER", *_SAN_CFLAGS)
    assert flags.lapack_extra == _SAN_CFLAGS
    assert flags.link_libs == ("-lm", "-fsanitize=address,undefined")


def test_address_only():
    """JAMMA_SANITIZE=address (no comma): -fsanitize=address only."""
    flags = _wheel_flags({"JAMMA_SANITIZE": " address "})
    assert flags.lapack_extra == (
        "-fsanitize=address",
        "-fno-omit-frame-pointer",
        "-O1",
    )
    assert flags.link_libs == ("-lm", "-fsanitize=address")


def test_sanitizer_follows_dev_flags_and_sentinel():
    """In dev mode the sanitizer flags come after -march=native and the
    sentinel macro, so the trailing -O1 still wins."""
    flags = resolve_flags(
        LMM_ACCEL_SPEC,
        dev_mode=True,
        system="Darwin",
        env={"JAMMA_SENTINEL_UB": "1", "JAMMA_SANITIZE": "address,undefined"},
    )
    assert flags.base_extra == ("-march=native", "-DJAMMA_SENTINEL_UB", *_SAN_CFLAGS)
    assert flags.platform_link == ("-undefined", "dynamic_lookup")


# ---------------------------------------------------------------------------
# resolve_cflags_for — lapack_extra integration
# ---------------------------------------------------------------------------


def _flags(*, base_extra=(), lapack_extra=()) -> ResolvedFlags:
    return ResolvedFlags(
        base_extra=base_extra, lapack_extra=lapack_extra, platform_link=(), link_libs=()
    )


def test_resolve_cflags_lapack_path_appends_lapack_extra():
    """LAPACK source: LAPACK_CFLAGS, then lapack_extra, then -I includes.
    Trailing -O1 wins over LAPACK_CFLAGS' -O2 (last -O on the command line).
    """
    flags = resolve_cflags_for(
        _flags(lapack_extra=("-fsanitize=address", "-O1")),
        ["/usr/include"],
        lapack=True,
    )
    assert flags == [*LAPACK_CFLAGS, "-fsanitize=address", "-O1", "-I/usr/include"]
    assert [f for f in flags if f.startswith("-O")][-1] == "-O1"


def test_resolve_cflags_lapack_path_without_extra():
    """No lapack_extra: result equals [*LAPACK_CFLAGS, *includes]."""
    flags = resolve_cflags_for(_flags(), ["/usr/include"], lapack=True)
    assert flags == [*LAPACK_CFLAGS, "-I/usr/include"]


def test_resolve_cflags_baseline_path_unaffected_by_lapack_extra():
    """A non-LAPACK source must NOT receive lapack_extra."""
    flags = resolve_cflags_for(
        _flags(base_extra=("-DUSER",), lapack_extra=("-fsanitize=address",)),
        [],
        lapack=False,
    )
    assert "-fsanitize=address" not in flags
    splice_idx = BASE_CFLAGS.index("-fno-finite-math-only")
    assert flags == [*BASE_CFLAGS[:splice_idx], "-DUSER", *BASE_CFLAGS[splice_idx:]]


# ---------------------------------------------------------------------------
# execute_build — forwarding of lapack_extra through _compile_sources
# ---------------------------------------------------------------------------


def test_execute_build_forwards_lapack_extra(monkeypatch, tmp_path):
    """execute_build must put lapack_extra on the LAPACK source's compile
    command line and nowhere else.
    """
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(
        "jamma._build_support.build_execution.subprocess.run",
        _fake_run,
    )
    spec = replace(JLINALG_SPEC, sources=("platform.c", "eigh.c"))
    toolchain = Toolchain(
        cc_cmd="cc",
        cc_extra=(),
        python_inc="",
        numpy_inc="",
        system="Linux",
        omp_compile=(),
        omp_link=(),
    )

    result = execute_build(
        spec,
        tmp_path,
        [],
        toolchain,
        _flags(lapack_extra=("-fsanitize=address", "-O1")),
        tmp_path / "out.so",
        tmp_path / "objs",
        BuildReport(detail=lambda _m: None, warn=lambda _m: None),
    )

    assert result.ok
    lapack_compile = next(c for c in calls if str(tmp_path / "eigh.c") in c)
    assert lapack_compile[-6:-4] == ["-fsanitize=address", "-O1"]
    base_compile = next(c for c in calls if str(tmp_path / "platform.c") in c)
    assert "-fsanitize=address" not in base_compile
