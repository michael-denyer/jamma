"""Tests for jamma._build_support.openmp_detect error/timeout branches.

Covers the defensive paths whose whole point is surfacing — not crashing
on — real build-environment failures:
  - ``brew --prefix libomp`` timeout (Darwin)
  - ``brew`` absent from PATH (Darwin)
  - numpy ImportError during the libiomp5 probe (Linux)
  - ``clang`` probe timeout / OSError (Linux)
  - ``clang`` probe shape: no ``-lomp`` at link, omp.h from intel-openmp

These are exactly the paths a regression would silently reintroduce
(hard hang, silent libgomp fallback on ILP64 boxes, crash on broken
clang symlink).
"""

from __future__ import annotations

import subprocess

import pytest

from jamma._build_support.build_models import BuildReport
from jamma._build_support.openmp_detect import (
    _clang_probe_commands,
    _detect_darwin_openmp_flags,
    _find_libiomp5,
    _intel_openmp_include_dir,
    _libiomp5_candidate,
    _openmp_flags_for_libiomp5,
)

pytestmark = pytest.mark.tier0

# ---------------------------------------------------------------------------
# Darwin: brew probe
# ---------------------------------------------------------------------------


def test_darwin_brew_timeout_returns_empty_and_logs():
    """A wedged ``brew`` must not hang the build — the 10s timeout trips and
    the detector returns empty flags with a log explaining why."""
    logs: list[str] = []

    def _fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="brew", timeout=10)

    import jamma._build_support.openmp_detect as mod

    original = mod.subprocess.run
    mod.subprocess.run = _fake_run
    try:
        cflags, lflags = _detect_darwin_openmp_flags(
            BuildReport(detail=logs.append, warn=logs.append)
        )
    finally:
        mod.subprocess.run = original

    assert cflags == []
    assert lflags == []
    assert any("timed out" in msg.lower() for msg in logs), (
        f"timeout must be explained in the log; got {logs!r}"
    )


def test_darwin_brew_not_found_returns_empty_and_logs():
    """On a box without Homebrew, ``brew`` raises FileNotFoundError —
    detector must degrade gracefully, not propagate."""
    logs: list[str] = []

    def _fake_run(*_args, **_kwargs):
        raise FileNotFoundError("brew: command not found")

    import jamma._build_support.openmp_detect as mod

    original = mod.subprocess.run
    mod.subprocess.run = _fake_run
    try:
        cflags, lflags = _detect_darwin_openmp_flags(
            BuildReport(detail=logs.append, warn=logs.append)
        )
    finally:
        mod.subprocess.run = original

    assert cflags == []
    assert lflags == []
    assert any("brew not found" in msg.lower() for msg in logs), (
        f"missing-brew reason must be logged; got {logs!r}"
    )


# ---------------------------------------------------------------------------
# Linux: numpy ImportError during libiomp5 probe
# ---------------------------------------------------------------------------


def test_libiomp5_candidate_prefers_runtime_over_debug_helper(tmp_path):
    """The Linux image's debugger helper must never win directory ordering."""
    runtime = tmp_path / "libiomp5.so"
    runtime.write_bytes(b"runtime")
    (tmp_path / "libiomp5_db.so").write_bytes(b"debug helper")

    assert _libiomp5_candidate(tmp_path) == runtime


def test_libiomp5_candidate_accepts_deterministic_versioned_and_hashed_names(
    tmp_path,
):
    """Bundled wheel names remain supported with stable preference ordering."""
    hashed = tmp_path / "libiomp5-a1b2c3.so"
    versioned = tmp_path / "libiomp5.so.5"
    hashed.write_bytes(b"hashed runtime")
    versioned.write_bytes(b"versioned runtime")

    assert _libiomp5_candidate(tmp_path) == versioned


@pytest.mark.parametrize(
    "name",
    [
        "libiomp5_db.so",
        "libiomp5.a",
        "libiomp5.so.debug",
        "libiomp5-debug.so",
        "not-libiomp5.so",
        "libiomp5.so.backup",
    ],
)
def test_libiomp5_candidate_rejects_non_runtime_names(tmp_path, name):
    (tmp_path / name).write_bytes(b"not a runtime")

    assert _libiomp5_candidate(tmp_path) is None


def test_find_libiomp5_logs_numpy_import_error(monkeypatch):
    """If numpy cannot be imported during the probe (the exact ILP64 ABI
    mismatch runtime recompile exists to fix), log it and fall through to
    system paths instead of failing silently."""
    logs: list[str] = []

    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("simulated ABI mismatch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    # System paths will also be absent in most CI containers; the return
    # value is either a Path or None, both valid — we only care that the
    # numpy-failure log was emitted and the call didn't raise.
    result = _find_libiomp5(BuildReport(detail=logs.append, warn=logs.append))

    # May be None (no system libiomp5) or a Path (system install present).
    # Both outcomes are valid; the contract is "don't crash".
    assert result is None or hasattr(result, "name")
    assert any("numpy import failed" in msg for msg in logs), (
        f"numpy ImportError must be logged; got {logs!r}"
    )


# ---------------------------------------------------------------------------
# Linux: clang probe error paths
# ---------------------------------------------------------------------------


def test_openmp_flags_for_libiomp5_clang_timeout_falls_back_to_gcc(
    monkeypatch, tmp_path
):
    """A wedged clang probe must not crash detection — fall back to the
    GCC+libiomp5 path with an always-visible warning."""
    logs: list[str] = []
    warns: list[str] = []

    import jamma._build_support.openmp_detect as mod

    # Pretend clang exists on PATH.
    monkeypatch.setattr(mod.shutil, "which", lambda _name: "/usr/bin/clang")

    def _fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="clang", timeout=10)

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    libiomp5 = tmp_path / "libiomp5.so"
    libiomp5.write_bytes(b"")

    cflags, lflags, cc_override = _openmp_flags_for_libiomp5(
        cc_cmd="gcc",
        libiomp5_path=libiomp5,
        report=BuildReport(detail=logs.append, warn=warns.append),
    )

    assert cc_override == "gcc", "clang probe timeout must fall back to original cc_cmd"
    assert cflags == ["-fopenmp"]
    assert any(str(libiomp5) in flag for flag in lflags), (
        f"link flags must still include libiomp5 path; got {lflags!r}"
    )
    assert any("probe failed" in msg for msg in logs), (
        f"clang probe failure must be logged; got {logs!r}"
    )
    # GCC+libiomp5 is a known-crashy config; the warning must surface.
    assert any("GOMP compatibility" in msg for msg in warns), (
        f"GCC+libiomp5 fallback warning missing; got {warns!r}"
    )


def test_openmp_flags_for_libiomp5_clang_oserror_falls_back_to_gcc(
    monkeypatch, tmp_path
):
    """A broken clang symlink (OSError/FileNotFoundError during exec) must
    not crash detection — same fallback path as the timeout case."""
    logs: list[str] = []
    warns: list[str] = []

    import jamma._build_support.openmp_detect as mod

    monkeypatch.setattr(mod.shutil, "which", lambda _name: "/usr/bin/clang")

    def _fake_run(*_args, **_kwargs):
        raise OSError("broken clang symlink")

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)

    libiomp5 = tmp_path / "libiomp5.so"
    libiomp5.write_bytes(b"")

    cflags, lflags, cc_override = _openmp_flags_for_libiomp5(
        cc_cmd="gcc",
        libiomp5_path=libiomp5,
        report=BuildReport(detail=logs.append, warn=warns.append),
    )

    assert cc_override == "gcc"
    assert cflags == ["-fopenmp"]
    assert any(str(libiomp5) in flag for flag in lflags)
    assert any("probe failed" in msg for msg in logs)
    assert any("GOMP compatibility" in msg for msg in warns)


def _intel_openmp_prefix(tmp_path):
    """Lay out the intel-openmp wheel's lib/ and opt/compiler/include/."""
    prefix = tmp_path / "venv"
    (prefix / "lib").mkdir(parents=True)
    libiomp5 = prefix / "lib" / "libiomp5.so"
    libiomp5.write_bytes(b"")
    include_dir = prefix / "opt" / "compiler" / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "omp.h").write_text("")
    return libiomp5, include_dir


def test_clang_probe_link_command_omits_fopenmp(tmp_path):
    """clang's -fopenmp at link time adds -lomp, which fails without libomp-dev.

    The probe links libiomp5 by path, as the real build does.
    """
    link_flags = ["/opt/venv/lib/libiomp5.so", "-Wl,-rpath,/opt/venv/lib"]

    compile_cmd, link_cmd = _clang_probe_commands(
        "/usr/bin/clang", ["-fopenmp", "-I/inc"], link_flags, tmp_path / "probe.c"
    )

    assert compile_cmd == [
        "/usr/bin/clang",
        "-fopenmp",
        "-I/inc",
        "-c",
        str(tmp_path / "probe.c"),
        "-o",
        str(tmp_path / "probe.o"),
    ]
    assert link_cmd == [
        "/usr/bin/clang",
        str(tmp_path / "probe.o"),
        "-o",
        str(tmp_path / "probe"),
        *link_flags,
    ]


def test_intel_openmp_include_dir_follows_symlinked_runtime(tmp_path):
    """A /usr/local/lib symlink to the wheel's libiomp5 finds the wheel's omp.h."""
    libiomp5, include_dir = _intel_openmp_prefix(tmp_path)
    link = tmp_path / "usr-local-lib" / "libiomp5.so"
    link.parent.mkdir()
    link.symlink_to(libiomp5)

    assert _intel_openmp_include_dir(link) == include_dir.resolve()
    assert _intel_openmp_include_dir(tmp_path / "libiomp5.so") is None


def _fake_clang_without_libomp(commands):
    """Emulate clang on a host with no libomp-dev and no clang omp.h."""

    def _run(command, **_kwargs):
        commands.append(command)
        if "-c" not in command:
            if "-fopenmp" in command:
                return subprocess.CompletedProcess(
                    command, 1, "", "ld: cannot find -lomp"
                )
            return subprocess.CompletedProcess(command, 0, "", "")
        if not any(flag.startswith("-I") for flag in command):
            return subprocess.CompletedProcess(
                command, 1, "", "fatal error: 'omp.h' file not found"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    return _run


def test_openmp_flags_for_libiomp5_uses_clang_without_libomp(monkeypatch, tmp_path):
    """With intel-openmp's libiomp5 and omp.h, clang is chosen, no libomp needed."""
    import jamma._build_support.openmp_detect as mod

    libiomp5, include_dir = _intel_openmp_prefix(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(mod.shutil, "which", lambda _name: "/usr/bin/clang")
    monkeypatch.setattr(mod.subprocess, "run", _fake_clang_without_libomp(commands))
    warns: list[str] = []

    cflags, lflags, cc_override = _openmp_flags_for_libiomp5(
        cc_cmd="gcc",
        libiomp5_path=libiomp5,
        report=BuildReport(detail=lambda _msg: None, warn=warns.append),
    )

    assert cc_override == "/usr/bin/clang"
    assert cflags == ["-fopenmp", f"-I{include_dir.resolve()}"]
    assert lflags == [str(libiomp5), f"-Wl,-rpath,{libiomp5.parent}"]
    assert len(commands) == 2
    assert warns == []


def test_openmp_flags_for_libiomp5_without_omp_h_falls_back_to_gcc(
    monkeypatch, tmp_path
):
    """No omp.h for clang means the real compile would fail, so use GCC."""
    import jamma._build_support.openmp_detect as mod

    libiomp5 = tmp_path / "libiomp5.so"
    libiomp5.write_bytes(b"")
    commands: list[list[str]] = []
    monkeypatch.setattr(mod.shutil, "which", lambda _name: "/usr/bin/clang")
    monkeypatch.setattr(mod.subprocess, "run", _fake_clang_without_libomp(commands))
    logs: list[str] = []
    warns: list[str] = []

    cflags, _lflags, cc_override = _openmp_flags_for_libiomp5(
        cc_cmd="gcc",
        libiomp5_path=libiomp5,
        report=BuildReport(detail=logs.append, warn=warns.append),
    )

    assert cc_override == "gcc"
    assert cflags == ["-fopenmp"]
    assert len(commands) == 1, "a failed compile must skip the link step"
    assert any("'omp.h' file not found" in msg for msg in logs)
    assert any("GOMP compatibility" in msg for msg in warns)
