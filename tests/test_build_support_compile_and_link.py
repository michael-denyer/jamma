"""Tests for jamma._build_support.compile_and_link.

Covers the constants, resolve_cflags_for dispatch, and execute_build and
run_build with subprocess monkeypatched so no test shells out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from jamma._build_support.build_execution import Toolchain, execute_build
from jamma._build_support.build_models import (
    BASE_CFLAGS,
    BASELINE_SOURCES,
    JLINALG_SPEC,
    LAPACK_CFLAGS,
    LAPACK_SOURCES,
    LINK_FLAGS_BY_PLATFORM,
    LINK_LIBS,
    LMM_ACCEL_SOURCES,
    LMM_ACCEL_SPEC,
    BuildReport,
    BuildResult,
    BuildSpec,
    ResolvedFlags,
    resolve_cflags_for,
    resolve_flags,
)
from jamma._build_support.compile_and_link import compile_extension, run_build

pytestmark = pytest.mark.tier0

# ---------------------------------------------------------------------------
# Constants: exhaustive value checks — any drift breaks the three entry
# points (hatch_build.py, _execute_build.py, _compile_accel.py) that
# import these constants.
# ---------------------------------------------------------------------------


def test_base_cflags_exact_sequence():
    assert BASE_CFLAGS == (
        "-O3",
        "-ftree-vectorize",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-funroll-loops",
        "-fno-finite-math-only",
        "-Wframe-larger-than=131072",
        "-fPIC",
        "-std=c11",
    )


def test_base_cflags_is_tuple_immutable():
    assert isinstance(BASE_CFLAGS, tuple)


def test_lapack_cflags_exact_sequence():
    assert LAPACK_CFLAGS == (
        "-O2",
        "-fno-fast-math",
        "-fno-finite-math-only",
        "-Wframe-larger-than=131072",
        "-fPIC",
        "-std=c11",
    )


def test_lapack_cflags_is_tuple_immutable():
    assert isinstance(LAPACK_CFLAGS, tuple)


def test_baseline_sources_exact():
    assert BASELINE_SOURCES == (
        "platform.c",
        "pymodule.c",
        "blas_dispatch.c",
        "blas_operations.c",
        "snp_stats.c",
    )


def test_lapack_sources_exact():
    assert LAPACK_SOURCES == ("eigh.c",)


def test_lmm_accel_sources_exact():
    """Every accelerator family must reach wheel and dev-mode builds."""
    assert LMM_ACCEL_SOURCES == (
        "_lmm_accel.c",
        "_lmm_accel_ncvt1.c",
        "_lmm_accel_general.c",
        "_lmm_support.c",
        "_lmm_stats.c",
        "_lmm_kernels_general.c",
        "_lmm_kernels_ncvt1.c",
        "_lmm_accel_bgen.c",
    )


def test_only_the_accelerator_links_zlib():
    """The BGEN decoder's zlib is linked into _lmm_accel and nothing else."""
    accel = resolve_flags(LMM_ACCEL_SPEC, dev_mode=False, system="Linux", env={})
    jlinalg = resolve_flags(JLINALG_SPEC, dev_mode=False, system="Linux", env={})
    assert accel.link_libs == (*LINK_LIBS, "-lz")
    assert jlinalg.link_libs == LINK_LIBS


def test_link_flags_linux():
    assert LINK_FLAGS_BY_PLATFORM["Linux"] == ("-ldl", "-lpthread")


def test_link_flags_darwin():
    assert LINK_FLAGS_BY_PLATFORM["Darwin"] == ("-undefined", "dynamic_lookup")


# ---------------------------------------------------------------------------
# resolve_cflags_for — dispatch behavior
# ---------------------------------------------------------------------------


def _flags(*, base_extra: tuple[str, ...] = ()) -> ResolvedFlags:
    return ResolvedFlags(
        base_extra=base_extra, lapack_extra=(), platform_link=(), link_libs=("-lm",)
    )


def test_resolve_cflags_lapack_source_gets_strict_ieee_flags():
    flags = resolve_cflags_for(_flags(), ["/foo", "/bar"], lapack=True)
    assert flags == [*LAPACK_CFLAGS, "-I/foo", "-I/bar"]
    # LAPACK sources must NOT receive baseline optimizations.
    assert "-O3" not in flags
    assert "-funroll-loops" not in flags


def test_resolve_cflags_baseline_source_gets_fast_flags():
    flags = resolve_cflags_for(_flags(), ["/foo"], lapack=False)
    assert flags == [*BASE_CFLAGS, "-I/foo"]
    # Baseline sources must NOT get the LAPACK -O2 regime.
    assert "-O2" not in flags


def test_extra_cflags_precede_fno_finite_math_only():
    """Load-bearing ordering: user CFLAGS (-Ofast) must come BEFORE
    -fno-finite-math-only so the trailing explicit flag overrides -Ofast's
    implicit -ffinite-math-only.
    """
    flags = resolve_cflags_for(_flags(base_extra=("-Ofast",)), [], lapack=False)
    assert flags.index("-Ofast") < flags.index("-fno-finite-math-only"), (
        "base_extra (-Ofast) must precede -fno-finite-math-only so the "
        "trailing explicit flag overrides user -Ofast. Ordering is load-bearing."
    )


def test_lapack_path_ignores_extra_cflags_for_ieee_safety():
    """LAPACK split is strict IEEE 754. Caller-supplied -Ofast would defeat
    the split, so base_extra is deliberately NOT merged into the LAPACK path.
    """
    flags = resolve_cflags_for(_flags(base_extra=("-Ofast",)), [], lapack=True)
    assert "-Ofast" not in flags
    assert "-O2" in flags


# ---------------------------------------------------------------------------
# execute_build — subprocess.run monkeypatched; no real compilation runs.
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def _write_output(cmd: list[str]) -> _FakeCompleted:
    """Mirror cc: ``-o <path>`` writes a file, which the atomic publish needs."""
    Path(cmd[cmd.index("-o") + 1]).write_bytes(b"")
    return _FakeCompleted(returncode=0)


def _toolchain(*, openmp: bool = True) -> Toolchain:
    return Toolchain(
        cc_cmd="cc",
        cc_extra=(),
        python_inc="",
        numpy_inc="",
        system="Linux",
        omp_compile=("-fopenmp",) if openmp else (),
        omp_link=("-liomp5",) if openmp else (),
    )


def _spec(*sources: str, lapack: tuple[str, ...] = ()) -> BuildSpec:
    return BuildSpec(
        package_parts=("pkg",),
        source_parts=(),
        include_parts=(),
        sources=sources,
        lapack_sources=lapack,
        output_stem="_pkg",
        sys_module_key="jamma.pkg._pkg",
        fallback_label="pkg",
    )


def _report(warnings: list[str] | None = None) -> BuildReport:
    sink = [] if warnings is None else warnings
    return BuildReport(detail=lambda _m: None, warn=sink.append)


def _execute(
    tmp_path: Path,
    spec: BuildSpec,
    *,
    toolchain: Toolchain | None = None,
    report: BuildReport | None = None,
) -> BuildResult:
    return execute_build(
        spec,
        tmp_path,
        ["/usr/include"],
        _toolchain() if toolchain is None else toolchain,
        _flags(),
        tmp_path / "out.so",
        tmp_path / "objs",
        _report() if report is None else report,
    )


def _patch_run(monkeypatch, fake) -> None:
    monkeypatch.setattr("jamma._build_support.build_execution.subprocess.run", fake)


def test_execute_build_smoke_success(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        return _write_output(cmd)

    _patch_run(monkeypatch, _fake_run)

    result = _execute(tmp_path, _spec("platform.c", "eigh.c", lapack=("eigh.c",)))

    assert result == BuildResult(
        phase="ok",
        output_path=tmp_path / "out.so",
        used_openmp=True,
        used_openmp_link=True,
    )
    # One subprocess call per source + one link call = exactly 3.
    assert len(calls) == 3, f"expected 3 subprocess calls, got {len(calls)}: {calls}"
    assert "-fopenmp" in calls[0]
    assert "-fopenmp" in calls[1]
    # eigh.c is a LAPACK source — strict IEEE 754 path must NOT get -O3.
    assert "-O2" in calls[1]
    assert "-O3" not in calls[1]
    assert "-O3" in calls[0]
    assert calls[-1][-2:] == ["-liomp5", "-lm"], calls[-1]


def test_execute_build_compile_failure_triggers_omp_retry(monkeypatch, tmp_path):
    """A failed OpenMP compile retries once without OpenMP, and the retried
    link drops the OpenMP runtime too.
    """
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if len(calls) == 1:
            return _FakeCompleted(returncode=1, stderr="omp compile failed")
        return _write_output(cmd)

    _patch_run(monkeypatch, _fake_run)
    warnings: list[str] = []

    result = _execute(tmp_path, _spec("platform.c"), report=_report(warnings))

    assert result.ok
    assert (result.used_openmp, result.used_openmp_link) == (False, False)
    assert len(calls) == 3, calls
    assert "-fopenmp" in calls[0]
    assert "-fopenmp" not in calls[1]
    assert "-liomp5" not in calls[2]
    assert sum("retrying without OpenMP" in w for w in warnings) == 1


def test_execute_build_compile_failure_reports_compiler_stderr(monkeypatch, tmp_path):
    """A terminal compile failure carries the compiler's stderr in ``error``
    and prints it through ``warn``.
    """
    _patch_run(
        monkeypatch,
        lambda cmd, **_kw: _FakeCompleted(1, stderr="platform.c:3: undeclared foo"),
    )
    warnings: list[str] = []

    result = _execute(
        tmp_path,
        _spec("platform.c"),
        toolchain=_toolchain(openmp=False),
        report=_report(warnings),
    )

    assert result.phase == "build"
    assert "platform.c:3: undeclared foo" in result.error
    assert "platform.c:3: undeclared foo" in warnings


def test_execute_build_link_failure_retries_without_omp_runtime(monkeypatch, tmp_path):
    """A failed OpenMP link retries the same objects without its runtime."""
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if "-c" not in cmd and "-liomp5" in cmd:
            return _FakeCompleted(returncode=1, stderr="omp link failed")
        return _write_output(cmd)

    _patch_run(monkeypatch, _fake_run)
    warnings: list[str] = []

    result = _execute(tmp_path, _spec("platform.c"), report=_report(warnings))

    assert result.ok
    assert result.used_openmp
    assert not result.used_openmp_link
    assert len(warnings) == 1
    assert "omp link failed" in warnings[0]
    assert "-liomp5" in calls[-2]
    assert "-liomp5" not in calls[-1]


@pytest.mark.parametrize("failing_step", ["compile", "link"])
def test_openmp_retry_notice_prints_once_on_a_shared_stream(
    monkeypatch, tmp_path, capsys, failing_step
):
    """With detail and warn on one stream (the wheel build), a retry prints once."""

    def _fake_run(cmd, **_kwargs):
        compiling = "-c" in cmd
        if failing_step == "compile" and compiling and "-fopenmp" in cmd:
            return _FakeCompleted(returncode=1, stderr="omp compile failed")
        if failing_step == "link" and not compiling and "-liomp5" in cmd:
            return _FakeCompleted(returncode=1, stderr="omp link failed")
        return _write_output(cmd)

    _patch_run(monkeypatch, _fake_run)

    result = _execute(
        tmp_path,
        _spec("platform.c"),
        report=BuildReport.to_stream(sys.stderr, verbose=True),
    )

    assert result.ok
    assert capsys.readouterr().err.count("retrying without OpenMP") == 1


def test_atomic_replace_failure_preserves_used_openmp_link(monkeypatch, tmp_path):
    """When link succeeded but atomic os.replace fails, the returned
    ``used_openmp_link`` must reflect the REAL link-time state (True here),
    not be zeroed out. Without this, telemetry misreports the build as
    "no OMP runtime linked" whenever the final rename races.
    """
    _patch_run(monkeypatch, lambda cmd, **_kw: _write_output(cmd))
    real_replace = Path.replace

    def _raise_replace(self, target):
        if ".tmp." in self.name:
            raise OSError("simulated atomic replace failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _raise_replace)

    result = _execute(tmp_path, _spec("platform.c"))

    assert result.phase == "build"
    assert "atomic replace" in result.error
    assert result.used_openmp is True
    assert result.used_openmp_link is True, (
        "used_openmp_link must reflect the successful link, not be zeroed "
        "by the os.replace failure"
    )


# ---------------------------------------------------------------------------
# run_build and compile_extension
# ---------------------------------------------------------------------------


def test_run_build_preflight_returns_the_reason_without_printing(tmp_path):
    """Missing sources stop the build before the compiler runs; run_build
    returns the reason and leaves the wording to its caller.
    """
    printed: list[str] = []

    result = run_build(
        _spec("absent.c"),
        tmp_path,
        _toolchain(),
        dev_mode=True,
        report=BuildReport(detail=printed.append, warn=printed.append),
    )

    assert result.phase == "preflight"
    assert "C source files missing" in result.error
    assert printed == []


def test_compile_extension_reports_a_preflight_failure_once(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "jamma._build_support.compile_and_link.detect_toolchain",
        lambda report, **_kwargs: _toolchain(),
    )
    warnings: list[str] = []

    ok = compile_extension(_spec("absent.c"), tmp_path, _report(warnings))

    assert ok is False
    assert len(warnings) == 1
    assert warnings[0].startswith("ERROR: _pkg compilation failed: C source files")


def test_toolchain_detected_once_across_run_build_of_both_specs(monkeypatch, tmp_path):
    """detect_toolchain() must run exactly once per process; run_build takes
    the resulting Toolchain as a plain parameter rather than re-detecting
    the compiler and OpenMP flags for every BuildSpec it builds.
    """
    find_calls = []
    omp_calls = []

    def _fake_find_c_compiler():
        find_calls.append(1)
        return ("cc", [])

    def _fake_detect_openmp_flags(cc_cmd, system, report):
        omp_calls.append(1)
        return ([], [], cc_cmd)

    monkeypatch.setattr(
        "jamma._build_support.find_compiler.find_c_compiler",
        _fake_find_c_compiler,
    )
    monkeypatch.setattr(
        "jamma._build_support.openmp_detect.detect_openmp_flags",
        _fake_detect_openmp_flags,
    )

    from jamma._build_support.build_execution import detect_toolchain

    toolchain = detect_toolchain(_report())
    assert isinstance(toolchain, Toolchain)
    assert len(find_calls) == 1
    assert len(omp_calls) == 1

    _patch_run(monkeypatch, lambda cmd, **_kw: _write_output(cmd))

    def _make_spec(name: str) -> BuildSpec:
        src_dir = tmp_path / name
        src_dir.mkdir()
        (src_dir / "one.c").write_text("// stub\n")
        return BuildSpec(
            package_parts=(name,),
            source_parts=(),
            include_parts=(),
            sources=("one.c",),
            lapack_sources=(),
            output_stem=f"_{name}",
            sys_module_key=f"jamma.{name}._{name}",
            fallback_label=name,
        )

    results = [
        run_build(_make_spec(n), tmp_path, toolchain, dev_mode=True, report=_report())
        for n in ("target_a", "target_b")
    ]

    assert all(r.ok for r in results), [r.error for r in results]
    # Both builds used the one Toolchain detected above — no second probe.
    assert len(find_calls) == 1
    assert len(omp_calls) == 1
