"""Tests for jamma._build_support.compile_and_link.

Covers the constants, resolve_cflags_for dispatch, and a smoke test for
execute_build (subprocess monkeypatched so the test doesn't shell out).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma._build_support.build_execution import (
    CompileResult,
    Toolchain,
    execute_build,
)
from jamma._build_support.build_models import (
    BASE_CFLAGS,
    BASELINE_SOURCES,
    LAPACK_CFLAGS,
    LAPACK_SOURCES,
    LINK_FLAGS_BY_PLATFORM,
    LMM_ACCEL_SOURCES,
    BuildSpec,
    resolve_cflags_for,
)
from jamma._build_support.compile_and_link import run_build

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
    )


def test_link_flags_linux():
    assert LINK_FLAGS_BY_PLATFORM["Linux"] == ("-ldl", "-lpthread")


def test_link_flags_darwin():
    assert LINK_FLAGS_BY_PLATFORM["Darwin"] == ("-undefined", "dynamic_lookup")


# ---------------------------------------------------------------------------
# resolve_cflags_for — dispatch behavior
# ---------------------------------------------------------------------------


def test_resolve_cflags_lapack_source_gets_strict_ieee_flags():
    flags = resolve_cflags_for(
        Path("eigh.c"),
        lapack_source_set={"eigh.c"},
        include_dirs=["/foo", "/bar"],
        extra_cflags=[],
    )
    assert "-O2" in flags
    assert "-fno-fast-math" in flags
    assert "-I/foo" in flags
    assert "-I/bar" in flags
    # LAPACK sources must NOT receive baseline optimizations.
    assert "-O3" not in flags
    assert "-funroll-loops" not in flags


def test_resolve_cflags_baseline_source_gets_fast_flags():
    flags = resolve_cflags_for(
        Path("platform.c"),
        lapack_source_set={"eigh.c"},
        include_dirs=["/foo"],
        extra_cflags=[],
    )
    assert "-O3" in flags
    assert "-funroll-loops" in flags
    assert "-fno-finite-math-only" in flags
    assert "-I/foo" in flags
    # Baseline sources must NOT get the LAPACK -O2 regime.
    assert "-O2" not in flags


def test_extra_cflags_precede_fno_finite_math_only():
    """Load-bearing ordering: user CFLAGS (-Ofast) must come BEFORE
    -fno-finite-math-only so the trailing explicit flag overrides -Ofast's
    implicit -ffinite-math-only.
    """
    flags = resolve_cflags_for(
        Path("platform.c"),
        lapack_source_set={"eigh.c"},
        include_dirs=[],
        extra_cflags=["-Ofast"],
    )
    idx_ofast = flags.index("-Ofast")
    idx_finite = flags.index("-fno-finite-math-only")
    assert idx_ofast < idx_finite, (
        "extra_cflags (-Ofast) must precede -fno-finite-math-only so the "
        "trailing explicit flag overrides user -Ofast. Ordering is load-bearing."
    )


def test_extra_source_includes_appended():
    flags = resolve_cflags_for(
        Path("platform.c"),
        lapack_source_set={"eigh.c"},
        include_dirs=[],
        extra_cflags=[],
        extra_source_includes=["/tmp/test"],
    )
    assert "-I/tmp/test" in flags


def test_lapack_path_ignores_extra_cflags_for_ieee_safety():
    """LAPACK split is strict IEEE 754. Caller-supplied -Ofast would defeat
    the split, so extra_cflags is deliberately NOT merged into the LAPACK path.
    """
    flags = resolve_cflags_for(
        Path("eigh.c"),
        lapack_source_set={"eigh.c"},
        include_dirs=[],
        extra_cflags=["-Ofast"],
    )
    assert "-Ofast" not in flags
    assert "-O2" in flags


def test_resolve_cflags_none_args_accepted():
    """None inputs for extra_cflags / extra_source_includes should be OK."""
    flags = resolve_cflags_for(
        Path("platform.c"),
        lapack_source_set={"eigh.c"},
        include_dirs=[],
        extra_cflags=None,
        extra_source_includes=None,
    )
    # No crash; flags are just the baseline set.
    assert "-O3" in flags


# ---------------------------------------------------------------------------
# CompileResult dataclass
# ---------------------------------------------------------------------------


def test_compile_result_fields():
    r = CompileResult(success=True, used_openmp=True, used_openmp_link=True)
    assert r.success is True
    assert r.used_openmp is True
    assert r.used_openmp_link is True
    assert r.output_path is None
    assert r.error is None


# ---------------------------------------------------------------------------
# execute_build smoke test — subprocess.run monkeypatched to return success.
# No real compilation runs; Wave 4 validates end-to-end.
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


def test_execute_build_smoke_success(monkeypatch, tmp_path):
    """Construct a execute_build call with stub paths; assert CompileResult
    fields after a monkeypatched subprocess.run that always returns success.
    """
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        # Mirror cc behavior: -o <path> writes a file at <path>. The link
        # step now writes to a sibling .tmp.<pid> path and the helper
        # os.replace()s it onto the real output (added in jamma-oy1c so
        # concurrent recompilers can't observe a half-written .so).
        # Without creating the file the atomic-replace step would fail.
        if "-o" in cmd:
            out_idx = cmd.index("-o") + 1
            if out_idx < len(cmd):
                from pathlib import Path

                Path(cmd[out_idx]).write_bytes(b"")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(
        "jamma._build_support.build_execution.subprocess.run",
        _fake_run,
    )

    # Create empty source files so execute_build's file existence checks
    # (if any) don't fail; the real cc_cmd is never invoked.
    src_a = tmp_path / "platform.c"
    src_a.write_text("// stub\n")
    src_b = tmp_path / "eigh.c"
    src_b.write_text("// stub\n")
    out = tmp_path / "out.so"

    result = execute_build(
        sources=[src_a, src_b],
        lapack_sources=[src_b],
        include_dirs=["/usr/include"],
        cc_cmd="cc",
        cc_extra=[],
        omp_compile=["-fopenmp"],
        omp_link=["-liomp5"],
        ldflags=["-lm"],
        output=out,
        tmp_dir=tmp_path / "objs",
    )

    assert isinstance(result, CompileResult)
    assert result.success is True
    assert result.output_path == out
    # One subprocess call per source + one link call = exactly 3.
    assert len(calls) == 3, f"expected 3 subprocess calls, got {len(calls)}: {calls}"

    # Happy path: OMP-enabled compile must use -fopenmp on each source and
    # the link step must pull in the Intel runtime (-liomp5) and -lm.
    assert result.used_openmp is True
    assert result.used_openmp_link is True
    assert "-fopenmp" in calls[0], f"platform.c compile missing -fopenmp: {calls[0]}"
    assert "-fopenmp" in calls[1], f"eigh.c compile missing -fopenmp: {calls[1]}"
    # eigh.c is in lapack_sources — strict IEEE 754 path must NOT get -O3.
    assert "-O2" in calls[1], f"eigh.c missing -O2 (LAPACK path): {calls[1]}"
    assert "-O3" not in calls[1], f"eigh.c must not get -O3: {calls[1]}"
    assert "-O3" in calls[0], f"platform.c must compile with -O3, got: {calls[0]}"
    # Link command is the last call; Intel runtime must be present.
    link_cmd = calls[-1]
    assert "-liomp5" in link_cmd, f"link missing -liomp5 (Intel runtime): {link_cmd}"
    assert "-lm" in link_cmd, f"link missing -lm: {link_cmd}"


def test_execute_build_compile_failure_triggers_omp_retry(monkeypatch, tmp_path):
    """When the first compile fails WITH omp_compile active, execute_build
    should retry once WITHOUT OpenMP and invoke on_retry().

    Also asserts the retry subprocess command actually has ``-fopenmp``
    stripped and the subsequent link command has ``-liomp5`` stripped —
    without that, a refactor could fire on_retry() while still passing OMP
    flags, defeating the whole retry path.
    """
    calls: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        # First compile attempt fails (has -fopenmp).
        if len(calls) == 1 and "-fopenmp" in cmd:
            return _FakeCompleted(returncode=1, stderr="omp compile failed")
        # Materialize the -o target so the link step's atomic os.replace
        # (jamma-oy1c) can find a real file at the .tmp.<pid> path.
        if "-o" in cmd:
            out_idx = cmd.index("-o") + 1
            if out_idx < len(cmd):
                from pathlib import Path

                Path(cmd[out_idx]).write_bytes(b"")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(
        "jamma._build_support.build_execution.subprocess.run",
        _fake_run,
    )

    retry_reasons: list[str] = []

    src = tmp_path / "platform.c"
    src.write_text("// stub\n")
    out = tmp_path / "out.so"

    result = execute_build(
        sources=[src],
        lapack_sources=[],
        include_dirs=[],
        cc_cmd="cc",
        cc_extra=[],
        omp_compile=["-fopenmp"],
        omp_link=["-liomp5"],
        ldflags=[],
        output=out,
        tmp_dir=tmp_path / "objs",
        on_retry=lambda reason: retry_reasons.append(reason),
    )

    assert result.success is True
    assert len(retry_reasons) >= 1
    assert result.used_openmp_link is False

    # Call pattern: [compile-with-omp (fail), compile-retry-no-omp (ok), link].
    assert len(calls) >= 3, f"expected >=3 calls, got {len(calls)}: {calls}"
    assert "-fopenmp" in calls[0], "first compile should have had -fopenmp"
    assert "-fopenmp" not in calls[1], (
        f"retry compile must NOT include -fopenmp; got {calls[1]}"
    )
    # Link (the last call) must not include the OMP runtime either.
    assert "-liomp5" not in calls[-1], (
        f"retry link must NOT include -liomp5; got {calls[-1]}"
    )


def test_atomic_replace_failure_preserves_used_openmp_link(monkeypatch, tmp_path):
    """When link succeeded but atomic os.replace fails, the returned
    ``used_openmp_link`` must reflect the REAL link-time state (True here),
    not be zeroed out. Without this, telemetry misreports the build as
    "no OMP runtime linked" whenever the final rename races.
    """

    def _fake_run(cmd, **_kwargs):
        # Materialize -o targets so the link step reaches the os.replace path.
        if "-o" in cmd:
            out_idx = cmd.index("-o") + 1
            if out_idx < len(cmd):
                Path(cmd[out_idx]).write_bytes(b"")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(
        "jamma._build_support.build_execution.subprocess.run",
        _fake_run,
    )

    # Patch Path.replace on the link_tmp path to raise OSError, exercising
    # the except branch at compile_and_link.py:434-445.
    real_replace = Path.replace

    def _raise_replace(self, target):
        # Only fail for the atomic publish step (target is the .so output).
        if str(self).endswith(".tmp") or ".tmp." in self.name:
            raise OSError("simulated atomic replace failure")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _raise_replace)

    src = tmp_path / "platform.c"
    src.write_text("// stub\n")
    out = tmp_path / "out.so"

    result = execute_build(
        sources=[src],
        lapack_sources=[],
        include_dirs=[],
        cc_cmd="cc",
        cc_extra=[],
        omp_compile=["-fopenmp"],
        omp_link=["-liomp5"],
        ldflags=[],
        output=out,
        tmp_dir=tmp_path / "objs",
    )

    assert result.success is False, (
        "atomic replace failure must surface as a failed build"
    )
    assert "atomic replace" in (result.error or ""), (
        f"error must mention atomic replace; got {result.error!r}"
    )
    # The telemetry regression: used_openmp_link must NOT be False.
    # The link call succeeded with -liomp5 in the flags — the rename is
    # what failed.
    assert result.used_openmp_link is True, (
        "used_openmp_link must reflect the successful link, not be zeroed "
        "by the os.replace failure"
    )
    assert result.used_openmp is True


# ---------------------------------------------------------------------------
# Toolchain detected once, reused across a run_build of both specs (F2).
# ---------------------------------------------------------------------------


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

    def _fake_detect_openmp_flags(cc_cmd, system, _print, _warn=None):
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

    toolchain = detect_toolchain()
    assert isinstance(toolchain, Toolchain)
    assert len(find_calls) == 1
    assert len(omp_calls) == 1

    def _fake_run(cmd, **kwargs):
        if "-o" in cmd:
            out_idx = cmd.index("-o") + 1
            if out_idx < len(cmd):
                Path(cmd[out_idx]).write_bytes(b"")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(
        "jamma._build_support.build_execution.subprocess.run", _fake_run
    )

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
        )

    package_dir = tmp_path
    result_a = run_build(_make_spec("target_a"), package_dir, toolchain, dev_mode=True)
    result_b = run_build(_make_spec("target_b"), package_dir, toolchain, dev_mode=True)

    assert result_a.ok, result_a.error
    assert result_b.ok, result_b.error
    # Both builds used the one Toolchain detected above — no second probe.
    assert len(find_calls) == 1
    assert len(omp_calls) == 1
