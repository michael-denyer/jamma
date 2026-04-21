"""Tests for jamma._build_support.compile_and_link.

Covers the constants, resolve_cflags_for dispatch, and a smoke test for
compile_jlinalg (subprocess monkeypatched so the test doesn't shell out).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma._build_support.compile_and_link import (
    BASE_CFLAGS,
    BASELINE_SOURCES,
    LAPACK_CFLAGS,
    LAPACK_SOURCES,
    LINK_FLAGS_BY_PLATFORM,
    CompileResult,
    compile_jlinalg,
    resolve_cflags_for,
)

# ---------------------------------------------------------------------------
# Constants: exhaustive value checks — any drift breaks the three entry
# points (hatch_build.py, _compile_jlinalg.py, _compile_accel.py) that
# import these constants.
# ---------------------------------------------------------------------------


@pytest.mark.tier0
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


@pytest.mark.tier0
def test_base_cflags_is_tuple_immutable():
    assert isinstance(BASE_CFLAGS, tuple)


@pytest.mark.tier0
def test_lapack_cflags_exact_sequence():
    assert LAPACK_CFLAGS == (
        "-O2",
        "-fno-fast-math",
        "-fno-finite-math-only",
        "-Wframe-larger-than=131072",
        "-fPIC",
        "-std=c11",
    )


@pytest.mark.tier0
def test_lapack_cflags_is_tuple_immutable():
    assert isinstance(LAPACK_CFLAGS, tuple)


@pytest.mark.tier0
def test_baseline_sources_exact():
    assert BASELINE_SOURCES == (
        "platform.c",
        "pymodule.c",
        "blas_dispatch.c",
        "snp_stats.c",
    )


@pytest.mark.tier0
def test_lapack_sources_exact():
    assert LAPACK_SOURCES == ("eigh.c",)


@pytest.mark.tier0
def test_link_flags_linux():
    assert LINK_FLAGS_BY_PLATFORM["Linux"] == ("-ldl", "-lpthread")


@pytest.mark.tier0
def test_link_flags_darwin():
    assert LINK_FLAGS_BY_PLATFORM["Darwin"] == ("-undefined", "dynamic_lookup")


# ---------------------------------------------------------------------------
# resolve_cflags_for — dispatch behavior
# ---------------------------------------------------------------------------


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
def test_extra_source_includes_appended():
    flags = resolve_cflags_for(
        Path("platform.c"),
        lapack_source_set={"eigh.c"},
        include_dirs=[],
        extra_cflags=[],
        extra_source_includes=["/tmp/test"],
    )
    assert "-I/tmp/test" in flags


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
def test_compile_result_fields():
    r = CompileResult(success=True, used_openmp=True, used_openmp_link=True)
    assert r.success is True
    assert r.used_openmp is True
    assert r.used_openmp_link is True
    assert r.output_path is None
    assert r.error is None


# ---------------------------------------------------------------------------
# compile_jlinalg smoke test — subprocess.run monkeypatched to return success.
# No real compilation runs; Wave 4 validates end-to-end.
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


@pytest.mark.tier0
def test_compile_jlinalg_smoke_success(monkeypatch, tmp_path):
    """Construct a compile_jlinalg call with stub paths; assert CompileResult
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
        "jamma._build_support.compile_and_link.subprocess.run",
        _fake_run,
    )

    # Create empty source files so compile_jlinalg's file existence checks
    # (if any) don't fail; the real cc_cmd is never invoked.
    src_a = tmp_path / "platform.c"
    src_a.write_text("// stub\n")
    src_b = tmp_path / "eigh.c"
    src_b.write_text("// stub\n")
    out = tmp_path / "out.so"

    result = compile_jlinalg(
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


@pytest.mark.tier0
def test_compile_jlinalg_compile_failure_triggers_omp_retry(monkeypatch, tmp_path):
    """When the first compile fails WITH omp_compile active, compile_jlinalg
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
        "jamma._build_support.compile_and_link.subprocess.run",
        _fake_run,
    )

    retry_reasons: list[str] = []

    src = tmp_path / "platform.c"
    src.write_text("// stub\n")
    out = tmp_path / "out.so"

    result = compile_jlinalg(
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


@pytest.mark.tier0
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
        "jamma._build_support.compile_and_link.subprocess.run",
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

    result = compile_jlinalg(
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
