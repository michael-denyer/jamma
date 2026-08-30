"""Tests for the JAMMA_SANITIZE env-var override and the LAPACK extra-flags
forwarding through ``resolve_cflags_for`` and ``compile_jlinalg``.

Covers the sanitizer flag injection seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jamma._build_support.compile_and_link import (
    BASE_CFLAGS,
    LAPACK_CFLAGS,
    apply_sanitizer_overrides,
    compile_jlinalg,
    resolve_cflags_for,
)

pytestmark = pytest.mark.tier0


class _FakeCompleted:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = ""


# ---------------------------------------------------------------------------
# apply_sanitizer_overrides — env-var driven flag injection
# ---------------------------------------------------------------------------


def test_no_sanitizer_returns_inputs_unchanged(monkeypatch):
    """JAMMA_SANITIZE unset: helper returns inputs verbatim plus empty lapack."""
    monkeypatch.delenv("JAMMA_SANITIZE", raising=False)
    cflags, link_flags, lapack_cflags = apply_sanitizer_overrides(["-foo"], ["-bar"])
    assert cflags == ["-foo"]
    assert link_flags == ["-bar"]
    assert lapack_cflags == []


def test_none_inputs_normalize_to_empty_lists(monkeypatch):
    """Passing None for either input must normalise to [], not propagate None."""
    monkeypatch.delenv("JAMMA_SANITIZE", raising=False)
    cflags, link_flags, lapack_cflags = apply_sanitizer_overrides(None, None)
    assert cflags == []
    assert link_flags == []
    assert lapack_cflags == []


@pytest.mark.parametrize("value", ["", "  ", "\t"])
def test_empty_sanitizer_treated_as_unset(monkeypatch, value):
    """Empty / whitespace-only JAMMA_SANITIZE: same behaviour as unset."""
    monkeypatch.setenv("JAMMA_SANITIZE", value)
    cflags, link_flags, lapack_cflags = apply_sanitizer_overrides(["-foo"], ["-bar"])
    assert cflags == ["-foo"]
    assert link_flags == ["-bar"]
    assert lapack_cflags == []


def test_zero_sanitizer_treated_as_unset(monkeypatch):
    """JAMMA_SANITIZE=0: same behaviour as unset (F3).

    Before F3, this fell through to ``-fsanitize=0`` — clang and gcc both
    reject "0" as a sanitizer name, so setting JAMMA_SANITIZE=0 (the
    documented way to turn a JAMMA_* toggle off) broke the C-extension
    build outright. Every JAMMA_* toggle shares presence-based truthiness
    (jamma.core.constants.env_flag): "" and "0" are off.
    """
    monkeypatch.setenv("JAMMA_SANITIZE", "0")
    cflags, link_flags, lapack_cflags = apply_sanitizer_overrides(["-foo"], ["-bar"])
    assert cflags == ["-foo"]
    assert link_flags == ["-bar"]
    assert lapack_cflags == []


def test_address_undefined_appends_sanitizer_flags(monkeypatch):
    """address,undefined: cflags get -fsanitize=address,undefined,
    -fno-omit-frame-pointer, -O1; link gets the same -fsanitize=...; lapack
    cflags equal the san_cflags list (caller decides where to splice them).
    """
    monkeypatch.setenv("JAMMA_SANITIZE", "address,undefined")
    cflags, link_flags, lapack_cflags = apply_sanitizer_overrides(
        ["-DUSER"], ["-luser"]
    )
    assert cflags == [
        "-DUSER",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-O1",
    ]
    assert link_flags == ["-luser", "-fsanitize=address,undefined"]
    assert lapack_cflags == [
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-O1",
    ]


def test_address_only(monkeypatch):
    """JAMMA_SANITIZE=address (no comma): -fsanitize=address only."""
    monkeypatch.setenv("JAMMA_SANITIZE", "address")
    cflags, link_flags, lapack_cflags = apply_sanitizer_overrides([], [])
    assert "-fsanitize=address" in cflags
    assert "-fsanitize=address,undefined" not in cflags
    assert "-fno-omit-frame-pointer" in cflags
    assert "-O1" in cflags
    assert link_flags == ["-fsanitize=address"]
    assert lapack_cflags[0] == "-fsanitize=address"


def test_lapack_cflags_returned_as_independent_list(monkeypatch):
    """Mutating the returned lapack_cflags must not leak into a subsequent
    call's cflags (defensive copy via list() in the helper).
    """
    monkeypatch.setenv("JAMMA_SANITIZE", "address")
    _c, _l, lapack_first = apply_sanitizer_overrides([], [])
    lapack_first.append("MUTATED")
    cflags2, _l2, lapack_second = apply_sanitizer_overrides([], [])
    assert "MUTATED" not in cflags2
    assert "MUTATED" not in lapack_second


# ---------------------------------------------------------------------------
# resolve_cflags_for — extra_lapack_cflags integration
# ---------------------------------------------------------------------------


def test_resolve_cflags_lapack_path_accepts_extra_lapack_cflags():
    """LAPACK source + extra_lapack_cflags: returned list starts with
    LAPACK_CFLAGS, then the extras, then -I includes. Trailing -O1 wins
    over LAPACK_CFLAGS' -O2 (last -O on the command line).
    """
    src = Path("/src/eigh.c")
    flags = resolve_cflags_for(
        src,
        lapack_source_set={"/src/eigh.c"},
        include_dirs=["/usr/include"],
        extra_lapack_cflags=["-fsanitize=address", "-O1"],
    )
    # First slice equals LAPACK_CFLAGS in order.
    assert flags[: len(LAPACK_CFLAGS)] == list(LAPACK_CFLAGS)
    # Extras present
    assert "-fsanitize=address" in flags
    # The last -O flag is -O1 (trailing -O wins).
    o_flags = [f for f in flags if f.startswith("-O")]
    assert o_flags[-1] == "-O1"
    # -I is at the end.
    assert flags[-1] == "-I/usr/include"


def test_resolve_cflags_lapack_path_no_extra_preserves_existing_behavior():
    """Omitting extra_lapack_cflags: result equals [*LAPACK_CFLAGS, *includes]."""
    src = Path("/src/eigh.c")
    flags = resolve_cflags_for(
        src,
        lapack_source_set={"/src/eigh.c"},
        include_dirs=["/usr/include"],
    )
    assert flags == [*LAPACK_CFLAGS, "-I/usr/include"]


def test_resolve_cflags_lapack_path_extra_lapack_none_equivalent_to_omit():
    """extra_lapack_cflags=None must behave identically to omitting it."""
    src = Path("/src/eigh.c")
    a = resolve_cflags_for(
        src,
        lapack_source_set={"/src/eigh.c"},
        include_dirs=[],
        extra_lapack_cflags=None,
    )
    b = resolve_cflags_for(
        src,
        lapack_source_set={"/src/eigh.c"},
        include_dirs=[],
    )
    assert a == b


def test_resolve_cflags_baseline_path_unaffected_by_extra_lapack_cflags():
    """Non-LAPACK source must NOT receive extra_lapack_cflags — they belong
    to the LAPACK branch only.
    """
    src = Path("/src/platform.c")
    flags = resolve_cflags_for(
        src,
        lapack_source_set={"/src/eigh.c"},
        include_dirs=[],
        extra_cflags=["-DUSER"],
        extra_lapack_cflags=["-fsanitize=address"],
    )
    assert "-fsanitize=address" not in flags
    # extra_cflags still spliced before -fno-finite-math-only on the BASE path.
    assert "-DUSER" in flags
    splice_idx = list(BASE_CFLAGS).index("-fno-finite-math-only")
    assert flags[:splice_idx] == list(BASE_CFLAGS[:splice_idx])


# ---------------------------------------------------------------------------
# compile_jlinalg — forwarding of extra_lapack_cflags through _compile_sources
# ---------------------------------------------------------------------------


def test_compile_jlinalg_forwards_extra_lapack_cflags(monkeypatch, tmp_path):
    """A compile_jlinalg call with extra_lapack_cflags must propagate the
    extras into the LAPACK source's compile command line; baseline sources
    must NOT receive them.
    """
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "-o" in cmd:
            out_idx = cmd.index("-o") + 1
            if out_idx < len(cmd):
                Path(cmd[out_idx]).write_bytes(b"")
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(
        "jamma._build_support.build_execution.subprocess.run",
        _fake_run,
    )

    src_base = tmp_path / "platform.c"
    src_base.write_text("// stub\n")
    src_lapack = tmp_path / "eigh.c"
    src_lapack.write_text("// stub\n")
    out = tmp_path / "out.so"

    result = compile_jlinalg(
        sources=[src_base, src_lapack],
        lapack_sources=[src_lapack],
        include_dirs=[],
        cc_cmd="cc",
        cc_extra=[],
        omp_compile=[],
        omp_link=[],
        ldflags=[],
        output=out,
        tmp_dir=tmp_path / "objs",
        extra_lapack_cflags=["-fsanitize=address", "-O1"],
    )

    assert result.success is True
    # Find the compile call for the LAPACK source.
    lapack_compile = next(c for c in calls if str(src_lapack) in c)
    assert "-fsanitize=address" in lapack_compile
    assert "-O1" in lapack_compile
    # Baseline source must NOT carry extra_lapack_cflags.
    base_compile = next(c for c in calls if str(src_base) in c)
    assert "-fsanitize=address" not in base_compile
