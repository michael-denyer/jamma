"""Self-tests for the tier-marker enforcement gate in conftest.py.

The gate (``pytest_collection_modifyitems`` in conftest.py) is a meta-rule:
every test file in this suite must declare a tier marker. If the gate
silently fails-open (e.g. a future refactor inverts the predicate, swaps
the marker check for ``True``, or wraps the raise in ``contextlib.suppress``),
unmarked tests would silently re-enter the default CI run.

These tests use pytest's ``pytester`` fixture to spin up an isolated
sub-pytest with a stub conftest that mirrors the real gate's logic, and
assert it fails collection on missing markers and succeeds on present ones.
"""

from __future__ import annotations

import textwrap

import pytest

pytestmark = pytest.mark.tier0

pytest_plugins = ["pytester"]


_GATE_CONFTEST = textwrap.dedent(
    '''
    """Stub conftest mirroring the real tier-marker gate from
    ``tests/conftest.py``. Kept in-line so this self-test is robust to
    refactors of the real conftest's helper imports.
    """
    from __future__ import annotations

    from pathlib import Path

    import pytest

    _REQUIRED_TIER_MARKERS = frozenset(
        {"tier0", "tier1", "tier2", "slow", "benchmark"}
    )
    _TIER_MARKER_EXEMPT_FILES: frozenset[str] = frozenset()


    def pytest_collection_modifyitems(config, items):
        if hasattr(config, "workerinput"):
            return
        files_with_marker: dict[str, bool] = {}
        for item in items:
            path = str(item.path) if hasattr(item, "path") else str(item.fspath)
            has_required = any(
                m.name in _REQUIRED_TIER_MARKERS for m in item.iter_markers()
            )
            files_with_marker[path] = (
                files_with_marker.get(path, False) or has_required
            )
        missing = [
            path
            for path, ok in files_with_marker.items()
            if not ok and Path(path).name not in _TIER_MARKER_EXEMPT_FILES
        ]
        if missing:
            listing = "\\n  ".join(sorted(Path(p).name for p in missing))
            raise pytest.UsageError(
                "The following test files have no tier marker "
                f"(tier0/tier1/tier2/slow/benchmark):\\n  {listing}"
            )
    '''
)


_INI = textwrap.dedent(
    """
    [pytest]
    markers =
        tier0: fast
        tier1: parity
        tier2: scale
        slow: independent
        benchmark: pytest-benchmark
    """
)


def test_gate_fails_when_a_file_has_no_tier_marker(pytester: pytest.Pytester) -> None:
    """A test file with zero tier markers must trigger a UsageError."""
    pytester.makeini(_INI)
    pytester.makeconftest(_GATE_CONFTEST)
    pytester.makepyfile(
        test_unmarked="""
        def test_does_a_thing():
            assert True
        """,
    )

    result = pytester.runpytest_subprocess("--collect-only")

    assert result.ret != 0, (
        "Gate should have failed collection on a file without tier markers"
    )
    result.stderr.fnmatch_lines(["*test_unmarked.py*"])
    result.stderr.fnmatch_lines(["*tier0/tier1/tier2/slow/benchmark*"])


def test_gate_passes_when_module_has_pytestmark(pytester: pytest.Pytester) -> None:
    """A file declaring ``pytestmark = pytest.mark.tier0`` is accepted."""
    pytester.makeini(_INI)
    pytester.makeconftest(_GATE_CONFTEST)
    pytester.makepyfile(
        test_marked_module="""
        import pytest
        pytestmark = pytest.mark.tier0

        def test_does_a_thing():
            assert True
        """,
    )

    result = pytester.runpytest_subprocess("--collect-only")
    assert result.ret == 0, (
        "Gate should pass when module pytestmark provides a tier marker"
    )


def test_gate_passes_when_each_test_marked(pytester: pytest.Pytester) -> None:
    """Per-test markers also satisfy the gate (no module-level marker required)."""
    pytester.makeini(_INI)
    pytester.makeconftest(_GATE_CONFTEST)
    pytester.makepyfile(
        test_per_test="""
        import pytest

        @pytest.mark.tier1
        def test_one():
            assert True

        @pytest.mark.slow
        def test_two():
            assert True
        """,
    )

    result = pytester.runpytest_subprocess("--collect-only")
    assert result.ret == 0


def test_gate_lists_every_unmarked_file(pytester: pytest.Pytester) -> None:
    """If multiple files lack markers, all of them appear in the error."""
    pytester.makeini(_INI)
    pytester.makeconftest(_GATE_CONFTEST)
    pytester.makepyfile(
        test_unmarked_a="""
        def test_a():
            assert True
        """,
        test_unmarked_b="""
        def test_b():
            assert True
        """,
        test_marked_c="""
        import pytest
        pytestmark = pytest.mark.tier0
        def test_c():
            assert True
        """,
    )

    result = pytester.runpytest_subprocess("--collect-only")
    assert result.ret != 0
    result.stderr.fnmatch_lines(["*test_unmarked_a.py*"])
    result.stderr.fnmatch_lines(["*test_unmarked_b.py*"])
    # The marked file must NOT appear in the failure listing.
    full_stderr = "\n".join(result.errlines)
    assert "test_marked_c.py" not in full_stderr


def test_gate_rejects_unrelated_marker(pytester: pytest.Pytester) -> None:
    """Markers outside the tier set don't satisfy the gate."""
    pytester.makeini(
        textwrap.dedent(
            """
            [pytest]
            markers =
                tier0: fast
                tier1: parity
                tier2: scale
                slow: independent
                benchmark: pytest-benchmark
                custom_unrelated: something else
            """
        )
    )
    pytester.makeconftest(_GATE_CONFTEST)
    pytester.makepyfile(
        test_wrong_marker="""
        import pytest
        pytestmark = pytest.mark.custom_unrelated
        def test_x():
            assert True
        """,
    )

    result = pytester.runpytest_subprocess("--collect-only")
    assert result.ret != 0, "An unrelated marker must not satisfy the tier-marker gate"
    result.stderr.fnmatch_lines(["*test_wrong_marker.py*"])
