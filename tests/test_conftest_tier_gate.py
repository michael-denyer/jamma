"""Self-tests for the tier-marker enforcement gate in conftest.py.

The gate is a meta-rule: every test file in this suite must declare a tier
marker. If the gate silently fails-open (e.g. a future refactor inverts the
predicate, swaps the marker check for ``True``, or wraps the raise in
``contextlib.suppress``), unmarked tests would silently re-enter the default
CI run.

The gate is implemented as **source-parsing** in ``pytest_configure``
(not ``pytest_collection_modifyitems``) because xdist forks workers AFTER
``pytest_configure`` and the controller's items list is empty under
``-n``. The collection-based design that preceded this one was empirically
a no-op when ``-n 3`` (the default in pyproject) was active. See
``test_gate_fires_under_xdist`` for the regression check.

These tests exercise the helper functions directly with synthetic ASTs and
add one ``pytester`` subprocess test that runs the gate under ``-n 2`` to
prove the xdist path works.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from tests.conftest import (
    _enforce_tier_markers,
    _file_declares_tier_marker,
    _module_level_marker_names,
    _per_test_marker_names,
)

pytestmark = pytest.mark.tier0

pytest_plugins = ["pytester"]


def _parse(src: str) -> ast.Module:
    return ast.parse(textwrap.dedent(src))


class TestModuleLevelMarkerNames:
    def test_single_marker(self) -> None:
        tree = _parse(
            """
            import pytest
            pytestmark = pytest.mark.tier0
            """
        )
        assert _module_level_marker_names(tree) == {"tier0"}

    def test_list_of_markers(self) -> None:
        tree = _parse(
            """
            import pytest
            pytestmark = [pytest.mark.tier1, pytest.mark.slow]
            """
        )
        assert _module_level_marker_names(tree) == {"tier1", "slow"}

    def test_marker_with_args(self) -> None:
        """``pytest.mark.foo(...)`` parametrised marker still recognised."""
        tree = _parse(
            """
            import pytest
            pytestmark = pytest.mark.skipif(True, reason="x")
            """
        )
        assert _module_level_marker_names(tree) == {"skipif"}

    def test_no_pytestmark(self) -> None:
        tree = _parse("def test_x(): pass\n")
        assert _module_level_marker_names(tree) == set()

    def test_unrelated_assignment(self) -> None:
        tree = _parse("pytestmark_unused = 1\n")
        assert _module_level_marker_names(tree) == set()


class TestPerTestMarkerNames:
    def test_function_decorator(self) -> None:
        tree = _parse(
            """
            import pytest

            @pytest.mark.tier1
            def test_x():
                pass
            """
        )
        assert "tier1" in _per_test_marker_names(tree)

    def test_class_decorator(self) -> None:
        tree = _parse(
            """
            import pytest

            @pytest.mark.benchmark
            class TestY:
                def test_a(self): pass
            """
        )
        assert "benchmark" in _per_test_marker_names(tree)

    def test_call_form_decorator(self) -> None:
        """``@pytest.mark.skipif(...)`` decorator with args."""
        tree = _parse(
            """
            import pytest

            @pytest.mark.skipif(True, reason="r")
            def test_x():
                pass
            """
        )
        assert "skipif" in _per_test_marker_names(tree)

    def test_no_decorators(self) -> None:
        tree = _parse("def test_x(): pass\n")
        assert _per_test_marker_names(tree) == set()


class TestFileDeclaresTierMarker:
    def _write(self, tmp_path, src: str):
        path = tmp_path / "test_target.py"
        path.write_text(textwrap.dedent(src))
        return path

    def test_module_pytestmark_passes(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            """
            import pytest
            pytestmark = pytest.mark.tier0
            def test_a(): pass
            """,
        )
        assert _file_declares_tier_marker(path)

    def test_per_test_marker_passes(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            """
            import pytest

            @pytest.mark.tier1
            def test_a(): pass
            """,
        )
        assert _file_declares_tier_marker(path)

    def test_no_markers_fails(self, tmp_path) -> None:
        path = self._write(
            tmp_path,
            """
            def test_a(): pass
            """,
        )
        assert not _file_declares_tier_marker(path)

    def test_unrelated_marker_fails(self, tmp_path) -> None:
        """``custom`` is not in the required-tier set."""
        path = self._write(
            tmp_path,
            """
            import pytest

            @pytest.mark.custom
            def test_a(): pass
            """,
        )
        assert not _file_declares_tier_marker(path)

    def test_syntax_error_treated_as_missing(self, tmp_path) -> None:
        """Unparsable source surfaces via the gate, not a swallowed exception."""
        path = tmp_path / "test_broken.py"
        path.write_text("def test_a(:\n    pass\n")
        assert not _file_declares_tier_marker(path)


class TestEnforceTierMarkersInProcess:
    def test_real_suite_passes(self) -> None:
        """The actual jamma test suite must satisfy the gate.

        If this fails, a real test file is missing a tier marker — fix the
        file rather than the gate. This is also a smoke test that
        ``_enforce_tier_markers`` walks ``tests/`` correctly.
        """
        _enforce_tier_markers()


class TestGateUnderXdist:
    """End-to-end regression: the real gate must fire under ``-n N`` (xdist).

    The previous collection-based gate silently no-op'd because xdist's
    controller hook receives an empty ``items`` list. This test imports
    the *real* ``_enforce_tier_markers`` from ``tests.conftest`` into a
    pytester sub-session and runs it under ``-n 2`` — so a regression
    that re-introduces the xdist hole would make this test fail rather
    than pass against a parallel stub.
    """

    # Stub conftest delegating to the real gate. Pytester runs in a
    # tmpdir without our pyproject, so `tests.conftest` won't import as
    # a package — instead we point the conftest at the real module file
    # via importlib.util.
    _GATE_CONFTEST = textwrap.dedent(
        f'''
        """Stub conftest delegating to the real ``_enforce_tier_markers``."""
        from __future__ import annotations

        import importlib.util
        from pathlib import Path

        import pytest

        _REAL_CONFTEST = Path({str(Path(__file__).parent / "conftest.py")!r})


        def _load_real_conftest():
            spec = importlib.util.spec_from_file_location(
                "_jamma_real_conftest", _REAL_CONFTEST
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod


        def pytest_configure(config):
            # Mirror the real conftest's worker guard.
            if hasattr(config, "workerinput"):
                return
            real = _load_real_conftest()
            # The real ``_enforce_tier_markers`` walks ``_TESTS_DIR``
            # (the directory containing the real conftest). For this
            # self-test we want it to walk the pytester rootpath so the
            # synthetic test files we just created are what's audited.
            # Repoint it temporarily.
            real._TESTS_DIR = Path(str(config.rootpath))
            real._enforce_tier_markers()
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

    def test_gate_fires_under_xdist(self, pytester: pytest.Pytester) -> None:
        """Unmarked file must fail the gate even with ``-n 2``."""
        pytester.makeini(self._INI)
        pytester.makeconftest(self._GATE_CONFTEST)
        pytester.makepyfile(
            test_unmarked="""
            def test_does_a_thing():
                assert True
            """,
        )
        result = pytester.runpytest_subprocess("-n", "2")
        assert result.ret != 0, (
            "Gate should fail under -n 2; the previous collection-based "
            "gate silently passed."
        )
        # Either stderr or stdout depending on xdist's plumbing — search both.
        combined = "\n".join([*result.errlines, *result.outlines])
        assert "test_unmarked.py" in combined
        assert "no tier marker" in combined

    def test_gate_passes_when_marked_under_xdist(
        self, pytester: pytest.Pytester
    ) -> None:
        """Counter-test: a marked file passes under ``-n 2``."""
        pytester.makeini(self._INI)
        pytester.makeconftest(self._GATE_CONFTEST)
        pytester.makepyfile(
            test_marked="""
            import pytest
            pytestmark = pytest.mark.tier0

            def test_a(): pass
            def test_b(): pass
            """,
        )
        result = pytester.runpytest_subprocess("-n", "2")
        assert result.ret == 0, (
            f"Gate should pass on marked file. ret={result.ret}\n"
            f"stdout={result.outlines!r}\nstderr={result.errlines!r}"
        )
